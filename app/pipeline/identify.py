"""Stage 4: resolve each spine reading to a catalogue work, and an edition where the spine supports it.

Sources: Open Library (no key) and Google Books (when GOOGLE_BOOKS_API_KEY is set). A match needs the
printed title to agree with the catalogue title and, when an author is printed, the author too.
The edition (and so the ISBN and physical dimensions) is chosen only when the printed publisher or
volume number points to it; otherwise the book is identified at work level with no ISBN.

When the spine is clearly legible but no catalogue has the book, the title is reported as printed,
with lower confidence and a review-queue reason, rather than discarded.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from rapidfuzz import fuzz

from app.pipeline.http import CachedHTTP
from app.pipeline.read import SpineRead

OL = "https://openlibrary.org"
GB = "https://www.googleapis.com/books/v1/volumes"
ACCEPT_TITLE = 85
IDENTIFIED_MIN_CONF = 0.55
# A title known only from the spine (no catalogue record) needs two agreeing reads: OCR slips such as
# "CS5" -> "CSS" are consistent within one crop but rarely repeat across two.
PRINTED_ONLY_MIN_CONF = 0.6


@dataclass
class Edition:
    source: str  # openlibrary | googlebooks
    url: str
    title: str
    authors: list[str]
    publisher: str = ""
    year: str = ""
    isbn: str | None = None
    height_cm: float | None = None
    width_cm: float | None = None
    thickness_cm: float | None = None
    pages: int | None = None
    volume: str = ""


@dataclass
class Identification:
    track_id: str
    status: str  # identified | unidentified
    title: str | None = None
    author: str | None = None
    edition: str | None = None
    isbn: str | None = None
    publisher: str | None = None
    year: str | None = None
    id_confidence: float = 0.0
    catalogue_url: str | None = None
    dims_cm: dict[str, float | None] = field(default_factory=dict)  # from the matched edition only
    reasons: list[str] = field(default_factory=list)
    retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_surname(author: str) -> str:
    """'Zakhour · Kannan · Gallardo' -> 'zakhour'; 'Benedetti & Cranley' -> 'benedetti'; 'Bernard Williams' -> 'williams'."""
    first = re.split(r"\s*(?:[·&,/;|]|\band\b|\bund\b)\s*", author or "")[0]
    words = norm(first).split()
    return words[-1] if words else ""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(hrsg\.?\)|\bhrsg\b|\beds?\.?\b|\bed\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_score(read_title: str, cand_title: str) -> float:
    a, b = norm(read_title), norm(cand_title)
    if not a or not b:
        return 0.0
    # A spine often carries a short form of a long title ("Griechen" for "Die Natur und die Griechen"),
    # but a short read must not match any long title that happens to contain it.
    main = norm(cand_title.split(":")[0])
    full = max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b))
    score = float(max(full, fuzz.ratio(a, main)))
    # Short titles differ by one token that matters ("HTML5" vs "HTML" is 89% similar as characters):
    # below 15 characters only an exact word match counts as a full match.
    if min(len(a), len(main)) < 15 and set(a.split()) != set(main.split()) and set(a.split()) != set(b.split()):
        score = min(score, 80.0)
    return score


def author_score(read_author: str, cand_authors: list[str]) -> float:
    if not read_author:
        return 50.0  # unknown: neither supports nor contradicts
    ra = norm(read_author)
    best = 0.0
    for a in cand_authors:
        na = norm(a)
        surname = na.split()[-1] if na else ""
        best = max(best, fuzz.token_set_ratio(ra, na), 100.0 if surname and surname in ra.split() else 0.0)
    return best


def roman_to_int(v: str) -> str:
    v = v.strip().upper().rstrip(".")
    table = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10"}
    return table.get(v, v)


VOLUME_TAIL = re.compile(r"[\s,.:;-]+(?:(?:bd|band|vol|volume|teil|part)\.?\s*)?([IVX]{1,4}|\d{1,2})\.?$", re.I)


def split_volume(title: str) -> tuple[str, str]:
    """'Philosophen der Antike II' -> ('Philosophen der Antike', '2')."""
    m = VOLUME_TAIL.search(title or "")
    if not m:
        return title, ""
    return title[: m.start()].strip(), roman_to_int(m.group(1))


def nice_case(s: str) -> str:
    return s.title() if s and s.isupper() else s


def parse_dims(text: str | None) -> tuple[float | None, float | None, float | None]:
    """'24 x 16 x 2 centimeters' or '7.4 x 4.5 x 1.1 inches' -> (height, width, thickness) in cm."""
    if not text:
        return None, None, None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) < 2:
        return None, None, None
    factor = 2.54 if "inch" in text.lower() or '"' in text else (0.1 if "mm" in text.lower() else 1.0)
    vals = sorted((n * factor for n in nums[:3]), reverse=True)
    h, w = vals[0], vals[1]
    t = vals[2] if len(vals) > 2 else None
    return round(h, 2), round(w, 2), (round(t, 2) if t else None)


def _gb_dim(v: str | None) -> float | None:
    if not v:
        return None
    m = re.match(r"([\d.]+)\s*cm", v)
    return float(m.group(1)) if m else None


class Catalogue:
    def __init__(self, http: CachedHTTP, google_books_key: str = "", concurrency: int = 8):
        self.http = http
        self.gb_key = google_books_key
        self.sem = asyncio.Semaphore(concurrency)
        self.failed_lookups = 0

    async def _get(self, url: str, params: dict) -> dict[str, Any] | None:
        """GET with retries. Returns None after repeated failure and counts it, so a failed lookup
        is reported as a failure rather than mistaken for "this book is not in the catalogue"."""
        async with self.sem:
            for attempt in range(3):
                try:
                    return await self.http.request_json("identify", "GET", url, params=params)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        return {"data": {}, "retrieved_at": None, "url": url}
                except Exception:
                    pass
                await asyncio.sleep(1.5 * 2 ** attempt)
        self.failed_lookups += 1
        return None

    async def search(self, title: str, author: str) -> tuple[list[Edition], str | None, bool]:
        """Candidates from all sources, the earliest retrieval time, and whether a lookup failed.

        Every matching work becomes a work-level candidate (title + authors, no ISBN), which is what
        the ambiguity check needs. Editions (ISBN, publisher, dimensions) are fetched only for the
        work whose title matches best: one extra request per book instead of one per candidate work.
        """
        surname = first_surname(author)
        cands: list[Edition] = []
        retrieved = []
        failed = False
        # Open Library hands queries to a Solr parser: a colon or bracket in a title ("JavaScript: The
        # Good Parts") silently returns nothing. Strip query syntax before searching.
        title = re.sub(r"[:()\[\]{}\"!^~*?\\/+|&]", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        # The general query (title words + author surname) finds books the title= field misses
        # (tested: "JavaScript The Good Parts" + crockford). Fall back to title= alone if it finds nothing.
        fields = "key,title,subtitle,author_name"
        res = await self._get(f"{OL}/search.json", {"q": f"{title} {surname}".strip(), "limit": 8, "fields": fields})
        if res is not None and not res.get("data", {}).get("docs"):
            res = await self._get(f"{OL}/search.json", {"title": title, "limit": 8, "fields": fields})
        if res is None:
            failed = True
        works = (res or {}).get("data", {}).get("docs", []) if res else []
        if res and res.get("retrieved_at"):
            retrieved.append(res["retrieved_at"])
        for w in works:
            if w.get("subtitle"):
                w["title"] = f"{w.get('title', '')}: {w['subtitle']}"
            cands.append(Edition("openlibrary", f"{OL}{w['key']}", w.get("title", ""), w.get("author_name", [])))
        ranked = sorted(works, key=lambda w: title_score(title, w.get("title", "")), reverse=True)
        for w in ranked[:1]:
            eds = await self._get(f"{OL}{w['key']}/editions.json", {"limit": 50})
            if not eds:
                failed = True
                continue
            if eds.get("retrieved_at"):
                retrieved.append(eds["retrieved_at"])
            for e in eds["data"].get("entries", []):
                h, wd, t = parse_dims(e.get("physical_dimensions"))
                isbn = (e.get("isbn_13") or e.get("isbn_10") or [None])[0]
                cands.append(Edition(
                    source="openlibrary", url=f"{OL}{e.get('key', w['key'])}", title=w.get("title", ""),
                    authors=w.get("author_name", []), publisher=", ".join(e.get("publishers", [])[:2]),
                    year=str(e.get("publish_date", "")), isbn=isbn, height_cm=h, width_cm=wd, thickness_cm=t,
                    pages=e.get("number_of_pages"), volume=" ".join(filter(None, [e.get("volume_number") and str(e.get("volume_number")), e.get("edition_name"), e.get("title", "")])),
                ))
        if self.gb_key:
            q = f'intitle:"{title}"' + (f" inauthor:{surname}" if surname else "")
            gb = await self._get(GB, {"q": q, "maxResults": 10, "printType": "books", "key": self.gb_key})
            if gb:
                retrieved.append(gb["retrieved_at"])
                for it in gb["data"].get("items", []) or []:
                    v = it.get("volumeInfo", {})
                    ids = {i["type"]: i["identifier"] for i in v.get("industryIdentifiers", [])}
                    d = v.get("dimensions") or {}
                    cands.append(Edition(
                        source="googlebooks", url=v.get("infoLink") or f"https://books.google.com/books?id={it.get('id')}",
                        title=" ".join(filter(None, [v.get("title"), v.get("subtitle") and f": {v['subtitle']}"])) if v.get("subtitle") else v.get("title", ""),
                        authors=v.get("authors", []), publisher=v.get("publisher", ""), year=v.get("publishedDate", ""),
                        isbn=ids.get("ISBN_13") or ids.get("ISBN_10"), height_cm=_gb_dim(d.get("height")),
                        width_cm=_gb_dim(d.get("width")), thickness_cm=_gb_dim(d.get("thickness")), pages=v.get("pageCount"),
                        volume=v.get("title", ""),
                    ))
            else:
                failed = True
        return cands, (min(retrieved) if retrieved else None), failed


def score_candidate(read: SpineRead, c: Edition) -> tuple[float, float, float]:
    return title_score(read.title or read.printed_text, c.title), author_score(read.author, c.authors), (
        100.0 if read.publisher and c.publisher and fuzz.partial_ratio(norm(read.publisher), norm(c.publisher)) >= 85 else 0.0
    )


FILLER = {"the", "a", "an", "and", "of", "in", "for", "to", "on", "with", "by", "edition", "ed", "new", "revised", "vol",
          "volume", "st", "nd", "rd", "th", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
          "ninth", "tenth", "ebook", "available", "hrsg", "eds"}


def extra_words(read: SpineRead, c: Edition) -> set[str]:
    """Meaningful printed words that are not explained by the candidate's title, author or publisher."""
    printed = set(norm(read.printed_text).split())
    explained = set(norm(c.title).split()) | set(norm(read.author).split()) | set(norm(read.publisher).split())
    explained |= {w for a in c.authors for w in norm(a).split()}
    return {w for w in printed - explained - FILLER if not w.isdigit() and len(w) > 1}


def distinct_authorships(cands: list[Edition]) -> int:
    """Number of groups of candidates with no author surname in common. Catalogues list co-authors
    in different orders, so 'Mattmann, Zitting' and 'Zitting, Mattmann' are one authorship."""
    groups: list[set[str]] = []
    for c in cands:
        names = {norm(a).split()[-1] for a in c.authors if norm(a)}
        hit = [g for g in groups if g & names]
        merged = names.union(*hit) if hit else names
        groups = [g for g in groups if g not in hit] + [merged]
    return len(groups)


def _work_key(c: Edition) -> tuple[str, str]:
    return norm(split_volume(c.title)[0]), norm(c.authors[0]).split()[-1] if c.authors and norm(c.authors[0]) else ""


def choose(read: SpineRead, read_conf: float, cands: list[Edition], track_id: str, retrieved_at: str | None,
           lookup_failed: bool = False) -> Identification:
    vol = roman_to_int(read.volume) if read.volume and len(read.volume) <= 4 else ""
    scored = []
    for c in cands:
        ts, au, pub = score_candidate(read, c)
        _, cand_vol = split_volume(c.title)
        if vol and cand_vol and cand_vol != vol:
            continue  # a different volume of the same work
        if ts >= ACCEPT_TITLE and au >= 50:
            scored.append((ts, au, pub, c))
    short_title_match = False
    if not scored and read.title and read.author and len(norm(read.title).split()) <= 2:
        # Spines often print a short form ("Griechen" for "Die Natur und die Griechen"). Accept only
        # if the author matches strongly and exactly one of their works contains those words.
        words = set(norm(read.title).split())
        hits = {}
        for c in cands:
            if author_score(read.author, c.authors) >= 90 and words <= set(norm(c.title).split()):
                hits.setdefault(norm(c.title), (100.0, 100.0, 0.0, c))
        if len(hits) == 1:
            scored = list(hits.values())
            short_title_match = True
    # Without an author on the spine, two checks decide whether the work can be resolved at all. Both
    # run before the publisher filter: filtering first can remove the only competitor and leave an
    # unverified candidate looking unique.
    if scored and not read.author:
        best_ts = max(x[0] for x in scored)
        top = [x for x in scored if x[0] >= best_ts - 5]
        authors = distinct_authorships([x[3] for x in top if x[3].authors])
        if authors >= 2:
            return Identification(track_id, "unidentified", retrieved_at=retrieved_at, reasons=[
                f"spine shows {read.printed_text!r} without an author; {authors} different authors have a book "
                "with this title, so the work cannot be determined"])

    # A printed publisher rules out a work whose known editions all name a different publisher.
    if read.publisher and scored:
        groups: dict[tuple[str, str], list] = {}
        for x in scored:
            groups.setdefault(_work_key(x[3]), []).append(x)
        kept = []
        for g in groups.values():
            pubs = [x for x in g if x[3].publisher]
            if pubs and not any(x[2] > 0 for x in pubs):
                continue
            kept += g
        scored = kept

    title_printed = read.title or ""
    shown_title = (title_printed.title() if title_printed.isupper() else title_printed) if title_printed else None

    if not scored:
        ident = Identification(track_id, "unidentified", reasons=[], retrieved_at=retrieved_at)
        # Title as printed only with a printed author as second evidence: a series name ("Sams Teach
        # Yourself") otherwise yields a shortened title that matches no real book.
        if read.legibility == "clear" and title_printed and read.author:
            ident.status = "identified"
            ident.title = shown_title + (f" {read.volume}" if read.volume and vol.isdigit() and len(vol) <= 2 else "")
            ident.author = nice_case(read.author) or None
            ident.id_confidence = round(read_conf * 0.7, 2)
            ident.reasons.append("title as printed on the spine; " + ("catalogue lookup failed" if lookup_failed else "no catalogue record found"))
            if ident.id_confidence < PRINTED_ONLY_MIN_CONF:
                ident.status = "unidentified"
                ident.reasons[-1] = f"printed text {read.printed_text!r} read with low confidence; no catalogue match"
                ident.title = ident.author = None
        elif lookup_failed:
            ident.reasons.append("catalogue lookup failed; spine text kept for review")
        else:
            ident.reasons.append("spine not legible enough to identify" if not title_printed else f"printed text {read.printed_text!r} matches no catalogue record")
        return ident

    # Work = best title/author agreement. Edition = a candidate of that work whose publisher or volume
    # agrees with the spine; prefer one with physical dimensions.
    # Rank by title + author agreement, minus printed words the candidate leaves unexplained.
    scored.sort(key=lambda x: (x[0] + x[1] - 10 * len(extra_words(read, x[3])), x[2]), reverse=True)
    ts, au, _, work = scored[0]
    extra = extra_words(read, work)
    if not read.author and len(extra) >= 2 and not short_title_match:
        return Identification(track_id, "unidentified", retrieved_at=retrieved_at, reasons=[
            f"spine shows {read.printed_text!r}; the closest catalogue title {work.title!r} leaves "
            f"{sorted(extra)} unexplained, so it is probably a different book"])
    same_work = [s for s in scored if norm(s[3].title) == norm(work.title)]
    edition_hits = [s for s in same_work if s[2] > 0 or (vol and re.search(rf"\b({re.escape(read.volume)}|{vol})\b", s[3].volume or "", re.I))]
    edition = None
    if edition_hits:
        edition = max(edition_hits, key=lambda s: (s[3].height_cm is not None, s[3].isbn is not None))[3]
    match = (ts / 100) * (0.7 + 0.3 * au / 100)
    conf = round(min(1.0, read_conf * (0.6 + 0.4 * match)), 2)
    if short_title_match:
        conf = min(conf, 0.6)
    base, _ = split_volume(work.title)
    # Keep the spine's capitalisation when it is the same title; catalogues often store lower case.
    if read.title and title_score(read.title, base) >= 95 and not read.title.isupper():
        base = split_volume(read.title)[0]
    title = base + (f" {read.volume}" if read.volume and vol.isdigit() and len(vol) <= 2 else "")
    ident = Identification(
        track_id=track_id,
        status="identified" if conf >= IDENTIFIED_MIN_CONF else "unidentified",
        title=title, author=", ".join(work.authors[:2]) or (read.author or None),
        edition=f"{edition.publisher} {edition.year}".strip() if edition else None,
        isbn=edition.isbn if edition else None,
        publisher=edition.publisher if edition else (read.publisher or None),
        year=edition.year if edition else None,
        id_confidence=conf, catalogue_url=(edition or work).url,
        dims_cm={"height": edition.height_cm, "width": edition.width_cm, "thickness": edition.thickness_cm} if edition else {},
        retrieved_at=retrieved_at,
    )
    if short_title_match and ident.status == "identified":
        ident.reasons.append(f"spine shows a short title ({read.title!r}); matched the author's only work containing it")
    if ident.status == "unidentified":
        ident.reasons.append(f"weak match to {work.title!r} (confidence {conf})")
        ident.title = ident.author = ident.isbn = ident.edition = None
    elif not edition:
        ident.reasons.append("edition not determinable from the spine; identified at work level")
    return ident


async def identify(catalogue: Catalogue, reads: dict[str, tuple[SpineRead | None, float]]) -> dict[str, Identification]:
    async def one(track_id: str, read: SpineRead | None, conf: float) -> Identification:
        if read is None:
            return Identification(track_id, "unidentified", reasons=["spine not readable"])
        query = read.title or read.printed_text
        cands, retrieved, failed = await catalogue.search(query, read.author) if query else ([], None, False)
        return choose(read, conf, cands, track_id, retrieved, failed)

    results = await asyncio.gather(*(one(t, r, c) for t, (r, c) in reads.items()))
    return {r.track_id: r for r in results}
