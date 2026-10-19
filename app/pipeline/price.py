"""Stage 6: prices from retrievable sources. A language model never produces a price.

New (replacement) price: Google Shopping results via Serper for the policyholder's country, filtered
to listings whose title matches the book, median of the matches. Used value: listings marked used /
pre-owned / second-hand. If the local market has nothing, the same search runs in a fallback market
and the price is converted at the ECB rate of the day and labelled converted. If nothing is found
anywhere, the price stays empty and the line is excluded from totals.

Each price records the listing it came from (seller, URL), when the search was retrieved, and how
many listings agreed; every listing considered is kept in the stage output for audit.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from rapidfuzz import fuzz

from app.pipeline.http import CachedHTTP
from app.pipeline.identify import first_surname, norm
from app.pipeline.ledger import SERPER_USD_PER_QUERY

SERPER_SHOPPING = "https://google.serper.dev/shopping"
FX_URL = "https://api.frankfurter.dev/v1/latest"

CURRENCY_OF = {"IN": "INR", "US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "AU": "AUD", "CA": "CAD", "SG": "SGD", "AE": "AED"}
SYMBOLS = {"₹": "INR", "Rs": "INR", "Rs.": "INR", "$": "USD", "US$": "USD", "£": "GBP", "€": "EUR", "A$": "AUD", "C$": "CAD", "S$": "SGD", "AED": "AED"}
FALLBACK_MARKET = {"IN": "US", "US": "GB", "GB": "US"}
# Appraisal threshold per currency: anything above this is routed to a human, not auto-priced.
STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "for", "on", "with", "by", "at", "from", "is", "are", "you", "your"}
VOLUME = re.compile(r"\b(?:vol|volume|bk|book|part|level|class|grade|std)\s*(\d+)\b")
# "by Ash, Russell" or Amazon's "Title : Charman, Andrew: ..." (surname, first name)
CREDITED_AUTHOR = re.compile(r"\bby\s+[A-Z][\w.'-]+|:\s*[A-Z][\w'-]+,\s+[A-Z][\w.]*")
FORMAT_WORDS = {"book", "books", "hardcover", "hardback", "paperback", "edition"}
SHOP_PREFIX = re.compile(r"^\s*(?:buy\b|amazon\.(?:com|in|co\.uk)\s*:)\s*", re.I)
MIN_RANGE_LISTINGS = 4  # a price range from fewer comparable products is not a range
APPRAISAL_THRESHOLD = {"INR": 5000, "USD": 60, "GBP": 50, "EUR": 55}

NOT_THE_BOOK = re.compile(r"\b(kindle|e-?book|audio ?book|audible|summary of|study guide|workbook for|box ?set|combo|set of \d|lot of|poster|t-?shirt|mug|dvd|cd)\b", re.I)
OLD_EDITION = re.compile(r"\b(old|previous|older|outdated)\s+edition\b", re.I)
USED = re.compile(r"\b(used|pre-?owned|second[- ]?hand|old book|refurbished|like new|good condition|acceptable)\b", re.I)


@dataclass
class Listing:
    title: str
    seller: str
    url: str
    price: float
    currency: str
    used: bool
    market: str
    original_price: float | None = None
    original_currency: str | None = None


@dataclass
class PriceResult:
    amount: float | None = None
    currency: str = ""
    source: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    converted: bool = False
    original_amount: float | None = None
    original_currency: str | None = None
    n_listings: int = 0
    condition_assumed: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BookPrice:
    track_id: str
    replacement: PriceResult = field(default_factory=PriceResult)
    used: PriceResult = field(default_factory=PriceResult)
    appraisal_reason: str | None = None
    listings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "replacement": self.replacement.to_dict(), "used": self.used.to_dict(),
                "appraisal_reason": self.appraisal_reason, "listings": self.listings}


def parse_price(text: str | float | None, default_currency: str) -> tuple[float | None, str]:
    """'₹1,299.00' -> (1299.0, 'INR'); '$12.99' -> (12.99, 'USD'). Ranges take the lower bound."""
    if text is None:
        return None, default_currency
    if isinstance(text, (int, float)):
        return float(text), default_currency
    s = str(text).strip()
    cur = default_currency
    for sym in sorted(SYMBOLS, key=len, reverse=True):
        if s.startswith(sym) or f" {sym}" in s or s.endswith(sym):
            cur = SYMBOLS[sym]
            break
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None, cur
    return float(m.group(0).replace(",", "")), cur


def listing_matches(listing_title: str, book_title: str, author: str | None) -> bool:
    """Does a retailer listing title belong to this book?

    Learned from the demo sweep's price audit:
    - With no author, the whole title must match: "555 Questions & Answers" is a subset of
      "Freemasonry at a Glance: Answers to 555 Questions" but not the same book.
    - With an author, the listing must carry the surname unless the title is long and distinctive:
      "Transportation" by Lambert matched "Transportation" by Satorre, and a children's adaptation of
      "The Hunchback of Notre Dame" matched a generic edition.
    - Catalogue authors come as "Last, First" too: the surname is the part before the comma.
    - Numbers must agree: "Phonics First 6" is not "Phonics First: Bk. 1".
    - A title found only in the listing's series brackets is a different volume:
      "Nature (Child's First Library of Learning)" is not "A Child's First Library of Learning".
    """
    if NOT_THE_BOOK.search(listing_title):
        return False
    lt, bt = norm(listing_title), norm(book_title)
    if not bt:
        return False
    if fuzz.token_set_ratio(bt, lt) < 85:
        return False
    meaningful = [w for w in bt.split() if w not in STOPWORDS]
    words = lt.split()
    if not all(any(fuzz.ratio(w, x) >= 85 for x in words) for w in meaningful):
        return False  # "Search and Learn" is not "Search and Find"
    names = set(norm(author or "").split())
    if len([w for w in meaningful if w not in FORMAT_WORDS]) <= 2 and not any(
            fuzz.token_sort_ratio(_strip_format(bt), " ".join(w for w in seg.split() if w not in names)) >= 85
            for seg in listing_title_segments(listing_title)):
        return False  # a short title must be the listing's own title: "Transportation" is not "The Wonderful World of Transportation"
    numbers = set(re.findall(r"\d+", bt))
    if not numbers <= set(lt.split()) or any(n not in numbers for n in VOLUME.findall(lt)):
        return False
    if "(" in listing_title:
        before = set(norm(listing_title_head(listing_title, "(")).split()) - STOPWORDS
        if before and not before & set(bt.split()):
            return False
    surname = first_surname(author) if author else ""
    if not surname:
        main = norm(listing_title.split(":")[0].split("|")[0])
        return max(fuzz.token_sort_ratio(bt, main), fuzz.partial_ratio(bt, lt) if len(bt.split()) >= 4 else 0) >= 88
    if surname in lt.split():
        return True
    # Title alone is enough only for a long, specific title (4+ meaningful words; "the", "of" ... do not
    # count) and only when the listing credits no other author: "Life and Times in Ancient Greece" by
    # Ferris matched "...: Charman, Andrew: 978...".
    if CREDITED_AUTHOR.search(listing_title):
        return False
    return len(meaningful) >= 4 and fuzz.partial_ratio(bt, lt) >= 95


def listing_title_head(listing_title: str, stops: str = "(:|") -> str:
    """The listing's own title, before any series, subtitle or shop text ("Buy Nature (Series...)" -> "Nature")."""
    t = SHOP_PREFIX.sub("", SHOP_PREFIX.sub("", listing_title))
    return re.split("[" + re.escape(stops) + "]", t, maxsplit=1)[0]


def _strip_format(s: str) -> str:
    return " ".join(w for w in s.split() if w not in FORMAT_WORDS)


def listing_title_segments(listing_title: str) -> list[str]:
    """Candidate titles inside a listing: "Kingfisher Knowledge: Endangered Planet by Burnie, David"
    -> ["kingfisher knowledge", "endangered planet", ...], without shop text, authors or format words."""
    t = re.split(r"[(|]", SHOP_PREFIX.sub("", SHOP_PREFIX.sub("", listing_title)), maxsplit=1)[0]
    segs = [re.split(r"\s+by\s+", seg, maxsplit=1, flags=re.I)[0] for seg in re.split(r":|\s-\s", t)]
    return [x for x in (_strip_format(norm(seg)) for seg in segs) if x]


def several_books(listings: list["Listing"], book_title: str) -> bool:
    """The matches are different books sharing this title: it is a series name, not one book."""
    bt = norm(book_title)
    heads: list[str] = []
    for x in listings:
        h = norm(listing_title_head(x.title))
        if h and fuzz.token_set_ratio(bt, h) < 85 and all(fuzz.token_set_ratio(h, o) < 85 for o in heads):
            heads.append(h)
    return len(heads) >= 2


class FX:
    def __init__(self, http: CachedHTTP):
        self.http = http

    async def rate(self, frm: str, to: str) -> tuple[float, str, str] | None:
        """(rate, date, url) to convert 1 `frm` into `to`, ECB reference rates."""
        if frm == to:
            return 1.0, "", ""
        try:
            r = await self.http.request_json("price", "GET", FX_URL, params={"base": frm, "symbols": to}, max_age_days=1)
        except Exception:
            return None
        rate = r["data"].get("rates", {}).get(to)
        return (float(rate), r["data"].get("date", ""), r["url"]) if rate else None


class Shopping:
    label = "Google Shopping (Serper)"

    def __init__(self, http: CachedHTTP, api_key: str):
        self.http = http
        self.key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    async def search(self, query: str, market: str) -> tuple[list[Listing], str | None, str]:
        """Listings, retrieval time, and the search URL (a human-openable equivalent of the query)."""
        if not self.key:
            return [], None, ""
        body = {"q": query, "gl": market.lower(), "hl": "en", "num": 20}
        try:
            r = await self.http.request_json("price", "POST", SERPER_SHOPPING, json_body=body,
                                             headers={"X-API-KEY": self.key, "Content-Type": "application/json"},
                                             usd_per_call=SERPER_USD_PER_QUERY, max_age_days=30)
        except Exception:
            return [], None, ""
        cur = CURRENCY_OF.get(market.upper(), "USD")
        out = []
        for it in r["data"].get("shopping", []) or []:
            amount, c = parse_price(it.get("price"), cur)
            if amount is None or amount <= 0:
                continue
            title = it.get("title", "")
            out.append(Listing(title, it.get("source", ""), it.get("link", ""), amount, c,
                               bool(USED.search(f"{title} {it.get('source', '')} {it.get('delivery', '')}")), market.upper()))
        human_url = f"https://www.google.com/search?tbm=shop&gl={market.lower()}&q={re.sub(r'\s+', '+', query)}"
        return out, r["retrieved_at"], human_url


async def to_currency(fx: FX, listings: list[Listing], currency: str) -> tuple[list[Listing], list[str]]:
    """Each listing converted on its own into `currency` at the ECB rate of the day.

    Listings for one product can come in several currencies (a US site in an Indian search). Taking
    a median or range across them first and converting after mixes units: in testing a range of
    rupee prices was multiplied by the pound rate. Listings with no available rate are dropped.
    """
    out, notes, rates = [], [], {}
    for x in listings:
        if x.currency == currency:
            out.append(x)
            continue
        if x.currency not in rates:
            rates[x.currency] = await fx.rate(x.currency, currency)
        r = rates[x.currency]
        if r is None:
            continue
        out.append(Listing(x.title, x.seller, x.url, round(x.price * r[0], 2), currency, x.used, x.market, x.price, x.currency))
    for c, r in rates.items():
        if r is not None:
            notes.append(f"{c} converted at ECB rate {r[0]} ({r[1]})")
        else:
            notes.append(f"{c} listings dropped: no exchange rate")
    return out, notes


def _pick(listings: list[Listing], quantile: float = 0.5) -> tuple[float, Listing]:
    """The offer at `quantile` of the matching offers (an actual listing, so the cited source shows that
    exact figure). Replacement cost uses the lower quartile: what it costs to buy a copy today. On a real
    sweep a book had offers of 131 and 11,397 INR (a marketplace import); their median, 5,764, sent a
    paperback to appraisal."""
    ordered = sorted(listings, key=lambda x: x.price)
    rep = ordered[int(quantile * (len(ordered) - 1))]
    return round(rep.price, 2), rep


async def price_book(shop: Shopping, fx: FX, track_id: str, title: str, author: str | None, isbn: str | None,
                     country: str, currency: str) -> BookPrice:
    bp = BookPrice(track_id)
    surname = first_surname(author) if author else ""
    queries = [f"{title} {surname} book".strip()] + ([isbn] if isbn else [])
    markets = [country] + ([FALLBACK_MARKET[country]] if country in FALLBACK_MARKET else [])
    failure: str | None = None
    for want_used in (False, True):
        target = bp.used if want_used else bp.replacement
        # Used offers come from the same local search as the new price. The fallback market is searched
        # only when the local market had nothing at all: each search spends the per-minute budget.
        local_found = bp.replacement.amount is not None and not bp.replacement.converted
        for market in (markets[:1] if want_used and local_found else markets):
            found: list[Listing] = []
            retrieved, url = None, ""
            for q in queries if not want_used else [f"{title} {surname} used book".strip()]:
                ls, ret, u = await shop.search(q, market)
                failure = failure or getattr(shop, "search_failure", lambda *_: None)(q, market)
                retrieved, url = retrieved or ret, url or u
                matched = [x for x in ls if x.used == want_used and listing_matches(x.title, title, author)]
                bp.listings += [{**asdict(x), "query": q, "matched": x in matched} for x in ls]
                found += matched
                if found:
                    break
            if not found:
                continue
            if several_books(found, title):
                target.note = "the title matches several different books (a series name); volume not known"
                break
            # A listing marked as an old edition is a different product: in testing "(Old Edition)" at
            # 270 INR undercut the current 5th edition at 349. Keep such listings only if nothing else matched.
            current = [x for x in found if not OLD_EDITION.search(x.title)]
            found = current or found
            found, fx_notes = await to_currency(fx, found, currency)
            if not found:
                target.note = "; ".join(fx_notes) or "no usable listing"
                break
            amount, rep = _pick(found, 0.5 if want_used else 0.25)
            target.n_listings, target.retrieved_at = len(found), retrieved
            target.source = (f"{rep.seller} via {getattr(shop, 'label', 'price search')} "
                             f"({len(found)} matching offer{'s' if len(found) > 1 else ''}, "
                             f"{'median' if want_used else 'lower quartile'})")
            target.url = rep.url or url
            target.amount, target.currency = amount, currency
            if any(x.original_currency for x in found):
                target.converted = True
                if rep.original_currency:
                    target.original_amount, target.original_currency = rep.original_price, rep.original_currency
                target.note = f"from {market} listings; " + "; ".join(fx_notes)
            if want_used:
                target.condition_assumed = "used, good (as listed by the seller)"
                # A used copy is never worth more than a new one: used imports (₹2,494) undercut by a
                # ₹275 local paperback were valuing the second-hand book above its replacement.
                if bp.replacement.amount is not None and target.amount > bp.replacement.amount:
                    target.note = (f"used offers ({target.amount:.0f} {currency}) cost more than a new copy; "
                                   f"capped at the replacement cost")
                    target.amount = bp.replacement.amount
            break
        if target.amount is None and not target.note:
            target.note = (f"listings found but not read: {failure}" if failure
                           else "no matching listing found" if shop.enabled else "price search not configured")
    return bp


def appraisal_reason(currency: str, replacement: float | None, year: str | None, looks_antiquarian: bool,
                     statements: list[str]) -> str | None:
    """Why a book must go to a human appraiser instead of being auto-priced, or None."""
    for s in statements:
        if re.search(r"\b(signed|autograph|first edition|1st edition|rare|inscribed|limited edition)\b", s, re.I):
            return f"policyholder said: {s!r}"
    if year:
        m = re.search(r"\b(1[5-9]\d\d)\b", year)
        if m and int(m.group(1)) < 1950:
            return f"published {m.group(1)} (before 1950)"
    if looks_antiquarian:
        return "binding looks antiquarian (leather/cloth with gilt lettering or visible age)"
    limit = APPRAISAL_THRESHOLD.get(currency)
    if replacement is not None and limit is not None and replacement > limit:
        return f"market price {replacement:.0f} {currency} is above the {limit} {currency} appraisal threshold"
    return None


@dataclass
class ItemPrice:
    item_id: str
    low: float | None = None
    high: float | None = None
    status: str = "range"  # priced | range | needs_appraisal
    source: str | None = None
    url: str | None = None
    retrieved_at: str | None = None
    note: str = ""
    listings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def price_item(shop: Shopping, fx: FX, item_id: str, query: str, brand_model: str | None, country: str,
                     currency: str, needs_appraisal: str | None = None) -> ItemPrice:
    ip = ItemPrice(item_id)
    if needs_appraisal:
        ip.status, ip.note = "needs_appraisal", needs_appraisal
        return ip
    # A bare brand ("Apple") is not a model: search "Apple laptop" and report a range, not one price.
    if brand_model and not (re.search(r"\d", brand_model) or len(brand_model.split()) >= 2):
        query, brand_model = f"{brand_model} {query}", None
    q = brand_model or query
    ls, retrieved, url = await shop.search(q, country)
    failure = getattr(shop, "search_failure", lambda *_: None)(q, country)
    ls = [x for x in ls if not x.used and not NOT_THE_BOOK.search(x.title)]
    if brand_model:
        ls = [x for x in ls if fuzz.token_set_ratio(norm(brand_model), norm(x.title)) >= 80]
    ip.listings = [asdict(x) for x in ls]
    if not ls:
        ip.note = (f"listings found but not read: {failure}" if failure
                   else "no listing found" if shop.enabled else "price search not configured")
        return ip
    ls, fx_notes = await to_currency(fx, ls, currency)
    if not ls:
        ip.note = "; ".join(fx_notes) or "no usable listing"
        return ip
    prices = sorted(x.price for x in ls)
    if not brand_model and len(ls) < MIN_RANGE_LISTINGS:
        ip.note = f"only {len(ls)} comparable listing(s) found; too few for a price range"
        return ip
    if brand_model:
        amount, rep = _pick(ls)
        ip.low = ip.high = amount
        ip.status = "priced"
    else:
        # brand and model unknown: interquartile range of comparable products
        q1 = prices[int(0.25 * (len(prices) - 1))]
        q3 = prices[int(round(0.75 * (len(prices) - 1)))]
        ip.low, ip.high = round(q1, 2), round(q3, 2)
    if fx_notes:
        ip.note = "; ".join(fx_notes)
    ip.source = f"{getattr(shop, 'label', 'price search')} ({country}), {len(ls)} comparable listings" + ("" if brand_model else ", interquartile range")
    ip.url = url
    ip.retrieved_at = retrieved
    return ip
