"""TinyFish price source: search retailer sites, fetch the product pages, read the offers on them.

1. Search (TinyFish Search API) restricted to retailers of the target market (amazon.in, flipkart.com,
   bookswagon.com, ... for India) for the book or item.
2. Fetch (TinyFish Fetch API) the best-matching product pages as markdown.
3. A small model lists the offers it sees on each page: price text copied verbatim, condition, format,
   seller. It reads prices off the fetched page; it is never asked to know one.
4. Code keeps an offer only if its price text occurs verbatim in the fetched page. An offer that is not
   on the page is dropped and counted, so no price can come from model memory.

The result has the same shape as the Serper source (a list of Listing), so price.py's matching,
median and conversion logic applies unchanged. One search serves both the new and the used price:
used offers come from the same pages (marketplace "used" offers) and from used-book shops included
in the search, which halves the number of searches against the per-minute limit.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from app.pipeline.gemini import QuotaExhausted, Vision
from app.pipeline.http import CachedHTTP
from app.pipeline.identify import norm

SEARCH_URL = "https://api.search.tinyfish.ai/"
FETCH_URL = "https://api.fetch.tinyfish.ai/"
PAGES_PER_SEARCH = 3
PAGE_CHARS = 24000  # offers sit near the top of product pages; the rest is reviews and recommendations
WINDOW = 350  # characters kept either side of each price on the page
MAX_EXTRACT_CHARS = 6000

BOOK_DOMAINS = {
    "IN": "amazon.in,flipkart.com,bookswagon.com,sapnaonline.com,crossword.in,bookchor.com,kitabay.com",
    "US": "amazon.com,barnesandnoble.com,bookshop.org,booksamillion.com,thriftbooks.com,abebooks.com",
    "GB": "amazon.co.uk,waterstones.com,blackwells.co.uk,wordery.com,worldofbooks.com,abebooks.co.uk",
}
ITEM_DOMAINS = {
    # ikea.com is left out for India: its US pages outrank ikea.com/in even with location=IN
    "IN": "amazon.in,flipkart.com,pepperfry.com,urbanladder.com,woodenstreet.com,nilkamalfurniture.com,croma.com,reliancedigital.in",
    "US": "amazon.com,walmart.com,target.com,ikea.com,wayfair.com,bestbuy.com",
    "GB": "amazon.co.uk,argos.co.uk,ikea.com,johnlewis.com,currys.co.uk",
}
FOREIGN_LOCALE = {"IN": re.compile(r"/(us|se|gb|uk|de|fr|au|ca|jp)(/|-en/|/en/)", re.I)}
ITEM_PROMPT = """\
Below is the text of a retailer web page, fetched just now. Target: comparable products to "{target}".

List the products on this page that are the same kind of product as the target (for example, any
bookcase for a bookshelf), up to 15, each with its price copied exactly as printed, including the
currency symbol. Skip accessories, spare parts and unrelated products, struck-through list prices and
EMI amounts. If the page shows no such products, set is_target to false. Never write a price that is
not in the text.

PAGE TITLE: {title}
PAGE TEXT:
{text}
"""
PRODUCT_URL = re.compile(r"/dp/|/gp/product/|flipkart\.com/.+/p/|/book/|/books/|/product|/p/|/ip/|/itm|isbn|\d{10,13}", re.I)

OFFER_SCHEMA = {
    "type": "object",
    "properties": {
        "is_target": {"type": "boolean", "description": "the page is for the target product (same title/author, or same kind of product)"},
        "page_product": {"type": "string", "description": "the product name as shown on the page"},
        "offers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "price_text": {"type": "string", "description": "the price exactly as printed on the page, with its currency symbol"},
                    "condition": {"type": "string", "enum": ["new", "used", "unknown"]},
                    "format": {"type": "string", "enum": ["paperback", "hardcover", "ebook", "audiobook", "other", "unknown"]},
                    "seller": {"type": "string"},
                },
                "required": ["price_text", "condition", "format", "seller"],
            },
        },
    },
    "required": ["is_target", "page_product", "offers"],
}

OFFER_PROMPT = """\
Below is the text of a retailer web page, fetched just now. Target product: {target}

List the purchase offers shown on this page for the target product: the main price and any other
seller or used offers. Copy each price exactly as it appears on the page, including the currency
symbol. Mark an offer "used" only if that specific offer is for a used / pre-owned copy; summary lines
such as "New & Used from ₹X" or "other sellers from ₹X" are not offers. Do not list struck-through list prices, EMI amounts, delivery charges, or prices of other
products (recommendations, "customers also bought"). If the page is not for the target product, set
is_target to false and return no offers. Never write a price that is not in the text.

PAGE TITLE: {title}
PAGE TEXT:
{text}
"""


MARKET_NAME = {"IN": "India", "US": "USA", "GB": "UK"}


class RateLimiter:
    """At most `per_minute` acquisitions in any 60 s window."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.stamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.stamps = [t for t in self.stamps if now - t < 60]
                if len(self.stamps) < self.per_minute:
                    self.stamps.append(now)
                    return
                await asyncio.sleep(60 - (now - self.stamps[0]) + 0.05)


CURRENCY = re.compile(r"₹|Rs\.?\s?\d|\$\s?\d|£\s?\d|€\s?\d")


def price_windows(text: str) -> str:
    """The parts of a page around prices, merged, capped at MAX_EXTRACT_CHARS.

    Sends the extractor the offers rather than navigation and reviews (about a third of the tokens of
    the page top). Verification still runs against the whole fetched page.
    """
    text = text[:PAGE_CHARS]
    spans: list[list[int]] = []
    for m in CURRENCY.finditer(text):
        a, b = max(0, m.start() - WINDOW), min(len(text), m.end() + WINDOW)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    out = "\n...\n".join(text[a:b] for a, b in spans)
    return out[:MAX_EXTRACT_CHARS] if out else text[:MAX_EXTRACT_CHARS]


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def verified(price_text: str, page_text: str) -> bool:
    """The offer's price string really occurs on the fetched page (whitespace-insensitive)."""
    p = squash(price_text)
    return bool(p) and bool(re.search(r"\d", p)) and p in squash(page_text)


@dataclass
class Page:
    url: str
    title: str
    text: str
    retrieved_at: str


class TinyFishShopping:
    """Same interface as price.Shopping: search(query, market) -> (listings, retrieved_at, url)."""

    label = "retailer page fetched with TinyFish"

    def __init__(self, http: CachedHTTP, api_key: str, extractor: Vision, searches_per_minute: int = 30, kind: str = "book"):
        self.http = http
        self.key = api_key
        self.extractor = extractor
        self.kind = kind
        self.limiter = RateLimiter(searches_per_minute)
        self.fetch_sem = asyncio.Semaphore(6)
        self._memo: dict[tuple[str, str], asyncio.Task] = {}
        self.dropped_unverified = 0
        self.extract_failure: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    async def search(self, query: str, market: str):
        # new and used price searches for the same book share one search ("... used book" -> "... book")
        base = re.sub(r"\bused\b", " ", query, flags=re.I)
        base = re.sub(r"\s+", " ", base).strip()
        key = (base.lower(), market.upper())
        if key not in self._memo:
            self._memo[key] = asyncio.ensure_future(self._search(base, market.upper()))
        return await self._memo[key]

    async def _search(self, query: str, market: str):
        from app.pipeline.price import CURRENCY_OF, Listing, parse_price

        if not self.key:
            return [], None, ""
        domains = (BOOK_DOMAINS if self.kind == "book" else ITEM_DOMAINS).get(market, "")
        if self.kind == "item":
            # Generic product words ("wardrobe") rank global sites first; say where and why.
            query = f"buy {query} online {MARKET_NAME.get(market, '')}".strip()
        await self.limiter.acquire()
        try:
            r = await self.http.request_json(
                "price", "GET", SEARCH_URL,
                params={"query": query, "location": market, "include_domains": domains,
                        "purpose": f"find retailer product pages with current prices for: {query}"},
                headers={"X-API-Key": self.key}, max_age_days=7)
        except Exception:
            return [], None, ""
        results = r["data"].get("results", []) or []
        target = norm(query.replace(" book", ""))
        foreign = FOREIGN_LOCALE.get(market)
        results = [x for x in results if not (foreign and foreign.search(x.get("url", "")))]
        if self.kind == "book":
            # one book: product pages whose title matches it
            ranked = [x for x in results if PRODUCT_URL.search(x.get("url", ""))]
            min_sim = 60
        else:
            # a generic item: product or category pages; a category page lists many comparable prices
            ranked, min_sim = list(results), 35
        ranked.sort(key=lambda x: fuzz.token_set_ratio(target, norm(x.get("title", ""))), reverse=True)
        urls = [x["url"] for x in ranked if fuzz.token_set_ratio(target, norm(x.get("title", ""))) >= min_sim][:PAGES_PER_SEARCH]
        search_url = f"{SEARCH_URL}?query={re.sub(r'\\s+', '+', query)}&location={market}"
        if not urls:
            return [], r["retrieved_at"], search_url
        pages = await self._fetch(urls)
        cur = CURRENCY_OF.get(market, "USD")
        listings: list[Listing] = []
        for page in pages:
            offers = await self._offers(query, page)
            new_prices = {squash(o.get("price_text", "")) for o in offers if o.get("condition") != "used"}
            for o in offers:
                if o.get("condition") == "used" and squash(o.get("price_text", "")) in new_prices:
                    continue  # the same figure as a new offer on this page: a summary line, not a used copy
                if o.get("format") in ("ebook", "audiobook"):
                    continue
                if not verified(o.get("price_text", ""), page.text):
                    self.dropped_unverified += 1
                    continue
                amount, c = parse_price(o["price_text"], cur)
                if amount is None or amount <= 0:
                    continue
                site = re.sub(r"^www\.", "", re.sub(r"^https?://([^/]+).*$", r"\1", page.url))
                seller = f"{site}" + (f" (seller: {o['seller']})" if o.get("seller") else "")
                used = o.get("condition") == "used" or site in ("bookchor.com", "kitabay.com", "thriftbooks.com", "abebooks.com", "abebooks.co.uk", "worldofbooks.com")
                listings.append(Listing(page.title or query, seller, page.url, amount, c, used, market))
        retrieved = min((p.retrieved_at for p in pages), default=r["retrieved_at"])
        return listings, retrieved, search_url

    async def _fetch(self, urls: list[str]) -> list[Page]:
        async with self.fetch_sem:
            try:
                r = await self.http.request_json(
                    "price", "POST", FETCH_URL,
                    json_body={"urls": urls, "format": "markdown", "per_url_timeout_ms": 25000,
                               "purpose": "read the current price offers for this product"},
                    headers={"X-API-Key": self.key}, max_age_days=7)
            except Exception:
                return []
        pages = []
        for res in r["data"].get("results", []) or []:
            text = res.get("text") or ""
            if text:
                pages.append(Page(res.get("final_url") or res.get("url", ""), res.get("title") or "", text, r["retrieved_at"]))
        return pages

    async def _offers(self, target: str, page: Page) -> list[dict[str, Any]]:
        text = price_windows(page.text)
        try:
            raw = await self.extractor.json(
                "price_extract",
                [(OFFER_PROMPT if self.kind == "book" else ITEM_PROMPT).format(target=target, title=page.title, text=text)],
                OFFER_SCHEMA, thinking="minimal", media_resolution="low")
        except QuotaExhausted as e:
            # Not "no listing": the page was found but could not be read. Say so on every price line.
            self.extract_failure = str(e)
            return []
        except Exception as e:
            self.extract_failure = self.extract_failure or f"price reading failed: {type(e).__name__}"
            return []
        if not isinstance(raw, dict) or not raw.get("is_target"):
            return []
        offers = raw.get("offers") or []
        for o in offers:
            o["page_product"] = raw.get("page_product", "")
        return offers


def describe() -> str:
    return json.dumps({"search": SEARCH_URL, "fetch": FETCH_URL, "pages_per_search": PAGES_PER_SEARCH})
