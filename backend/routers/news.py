import asyncio
import logging
import os
import time
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx
from fastapi import APIRouter

logger = logging.getLogger("news")

router = APIRouter()

news_cache: dict = {"data": None, "cached_at": 0}
CACHE_TTL = 300  # 5 minutes

# ── Free RSS feeds (no API key needed) ──────────────────────────────────────
RSS_FEEDS = [
    {
        "url": "https://feeds.bbci.co.uk/sport/golf/rss.xml",
        "source": "BBC Sport",
        "default_category": "Tour",
    },
    {
        "url": "https://www.espn.com/espn/rss/golf/news",
        "source": "ESPN",
        "default_category": "News",
    },
    {
        "url": "https://www.skysports.com/rss/12040",
        "source": "Sky Sports",
        "default_category": "Tour",
    },
    {
        "url": "https://golf.com/feed/",
        "source": "Golf.com",
        "default_category": "News",
    },
    {
        "url": "https://www.independent.co.uk/sport/golf/rss",
        "source": "The Independent",
        "default_category": "News",
    },
    {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Golf.xml",
        "source": "NY Times",
        "default_category": "News",
    },
    {
        "url": "https://www.cbssports.com/rss/headlines/golf/",
        "source": "CBS Sports",
        "default_category": "Tournament",
    },
]

# ── Fallback images keyed by category ───────────────────────────────────────
CATEGORY_IMAGES = {
    "Tournament": "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=600&h=400&fit=crop",
    "Player News": "https://images.unsplash.com/photo-1587174486073-ae5e5cff23aa?w=600&h=400&fit=crop",
    "Training":    "https://images.unsplash.com/photo-1632501641765-e568d28b0015?w=600&h=400&fit=crop",
    "Technology":  "https://images.unsplash.com/photo-1593111774240-d529f12cf4bb?w=600&h=400&fit=crop",
    "Equipment":   "https://images.unsplash.com/photo-1591491653056-4e9d563a4b0e?w=600&h=400&fit=crop",
    "Courses":     "https://images.unsplash.com/photo-1500932334442-8761ee4810a7?w=600&h=400&fit=crop",
    "Rules":       "https://images.unsplash.com/photo-1600183952451-e1849d26e4e3?w=600&h=400&fit=crop",
    "Tour":        "https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=600&h=400&fit=crop",
    "News":        "https://images.unsplash.com/photo-1622397815403-003fae7a8eb4?w=600&h=400&fit=crop",
}
DEFAULT_IMAGE = "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?w=600&h=400&fit=crop"

NS = {
    "media":   "http://search.yahoo.com/mrss/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc":      "http://purl.org/dc/elements/1.1/",
    "atom":    "http://www.w3.org/2005/Atom",
}


def _extract_image(item_el: ET.Element) -> str:
    """Try various RSS image conventions."""
    # <media:thumbnail>
    m = item_el.find("media:thumbnail", NS)
    if m is not None:
        url = m.get("url", "")
        if url:
            return url
    # <media:content>
    m = item_el.find("media:content", NS)
    if m is not None:
        url = m.get("url", "")
        if url:
            return url
    # <enclosure type="image/...">
    enc = item_el.find("enclosure")
    if enc is not None and "image" in enc.get("type", ""):
        url = enc.get("url", "")
        if url:
            return url
    return ""


def _safe_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _extract_link(item_el: ET.Element) -> str:
    """Extract URL from <link>, handling quirks of various RSS generators."""
    import re
    link_el = item_el.find("link")
    if link_el is not None:
        # Standard: <link>https://...</link>
        url = (link_el.text or "").strip()
        if url and url.startswith("http"):
            return url
        # Some feeds put URL as tail text after <link/>
        tail = (link_el.tail or "").strip()
        if tail and tail.startswith("http"):
            return tail.split()[0]

    # Fallback: <guid isPermaLink="true">
    guid_el = item_el.find("guid")
    if guid_el is not None:
        permalink = guid_el.get("isPermaLink", "true")
        url = (guid_el.text or "").strip()
        if url and url.startswith("http") and permalink.lower() != "false":
            return url

    # Last resort: scan raw XML text for http links
    raw = ET.tostring(item_el, encoding="unicode", method="xml")
    m = re.search(r"<link[^>]*>([^<]+)</link>", raw)
    if m:
        url = m.group(1).strip()
        if url.startswith("http"):
            return url

    return ""


def _parse_rss(xml_bytes: bytes, source: str, default_category: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("[RSS] %s XML parse error: %s", source, exc)
        return []

    items = root.findall(".//item")
    articles = []
    for item in items:
        title = _safe_text(item.find("title"))
        link  = _extract_link(item)
        desc  = _safe_text(item.find("description"))
        pub   = _safe_text(item.find("pubDate"))

        # Remove HTML tags from description (very light)
        import re
        desc = re.sub(r"<[^>]+>", "", desc)[:200].strip()

        category_el = item.find("category")
        category = _safe_text(category_el) if category_el is not None else default_category
        if not category:
            category = default_category

        image = _extract_image(item)
        if not image:
            image = CATEGORY_IMAGES.get(category, DEFAULT_IMAGE)

        if not title or not link:
            continue

        uid = hashlib.md5(link.encode()).hexdigest()[:8]
        articles.append({
            "id":           uid,
            "title":        title,
            "summary":      desc,
            "image":        image,
            "source":       source,
            "category":     category,
            "published_at": pub,
            "url":          link,
        })
    return articles


def _published_ts(value: str) -> float:
    if not value:
        return 0.0
    # ISO timestamp from APIs
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    # RFC822 timestamp from RSS pubDate
    try:
        return parsedate_to_datetime(value).timestamp()
    except Exception:
        return 0.0


async def _fetch_rss(feed: dict, client: httpx.AsyncClient) -> list[dict]:
    try:
        resp = await client.get(feed["url"], timeout=10.0, follow_redirects=True)
        if resp.status_code == 200:
            articles = _parse_rss(resp.content, feed["source"], feed["default_category"])
            logger.info("[RSS] %s → %d articles", feed["source"], len(articles))
            return articles
        logger.warning("[RSS] %s → HTTP %d", feed["source"], resp.status_code)
    except Exception as exc:
        logger.warning("[RSS] %s → %s", feed["source"], exc)
    return []


async def _fetch_newsapi(api_key: str, limit: int) -> list[dict]:
    """Optional: https://newsapi.org free tier (100 req/day)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        "golf",
                    "language": "en",
                    "sortBy":   "publishedAt",
                    "pageSize": str(limit),
                    "apiKey":   api_key,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                for i, a in enumerate(data.get("articles", [])):
                    category = "News"
                    title = (a.get("title") or "").strip()
                    if not title or title == "[Removed]":
                        continue
                    articles.append({
                        "id":           str(i + 1),
                        "title":        title,
                        "summary":      (a.get("description") or "")[:200],
                        "image":        a.get("urlToImage") or DEFAULT_IMAGE,
                        "source":       (a.get("source") or {}).get("name", "Golf News"),
                        "category":     category,
                        "published_at": a.get("publishedAt", ""),
                        "url":          a.get("url", "#"),
                    })
                logger.info("[NewsAPI] → %d articles", len(articles))
                return articles
            logger.warning("[NewsAPI] → HTTP %d", resp.status_code)
    except Exception as exc:
        logger.warning("[NewsAPI] → %s", exc)
    return []


async def _fetch_guardian_with_client(client: httpx.AsyncClient, limit: int) -> list[dict]:
    """Wrapper to call _fetch_guardian using the shared client."""
    return await _fetch_guardian(limit, client)


async def _fetch_guardian(limit: int, client: httpx.AsyncClient | None = None) -> list[dict]:
    """
    Free source: The Guardian Open Platform (demo key: 'test').
    Returns reliable thumbnails via fields.thumbnail.
    """
    own_client = client is None
    _client = client or httpx.AsyncClient(timeout=10.0)
    try:
        resp = await _client.get(
            "https://content.guardianapis.com/search",
            params={
                "q": "golf",
                "section": "sport",
                "show-fields": "thumbnail,trailText,byline",
                "order-by": "newest",
                "page-size": str(min(max(limit, 5), 20)),
                "api-key": os.getenv("GUARDIAN_API_KEY", "test"),
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.warning("[Guardian] → HTTP %d", resp.status_code)
            return []
        payload = resp.json().get("response", {})
        results = payload.get("results", [])
        logger.info("[Guardian] → %d raw results", len(results))
        articles: list[dict] = []
        for item in results:
            fields = item.get("fields") or {}
            url = item.get("webUrl", "")
            title = (item.get("webTitle") or "").strip()
            if not url or not title:
                continue
            image = (fields.get("thumbnail") or "").strip() or CATEGORY_IMAGES["News"]
            summary = (fields.get("trailText") or "").strip()
            if summary:
                import re
                summary = re.sub(r"<[^>]+>", "", summary).strip()
            source = fields.get("byline", "").strip() or "The Guardian"
            pub = item.get("webPublicationDate", "")
            if pub:
                try:
                    pub = datetime.fromisoformat(pub.replace("Z", "+00:00")).isoformat()
                except Exception:
                    pass
            uid = hashlib.md5(url.encode()).hexdigest()[:8]
            articles.append({
                "id": uid,
                "title": title,
                "summary": summary[:200],
                "image": image,
                "source": source,
                "category": "Tour" if "tour" in title.lower() else "News",
                "published_at": pub,
                "url": url,
            })
        return articles
    except Exception as exc:
        logger.warning("[Guardian] → %s", exc)
        return []
    finally:
        if own_client:
            await _client.aclose()


MOCK_NEWS = [
    {"id": "1",  "title": "Scottie Scheffler Dominates Masters with Record Performance",          "summary": "World No.1 Scottie Scheffler shoots 63 in the final round to claim his third green jacket.", "image": CATEGORY_IMAGES["Tournament"],  "source": "Golf Digest",      "category": "Tournament",  "published_at": "2026-03-18T10:00:00Z", "url": "https://www.golfdigest.com/"},
    {"id": "2",  "title": "AI Technology Revolutionizes Golf Swing Analysis",                    "summary": "Machine learning detects swing flaws invisible to the human eye.",                            "image": CATEGORY_IMAGES["Technology"],  "source": "Golf Tech Weekly", "category": "Technology",  "published_at": "2026-03-17T14:30:00Z", "url": "https://www.golf.com/"},
    {"id": "3",  "title": "Rory McIlroy's Swing Changes Pay Off with PGA Tour Win",              "summary": "After months of swing retooling, McIlroy captures his first victory of the season.",         "image": CATEGORY_IMAGES["Player News"], "source": "PGA Tour",         "category": "Player News", "published_at": "2026-03-16T08:15:00Z", "url": "https://www.pgatour.com/news"},
    {"id": "4",  "title": "Top 10 Golf Training Drills to Improve Your Short Game",              "summary": "Professional coaches share favorite drills for chipping and putting.",                       "image": CATEGORY_IMAGES["Training"],    "source": "Golf Magazine",    "category": "Training",    "published_at": "2026-03-15T11:00:00Z", "url": "https://www.golf.com/instruction/"},
    {"id": "5",  "title": "USGA Announces New Equipment Rules for 2027 Season",                  "summary": "Ball distance rollback regulations take effect January 2027.",                               "image": CATEGORY_IMAGES["Rules"],       "source": "USGA",             "category": "Rules",       "published_at": "2026-03-14T16:45:00Z", "url": "https://www.usga.org/"},
    {"id": "6",  "title": "Jin Young Ko Sets New LPGA Scoring Record",                          "summary": "The Korean star shoots an unprecedented 59 in tournament play.",                             "image": CATEGORY_IMAGES["Tournament"],  "source": "LPGA",             "category": "Tournament",  "published_at": "2026-03-13T09:30:00Z", "url": "https://www.lpga.com/news"},
    {"id": "7",  "title": "Best Golf Courses to Play in 2026: Editor's Picks",                   "summary": "From Pebble Beach to St Andrews, must-play courses this year.",                             "image": CATEGORY_IMAGES["Courses"],     "source": "Travel Golf",      "category": "Courses",     "published_at": "2026-03-12T13:00:00Z", "url": "https://www.golf.com/travel/"},
    {"id": "8",  "title": "How Wearable Tech is Changing Amateur Golf Performance",              "summary": "Smart sensors and GPS watches help track and improve your game.",                            "image": CATEGORY_IMAGES["Technology"],  "source": "Tech Golf",        "category": "Technology",  "published_at": "2026-03-11T15:20:00Z", "url": "https://www.golfdigest.com/gear"},
    {"id": "9",  "title": "Youth Golf Programs See Record Enrollment Numbers",                    "summary": "Golf participation among ages 6–17 grows by 25%.",                                          "image": CATEGORY_IMAGES["News"],        "source": "Golf Foundation",  "category": "Community",   "published_at": "2026-03-10T10:00:00Z", "url": "https://www.thefirsttee.org/"},
    {"id": "10", "title": "Grip Pressure Secrets: What Tour Pros Don't Tell You",                "summary": "Biomechanics experts reveal optimal grip pressure patterns for maximum distance.",           "image": CATEGORY_IMAGES["Training"],    "source": "Golf Science",     "category": "Technique",   "published_at": "2026-03-09T12:30:00Z", "url": "https://www.golf.com/instruction/"},
]


@router.get("/news")
async def get_golf_news(limit: int = 10):
    now = time.time()

    # Serve from cache if fresh
    if news_cache["data"] and (now - news_cache["cached_at"]) < CACHE_TTL:
        return {"news": news_cache["data"][:limit], "source": "cache"}

    all_articles: list[dict] = []

    # ── Option 1: NewsAPI.org (free, 100 req/day) ─────────────────────────
    newsapi_key = os.getenv("GOLF_NEWS_API_KEY", "")
    if newsapi_key:
        newsapi_articles = await _fetch_newsapi(newsapi_key, limit)
        all_articles.extend(newsapi_articles)

    # ── Option 2: The Guardian free API + RSS feeds — fetch concurrently ──
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (compatible; StellarGolfBot/1.0)"},
        follow_redirects=True,
    ) as client:
        tasks = [_fetch_rss(feed, client) for feed in RSS_FEEDS]
        tasks.append(_fetch_guardian_with_client(client, limit))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

    # Deduplicate by id, sort newest first
    seen: set[str] = set()
    unique: list[dict] = []
    for a in all_articles:
        if a["id"] not in seen:
            seen.add(a["id"])
            unique.append(a)
    unique.sort(key=lambda x: _published_ts(x.get("published_at", "")), reverse=True)

    if unique:
        logger.info("Aggregated %d unique articles from all sources", len(unique))
        news_cache["data"] = unique
        news_cache["cached_at"] = now
        return {"news": unique[:limit], "source": "aggregated"}

    # ── Fallback: mock data ───────────────────────────────────────────────────
    logger.warning("All news sources failed, serving mock data")
    news_cache["data"] = MOCK_NEWS
    news_cache["cached_at"] = now
    return {"news": MOCK_NEWS[:limit], "source": "mock"}
