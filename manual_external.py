# -*- coding: utf-8 -*-
"""Manual external entry by URL (X / Reddit).

    Admin form (source + url + text)  ->  build_item()  ->  ai_pipeline.py

This module does NO network I/O — the admin supplies the comment text
directly, so we only validate + normalize into the standard item shape the
storage layer expects. Keeping it network-free means the manual path always
works even when a platform's API keys are missing or rate-limited (which is
the whole point of having a manual path alongside the API path).

Item shape produced (same keys the API paths use, plus the unified tags):
    {external_id, text, author, created_at, likes, source, url,
     source_type, source_url, kind}
"""
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

# Admin-facing source keys -> (internal `source` value, host patterns).
# `source` stays 'x' / 'reddit' so all existing filters, badges and analytics
# keep grouping manual + API rows together by platform, while `source_type`
# distinguishes the FOUR ingestion paths.
_SPECS = {
    "X": {
        "source": "x",
        "source_type": "X_Manual",
        "hosts": ("x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com"),
    },
    "Reddit": {
        "source": "reddit",
        "source_type": "Reddit_Manual",
        "hosts": ("reddit.com", "www.reddit.com", "old.reddit.com", "np.reddit.com"),
    },
}

SUPPORTED = tuple(_SPECS.keys())


class ManualSourceError(Exception):
    """Bad admin input (unknown source / bad URL / empty text). Carries a
    bilingual message + HTTP status so the route can surface it directly."""

    def __init__(self, message_en, message_ar, status=400):
        super().__init__(message_en)
        self.message_en = message_en
        self.message_ar = message_ar
        self.status = status


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _external_id(source: str, url: str) -> str:
    """Stable de-dup id derived from the URL.

    Prefer the platform's own id embedded in the link so a manual entry and
    the SAME post fetched later by the API collapse to one row:
      - X:      .../status/1234567890            -> "x:1234567890"
      - Reddit: .../comments/abc123/title/...    -> "reddit:abc123"
    Fall back to the normalized full URL when no id is found."""
    if source == "x":
        m = re.search(r"/status(?:es)?/(\d+)", url)
        if m:
            return "x:" + m.group(1)
    elif source == "reddit":
        m = re.search(r"/comments/([0-9a-z]+)", url, re.I)
        if m:
            return "reddit:" + m.group(1).lower()
    # Fallback: strip query/fragment + trailing slash for a stable key.
    clean = re.sub(r"[?#].*$", "", (url or "").strip()).rstrip("/")
    return "manual:" + (clean or url or "")


def build_item(source: str, url: str, text: str) -> dict:
    """Validate the admin form and return a pipeline-ready item dict.

    Raises ManualSourceError on any invalid input (unknown source, a URL that
    doesn't belong to the chosen platform, or empty text)."""
    source = (source or "").strip()
    spec = _SPECS.get(source) or _SPECS.get(source.capitalize())
    if not spec:
        raise ManualSourceError(
            f"Unsupported source '{source}' (use one of: {', '.join(SUPPORTED)})",
            f"مصدر غير مدعوم '{source}' (اختر أحد: {'، '.join(SUPPORTED)})", 400)

    url = (url or "").strip()
    text = (text or "").strip()
    if not text:
        raise ManualSourceError("Comment text is required",
                                "نص التعليق مطلوب", 400)
    if not url:
        raise ManualSourceError("The external link is required",
                                "الرابط الخارجي مطلوب", 400)

    host = _host(url)
    if not host:
        raise ManualSourceError("That doesn't look like a valid URL",
                                "الرابط غير صالح", 400)
    if not any(host == h or host.endswith("." + h) for h in spec["hosts"]):
        raise ManualSourceError(
            f"This link is not a {source} link (host: {host})",
            f"هذا الرابط ليس رابط {source} (النطاق: {host})", 400)

    return {
        "external_id": _external_id(spec["source"], url),
        "text": text,
        "author": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "likes": 0,
        "source": spec["source"],
        "url": url,
        "source_type": spec["source_type"],
        "source_url": url,
        "kind": "post",
    }
