# -*- coding: utf-8 -*-
"""bulk_import — detect the platform from an account/page/video URL and
fetch the available content, ready for the existing analysis pipeline.

    URL  ->  detect_source()  ->  x_api / reddit_api / youtube_api  ->  items
    items -> app._analyze_and_store_items (dedup + relevance + sentiment + save)

This module only DISPATCHES to the existing per-source fetchers (x_api.py,
reddit_api.py, youtube_api.py). It does not add a second integration and
never fabricates data — every failure is surfaced as a clear bilingual
BulkImportError so the UI can show a precise reason.
"""
import re

import x_api
import reddit_api
import youtube_api


class BulkImportError(Exception):
    def __init__(self, message_en, message_ar, status=400):
        super().__init__(message_en)
        self.message_en = message_en
        self.message_ar = message_ar
        self.status = status


def detect_source(url: str) -> str:
    """Return 'x' | 'reddit' | 'youtube' from the URL host, or '' if unknown."""
    s = (url or "").strip().lower()
    if not s:
        return ""
    if re.search(r"(^|//|\.)(x\.com|twitter\.com)/", s) or "twitter.com" in s or "x.com" in s:
        return "x"
    if "reddit.com" in s:
        return "reddit"
    if "youtube.com" in s or "youtu.be" in s:
        return "youtube"
    return ""


# maps detected source -> (module, its fetch-error type, human label)
_DISPATCH = {
    "x": (x_api, x_api.XFetchError, "X"),
    "reddit": (reddit_api, reddit_api.RedditFetchError, "Reddit"),
    "youtube": (youtube_api, youtube_api.YouTubeFetchError, "YouTube"),
}


def fetch(url: str, limit: int = 150) -> dict:
    """Detect the source and fetch up to `limit` items from the URL.

    Returns {"source": 'x'|'reddit'|'youtube', "label": str, "query": str,
    "items": [...]} in the standard pipeline item shape. Raises
    BulkImportError (with a bilingual message + HTTP status) on any failure —
    including an unrecognized link or a source-specific fetch error, which is
    re-wrapped so the caller has a single exception type to handle."""
    source = detect_source(url)
    if not source:
        raise BulkImportError(
            "Unrecognized link — use an X, Reddit or YouTube URL",
            "رابط غير معروف — استخدم رابط X أو Reddit أو YouTube", 400)
    module, err_type, label = _DISPATCH[source]
    try:
        result = module.fetch_from_url(url, limit=limit)
    except err_type as e:
        # re-wrap the source-specific error into the unified type, keeping the
        # original bilingual message + status.
        raise BulkImportError(e.message_en, e.message_ar, getattr(e, "status", 502))
    return {
        "source": source,
        "label": label,
        "query": result.get("query"),
        "items": result.get("items") or [],
    }
