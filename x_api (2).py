# -*- coding: utf-8 -*-
"""X (Twitter) API path — auto-fetch Hajj/Umrah posts.

    X API  ->  x_api.py  ->  ai_pipeline.py   (module lives at project root)  (via app._analyze_and_store_items)

This is a THIN wrapper over the existing, battle-tested search logic in
external_sources.search_x_posts(). It does NOT re-implement the HTTP call —
it reuses it and adds the two things the new unified schema needs:

    source_type = "X_API"     (so every row knows which of the 4 paths made it)
    source_url  = <post url>  (the permalink already returned by the search)

Nothing here is destructive: the original external_sources.fetch_x /
search_x_posts remain available and unchanged.
"""
import os

import external_sources

# Re-export the existing error type so callers keep one import site.
XFetchError = external_sources.XFetchError

SOURCE = "x"            # existing `source` column value (unchanged)
SOURCE_TYPE = "X_API"   # new `source_type` column value for this path


def configured() -> bool:
    """True when an X bearer token is present in the environment."""
    return bool(os.environ.get("X_BEARER_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN"))


def fetch(max_results: int = 50) -> dict:
    """Search X for Hajj/Umrah posts and return them ready for the pipeline.

    Returns {"query": str, "items": [item, ...]} — the SAME shape as
    external_sources.search_x_posts, but every item is tagged with
    source_type/source_url. Raises XFetchError when X isn't configured or
    the API rejects the request (handled by the route)."""
    result = external_sources.search_x_posts(max_results=max_results)
    items = result.get("items") or []
    for it in items:
        it["source_type"] = SOURCE_TYPE
        it["source_url"] = it.get("url")
    return {"query": result.get("query"), "items": items}


# ------------------------------------------------------------------ #
# v15.10 — BULK import by account URL:  https://x.com/<username>
# Reuses the same X_BEARER_TOKEN + item shape as search_x_posts above; the
# only difference is WHAT we fetch (a user's recent timeline instead of a
# keyword search). Never returns fake data; raises XFetchError on any failure.
# ------------------------------------------------------------------ #
import re as _re
import requests as _requests

_API = "https://api.twitter.com/2"
_TIMEOUT = getattr(external_sources, "TIMEOUT", 15)
# paths on x.com that are NOT usernames
_NON_USER = {"i", "home", "search", "hashtag", "explore", "notifications",
             "messages", "settings", "compose", "intent", "share"}


def parse_username(url: str) -> str:
    """Extract a bare @handle from an x.com / twitter.com URL (or a raw
    @handle / handle). Returns '' when no username can be read."""
    s = (url or "").strip()
    if not s:
        return ""
    m = _re.search(r"(?:x\.com|twitter\.com)/@?([A-Za-z0-9_]{1,15})", s, _re.I)
    if m:
        handle = m.group(1)
    else:
        handle = s.lstrip("@")
    if not _re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle or ""):
        return ""
    if handle.lower() in _NON_USER:
        return ""
    return handle


def _bearer():
    tok = os.environ.get("X_BEARER_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN")
    if not tok:
        raise XFetchError(
            "X isn't configured on the server (missing X_BEARER_TOKEN)",
            "خدمة X غير مُفعّلة على الخادم (مفتاح X_BEARER_TOKEN مفقود)", 501)
    return tok


def fetch_from_url(url: str, limit: int = 150) -> dict:
    """Fetch up to `limit` recent posts from a public X account URL.

    Returns {"query", "items": [...]} in the standard pipeline item shape.
    Only the platform-allowed amount is fetched (paginated within the API's
    limits). Raises XFetchError with a clear bilingual reason on any failure
    — no fake data is ever produced."""
    bearer = _bearer()
    username = parse_username(url)
    if not username:
        raise XFetchError("Could not read an X username from that link",
                          "تعذّر التعرّف على اسم حساب X من الرابط", 400)
    headers = {"Authorization": f"Bearer {bearer}"}
    try:
        u = _requests.get(f"{_API}/users/by/username/{username}",
                          headers=headers, timeout=_TIMEOUT)
    except Exception as e:
        print(f"[x_api] user lookup network error: {e}")
        raise XFetchError("Network error contacting X", "خطأ في الاتصال بخدمة X", 502)
    if u.status_code in (401, 403):
        raise XFetchError(
            "X rejected the token (invalid, or the plan lacks this access)",
            "رفضت X المفتاح (غير صالح أو الباقة لا تدعم هذا الوصول)", 403)
    if u.status_code == 404 or not (u.json() or {}).get("data"):
        raise XFetchError(f"X account @{username} not found",
                          f"حساب X ‏@{username} غير موجود", 404)
    uid = u.json()["data"]["id"]

    limit = max(5, min(int(limit or 150), 1000))
    items, token, guard = [], None, 0
    while len(items) < limit and guard < 40:
        guard += 1
        params = {"max_results": min(100, max(5, limit - len(items))),
                  "tweet.fields": "created_at,public_metrics,lang",
                  "exclude": "retweets,replies"}
        if token:
            params["pagination_token"] = token
        try:
            r = _requests.get(f"{_API}/users/{uid}/tweets", params=params,
                              headers=headers, timeout=_TIMEOUT)
        except Exception as e:
            print(f"[x_api] timeline network error: {e}")
            raise XFetchError("Network error contacting X", "خطأ في الاتصال بخدمة X", 502)
        if r.status_code == 429:
            if items:
                break  # keep what we already have
            raise XFetchError("X rate limit reached — try again shortly",
                              "تم تجاوز حد طلبات X — حاول بعد قليل", 429)
        if r.status_code != 200:
            if items:
                break
            raise XFetchError(f"X API error (HTTP {r.status_code})",
                              f"خطأ من X (HTTP {r.status_code})", 502)
        body = r.json() or {}
        for tw in body.get("data", []) or []:
            text = (tw.get("text") or "").strip()
            if not text:
                continue
            tid = str(tw.get("id"))
            metrics = tw.get("public_metrics", {}) or {}
            items.append({
                "external_id": "x:" + tid,
                "text": text,
                "author": "@" + username,
                "created_at": tw.get("created_at") or external_sources.datetime.now(
                    external_sources.timezone.utc).isoformat(),
                "likes": int(metrics.get("like_count", 0) or 0),
                "source": "x",
                "url": f"https://twitter.com/{username}/status/{tid}",
                "kind": "post",
                "source_type": "X_URL_Import",
                "source_url": f"https://twitter.com/{username}/status/{tid}",
            })
            if len(items) >= limit:
                break
        token = (body.get("meta") or {}).get("next_token")
        if not token:
            break
    return {"query": f"@{username}", "items": items}
