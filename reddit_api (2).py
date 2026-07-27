# -*- coding: utf-8 -*-
"""Reddit API path — auto-fetch Hajj/Umrah posts + top comments.

    Reddit API  ->  reddit_api.py  ->  ai_pipeline.py

Thin wrapper over external_sources.search_reddit_posts() (OAuth2 client
credentials, subreddit/keyword search — already implemented and tested). It
adds the unified-schema tags:

    source_type = "Reddit_API"
    source_url  = <permalink>

The existing external_sources.fetch_reddit / search_reddit_posts stay intact.
"""
import os

import external_sources

RedditFetchError = external_sources.RedditFetchError

SOURCE = "reddit"
SOURCE_TYPE = "Reddit_API"


def configured() -> bool:
    """True when Reddit client credentials are present."""
    return bool(os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET"))


def fetch(max_posts: int = 20, max_comments_per_post: int = 8) -> dict:
    """Search Reddit for Hajj/Umrah posts (+ a few top comments each) and
    return them ready for the pipeline.

    Returns {"query": str, "items": [item, ...]} with every item tagged
    source_type="Reddit_API" and source_url=<permalink>. Raises
    RedditFetchError when Reddit isn't configured or the API rejects the
    request."""
    result = external_sources.search_reddit_posts(
        max_posts=max_posts, max_comments_per_post=max_comments_per_post)
    items = result.get("items") or []
    for it in items:
        it["source_type"] = SOURCE_TYPE
        it["source_url"] = it.get("url")
    return {"query": result.get("query"), "items": items}


# ------------------------------------------------------------------ #
# v15.10 — BULK import by URL:
#   subreddit:  https://www.reddit.com/r/<sub>/
#   user:       https://www.reddit.com/user/<name>/  (or /u/<name>)
# Reuses external_sources._reddit_access_token() (same OAuth2 creds) and the
# standard item shape. Never returns fake data; raises RedditFetchError.
# ------------------------------------------------------------------ #
import re as _re
import requests as _requests

_TIMEOUT = getattr(external_sources, "TIMEOUT", 15)


def parse_target(url: str):
    """Return ('subreddit'|'user', name) parsed from a Reddit URL, or
    (None, None) if it isn't a recognizable subreddit/user link."""
    s = (url or "").strip()
    m = _re.search(r"reddit\.com/r/([A-Za-z0-9_]{2,50})", s, _re.I)
    if m:
        return "subreddit", m.group(1)
    m = _re.search(r"reddit\.com/(?:user|u)/([A-Za-z0-9_\-]{2,50})", s, _re.I)
    if m:
        return "user", m.group(1)
    return None, None


def _listing(access, agent, path, limit):
    """Page a Reddit listing endpoint up to `limit` items, best-effort."""
    headers = {"Authorization": f"Bearer {access}", "User-Agent": agent}
    out, after, guard = [], None, 0
    while len(out) < limit and guard < 40:
        guard += 1
        params = {"limit": min(100, max(1, limit - len(out))), "raw_json": 1}
        if after:
            params["after"] = after
        try:
            r = _requests.get("https://oauth.reddit.com" + path,
                              params=params, headers=headers, timeout=_TIMEOUT)
        except Exception as e:
            print(f"[reddit_api] listing network error: {e}")
            raise RedditFetchError("Network error contacting Reddit",
                                   "خطأ في الاتصال بخدمة Reddit", 502)
        if r.status_code == 429:
            if out:
                break
            raise RedditFetchError("Reddit rate limit reached — try again shortly",
                                   "تم تجاوز حد طلبات Reddit — حاول بعد قليل", 429)
        if r.status_code == 404:
            raise RedditFetchError("That subreddit/user was not found",
                                   "المجتمع أو المستخدم غير موجود", 404)
        if r.status_code != 200:
            if out:
                break
            raise RedditFetchError(f"Reddit API error (HTTP {r.status_code})",
                                   f"خطأ من Reddit (HTTP {r.status_code})", 502)
        data = (r.json() or {}).get("data", {}) or {}
        children = data.get("children", []) or []
        if not children:
            break
        out.extend(children)
        after = data.get("after")
        if not after:
            break
    return out[:limit]


def fetch_from_url(url: str, limit: int = 150) -> dict:
    """Fetch up to `limit` posts/comments from a subreddit or user URL.

    Returns {"query", "items": [...]} in the standard pipeline shape. Raises
    RedditFetchError with a clear bilingual reason on failure."""
    kind_target, name = parse_target(url)
    if not kind_target:
        raise RedditFetchError(
            "Could not read a subreddit (/r/...) or user (/user/...) from that link",
            "تعذّر التعرّف على مجتمع (/r/) أو مستخدم (/user/) من الرابط", 400)
    access, agent = external_sources._reddit_access_token()
    limit = max(5, min(int(limit or 150), 1000))

    if kind_target == "subreddit":
        children = _listing(access, agent, f"/r/{name}/new", limit)
        query = f"r/{name}"
    else:
        # a user's submitted posts + their comments, merged newest-first-ish
        subs = _listing(access, agent, f"/user/{name}/submitted", limit)
        coms = _listing(access, agent, f"/user/{name}/comments", max(0, limit - len(subs)))
        children = subs + coms
        query = f"u/{name}"

    now_iso = external_sources.datetime.now(external_sources.timezone.utc).isoformat()
    items = []
    for ch in children:
        d = ch.get("data", {}) or {}
        kind = ch.get("kind")  # t1=comment, t3=post
        if kind == "t3":
            text = (d.get("selftext") or d.get("title") or "").strip()
            ext = "reddit:" + str(d.get("id") or "")
            item_kind = "post"
            title = d.get("title")
        else:
            text = (d.get("body") or "").strip()
            ext = "reddit:c_" + str(d.get("id") or "")
            item_kind = "comment"
            title = None
        if not text:
            continue
        created = d.get("created_utc")
        permalink = ("https://reddit.com" + d["permalink"]) if d.get("permalink") else url
        items.append({
            "external_id": ext,
            "text": text[:2000],
            "author": "u/" + (d.get("author") or name),
            "created_at": (external_sources.datetime.fromtimestamp(
                created, tz=external_sources.timezone.utc).isoformat() if created else now_iso),
            "likes": int(d.get("ups") or d.get("score") or 0),
            "source": "reddit",
            "url": permalink,
            "kind": item_kind,
            "title": title,
            "community": d.get("subreddit"),
            "source_type": "Reddit_URL_Import",
            "source_url": permalink,
        })
        if len(items) >= limit:
            break
    return {"query": query, "items": items}
