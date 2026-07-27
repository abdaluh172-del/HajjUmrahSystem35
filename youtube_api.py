# -*- coding: utf-8 -*-
"""YouTube URL import — thin wrapper over the existing external_sources
YouTube functions (same YOUTUBE_API_KEY, same item/comment shapes).

    YouTube URL  ->  youtube_api.py  ->  ai_pipeline.py

Supports two kinds of link:
  * a VIDEO url  -> pull that video's comments (external_sources.fetch_video_comments)
  * a CHANNEL url (@handle, /channel/UC..., /c/name, /user/name)
        -> resolve the channel, take its latest videos, pull their comments

Never returns fake data; raises external_sources.YouTubeFetchError with a
clear bilingual reason on any failure. The original external_sources
functions stay unchanged.
"""
import os
import re

import requests

import external_sources

YouTubeFetchError = external_sources.YouTubeFetchError
SOURCE = "youtube"
SOURCE_TYPE = "YouTube_URL_Import"

_API = "https://www.googleapis.com/youtube/v3"
_TIMEOUT = getattr(external_sources, "TIMEOUT", 15)


def configured() -> bool:
    return bool(os.environ.get("YOUTUBE_API_KEY"))


def _key():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeFetchError(
            "YouTube isn't configured on the server (missing YOUTUBE_API_KEY)",
            "خدمة YouTube غير مُفعّلة على الخادم (مفتاح YOUTUBE_API_KEY مفقود)", 501)
    return key


def _parse_channel(url: str):
    """Return ('handle'|'id'|'name'|'user', value) for a channel URL, or
    (None, None). Video links are handled separately via extract_video_id."""
    s = (url or "").strip()
    m = re.search(r"youtube\.com/@([A-Za-z0-9_.\-]+)", s, re.I)
    if m:
        return "handle", m.group(1)
    m = re.search(r"youtube\.com/channel/(UC[A-Za-z0-9_\-]{20,})", s, re.I)
    if m:
        return "id", m.group(1)
    m = re.search(r"youtube\.com/c/([A-Za-z0-9_.\-]+)", s, re.I)
    if m:
        return "name", m.group(1)
    m = re.search(r"youtube\.com/user/([A-Za-z0-9_.\-]+)", s, re.I)
    if m:
        return "user", m.group(1)
    return None, None


def _resolve_channel_id(key, kind, value):
    """Best-effort resolution of a channel URL piece to a UC... channel id."""
    if kind == "id":
        return value
    try:
        if kind == "handle":
            r = requests.get(f"{_API}/channels", params={
                "part": "id", "forHandle": value, "key": key}, timeout=_TIMEOUT)
            items = (r.json() or {}).get("items") if r.status_code == 200 else None
            if items:
                return items[0]["id"]
        if kind == "user":
            r = requests.get(f"{_API}/channels", params={
                "part": "id", "forUsername": value, "key": key}, timeout=_TIMEOUT)
            items = (r.json() or {}).get("items") if r.status_code == 200 else None
            if items:
                return items[0]["id"]
        # fallback: search by name/handle text
        r = requests.get(f"{_API}/search", params={
            "part": "id", "q": value, "type": "channel", "maxResults": 1,
            "key": key}, timeout=_TIMEOUT)
        if r.status_code == 200:
            items = (r.json() or {}).get("items") or []
            if items:
                return items[0]["id"]["channelId"]
    except Exception as e:
        print(f"[youtube_api] channel resolve error: {e}")
    return None


def _latest_video_ids(key, channel_id, max_videos):
    try:
        r = requests.get(f"{_API}/search", params={
            "part": "id", "channelId": channel_id, "type": "video",
            "order": "date", "maxResults": min(50, max(1, max_videos)),
            "key": key}, timeout=_TIMEOUT)
    except Exception as e:
        print(f"[youtube_api] latest videos network error: {e}")
        raise YouTubeFetchError("Network error contacting YouTube",
                                "خطأ في الاتصال بخدمة YouTube", 502)
    if r.status_code != 200:
        raise YouTubeFetchError(f"YouTube API error (HTTP {r.status_code})",
                                f"خطأ من YouTube (HTTP {r.status_code})", 502)
    return [it["id"]["videoId"] for it in (r.json() or {}).get("items", [])
            if it.get("id", {}).get("videoId")]


def _tag(comments):
    """Normalize external_sources comment dicts into the unified item shape."""
    items = []
    for c in comments or []:
        if not c.get("text"):
            continue
        vid = c.get("video_id")
        items.append({
            "external_id": c["external_id"],
            "text": c["text"],
            "author": c.get("author"),
            "created_at": c.get("created_at"),
            "likes": int(c.get("likes", 0) or 0),
            "source": "youtube",
            "url": (f"https://www.youtube.com/watch?v={vid}" if vid else None),
            "kind": "comment",
            "source_type": SOURCE_TYPE,
            "source_url": (f"https://www.youtube.com/watch?v={vid}" if vid else None),
        })
    return items


def fetch_from_url(url: str, limit: int = 150) -> dict:
    """Fetch up to `limit` comments from a YouTube video or channel URL.

    Returns {"query", "items": [...]} in the standard pipeline shape. Raises
    YouTubeFetchError with a clear bilingual reason on failure."""
    key = _key()
    limit = max(5, min(int(limit or 150), 1000))

    # 1) a direct video link?
    video_id = external_sources.extract_video_id(url)
    if video_id:
        res = external_sources.fetch_video_comments(video_id, max_comments=limit)
        return {"query": f"video:{video_id}", "items": _tag(res.get("comments"))}

    # 2) otherwise treat it as a channel link
    kind, value = _parse_channel(url)
    if not kind:
        raise YouTubeFetchError(
            "Could not read a YouTube video or channel from that link",
            "تعذّر التعرّف على فيديو أو قناة YouTube من الرابط", 400)
    channel_id = _resolve_channel_id(key, kind, value)
    if not channel_id:
        raise YouTubeFetchError("YouTube channel not found",
                                "قناة YouTube غير موجودة", 404)

    video_ids = _latest_video_ids(key, channel_id, max_videos=10)
    if not video_ids:
        raise YouTubeFetchError("No videos found on this channel",
                                "لا توجد فيديوهات على هذه القناة", 404)

    items = []
    for vid in video_ids:
        if len(items) >= limit:
            break
        try:
            res = external_sources.fetch_video_comments(vid, max_comments=limit - len(items))
            items.extend(_tag(res.get("comments")))
        except YouTubeFetchError as e:
            # one video with comments disabled / not found must not abort the rest
            print(f"[youtube_api] video {vid} skipped: {e.message_en}")
            continue
    return {"query": f"channel:{channel_id}", "items": items[:limit]}
