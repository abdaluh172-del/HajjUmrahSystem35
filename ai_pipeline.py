# -*- coding: utf-8 -*-
"""Unified AI processing pipeline (v14, extended in v15).

EVERY comment — site reviews and live comments fetched from YouTube / X /
Reddit / any future source — passes through process():

    fetch -> detect language -> translate -> sentiment -> moderation
          -> relevance -> save -> display approved only

The pipeline is tiered so it always works, and gets MORE accurate as the
site owner adds his own API keys on Render (plain env vars — no platform
lock-in, no paid connectors):

TIER A — LLM analysis (highest accuracy; recommended for production):
    Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY. ONE call per
    comment returns language, sentiment (positive/negative/neutral — with
    real understanding of context and sarcasm, always picking the single
    dominant sentiment), a topic category (see CATEGORIES below),
    moderation categories (profanity, insults, hate speech, harassment,
    racism, sexual content, violence, spam) and Hajj/Umrah relevance — all
    as strict JSON. Models: claude-haiku / gpt-4o-mini / gemini-flash class
    (fast + cheap, cents per thousand comments). If more than one key is
    set, priority is Anthropic -> OpenAI -> Gemini (see llm_configured()).

TIER B — specialized models / rules (no LLM key):
    * sentiment: sentiment.py (HF multilingual transformer if HF_API_TOKEN,
      else VADER / enhanced built-in on the English translation) — always
      one of positive/negative/neutral.
    * category: keyword-scored classification into the CATEGORIES taxonomy
      (v15) — same idea as the relevance heuristic below.
    * moderation: HF toxicity transformer if HF_API_TOKEN
      (unitary/multilingual-toxic-xlm-roberta) + local wordlists (ar+en)
      and spam heuristics — local checks always run as a safety net
    * relevance: topic heuristics on the original + English translation
      (shared list in knowledge_base.TOPIC_WORDS). Applied to EXTERNAL
      comments only (they come from broad searches); reviews written ON the
      site are presumed on-topic in this tier — only the LLM tier is
      precise enough to reject user reviews safely. v15: a Google Maps
      review whose place_type is clearly pilgrim-related (hotel for
      pilgrims, transport/Hajj/Umrah company or campaign, the Grand Mosque,
      the Prophet's Mosque, crowd management, government pilgrim services)
      is treated as relevant even if the free-text heuristic misses it.

v15 also adds a content fingerprint (dedup.py) so the SAME opinion posted on
multiple platforms (Google Maps / X / YouTube / Reddit) can be recognized
as a duplicate instead of inflating the counts — see fingerprint_for().

Translation always uses translation.py (free, all languages). Failures at
any stage degrade gracefully — a comment is never lost to an AI error.
"""
import json
import os
import re

import requests

import translation
import sentiment
import dedup
import knowledge_base

LLM_TIMEOUT = 25
MODERATION_FLAG_KEYS = ["profanity", "insult", "hate_speech", "harassment",
                        "racism", "sexual", "violence", "spam"]
# v15.1: "mixed" removed per product decision — every comment is classified
# into exactly one of these three labels now (see _LLM_PROMPT and
# _maybe_mixed below, which used to promote borderline cases to "mixed").
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# v15.1 fix: "claude-haiku-4-5" (no date suffix) is not a valid Anthropic
# model id — requests using it 404, get caught by the except below, and the
# pipeline silently falls back to the non-LLM tier on every single comment,
# even with a valid API key configured. Overridable via LLM_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# v15.3: pinned to the stable GA model "gemini-2.5-flash". The
# "gemini-flash-latest" alias can resolve to a 3.x model that rejects
# thinkingConfig.thinkingBudget=0 (HTTP 400) — which silently knocked EVERY
# comment down to the non-LLM tier (VADER on the English translation), the
# exact cause of the Arabic mis-classifications the user reported (e.g.
# "ممتاز" landing as neutral). gemini-2.5-flash accepts the thinking-off
# config; _llm_call() also retries WITHOUT it, so a 3.x model set via
# LLM_MODEL still works. Overridable via LLM_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Moderation is the whole point of this tier, so the four adjustable Gemini
# safety categories are opened up explicitly instead of left on Google's
# default: the model has to actually SEE profanity/hate/violence/sexual text
# to classify it. A silently-blocked candidate would just look like another
# failed LLM call and fall back to Tier B, defeating the point of this tier.
GEMINI_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

# ------------------------------------------------------------------ #
# v15 — smart classification into the platform's topic taxonomy.
# Internal codes are stable (used for filtering/analytics); display labels
# are translated in the frontend. "general" is the catch-all fallback and
# is always a valid choice, matching legacy data.
# ------------------------------------------------------------------ #
CATEGORIES = [
    "customer_service", "service_quality", "transportation", "accommodation",
    "cleanliness", "crowd_management", "accessibility", "haram_experience",
    "nabawi_experience", "hajj_experience", "umrah_experience", "general",
]
CATEGORY_LABELS_AR = {
    "customer_service": "التعامل وخدمة العملاء", "service_quality": "جودة الخدمات",
    "transportation": "النقل والمواصلات", "accommodation": "السكن والفنادق",
    "cleanliness": "النظافة", "crowd_management": "التنظيم وإدارة الحشود",
    "accessibility": "سهولة الوصول", "haram_experience": "تجربة الحرم المكي",
    "nabawi_experience": "تجربة المسجد النبوي", "hajj_experience": "تجربة الحج",
    "umrah_experience": "تجربة العمرة", "general": "عام",
}
CATEGORY_LABELS_EN = {
    "customer_service": "Customer Service", "service_quality": "Service Quality",
    "transportation": "Transportation", "accommodation": "Accommodation & Hotels",
    "cleanliness": "Cleanliness", "crowd_management": "Crowd Management",
    "accessibility": "Accessibility", "haram_experience": "Grand Mosque Experience",
    "nabawi_experience": "Prophet's Mosque Experience", "hajj_experience": "Hajj Experience",
    "umrah_experience": "Umrah Experience", "general": "General",
}

# ------------------------------------------------------------------ #
# v15.8 — richer analysis: intent tags, abuse types, and root causes.
# All are MULTI-LABEL (a comment can be e.g. a complaint AND a question).
# Internal codes are stable; display labels are translated below and sent to
# the frontend so old data keeps working. These are ADDITIVE — the legacy
# single `sentiment`, `category`, and `moderation_flags` fields are unchanged.
# ------------------------------------------------------------------ #
# Intent / type of the comment (what the writer is DOING).
CATEGORY_TAGS = ["question", "complaint", "suggestion", "praise",
                 "inquiry", "feedback", "issue_report", "spam"]
CATEGORY_TAG_LABELS_AR = {
    "question": "سؤال", "complaint": "شكوى", "suggestion": "اقتراح",
    "praise": "إشادة", "inquiry": "استفسار", "feedback": "ملاحظة",
    "issue_report": "بلاغ مشكلة", "spam": "محتوى دعائي",
}
CATEGORY_TAG_LABELS_EN = {
    "question": "Question", "complaint": "Complaint", "suggestion": "Suggestion",
    "praise": "Praise", "inquiry": "Inquiry", "feedback": "Feedback",
    "issue_report": "Issue Report", "spam": "Spam",
}

# Specific abuse types — far more precise than "negative comment". IMPORTANT:
# neutral mention/description of a religion or a school of thought (madhhab) is
# NOT abuse; only a real insult, attack, contempt, or incitement is. This is
# enforced both in the prompt and in _sanitize_abuse() below.
ABUSE_TYPES = ["toxic", "profanity", "insult", "harassment", "bullying",
               "threat", "violence", "hate_speech", "racism",
               "religious_hate", "sectarian_hate", "sexual", "extremism"]
ABUSE_LABELS_AR = {
    "toxic": "محتوى مؤذٍ", "profanity": "ألفاظ بذيئة", "insult": "إهانة",
    "harassment": "تحرش/مضايقة", "bullying": "تنمّر", "threat": "تهديد",
    "violence": "تحريض على العنف", "hate_speech": "خطاب كراهية",
    "racism": "عنصرية", "religious_hate": "إساءة دينية",
    "sectarian_hate": "إساءة مذهبية/طائفية", "sexual": "محتوى جنسي",
    "extremism": "محتوى متطرف",
}
ABUSE_LABELS_EN = {
    "toxic": "Toxic", "profanity": "Profanity", "insult": "Insult",
    "harassment": "Harassment", "bullying": "Bullying", "threat": "Threat",
    "violence": "Violence/Incitement", "hate_speech": "Hate Speech",
    "racism": "Racism", "religious_hate": "Religious Hate",
    "sectarian_hate": "Sectarian Hate", "sexual": "Sexual", "extremism": "Extremism",
}
# Abuse types that must NOT be assigned on the mere presence of a
# religion/sect name — they require an actual insult/attack/incitement signal.
_SENSITIVE_ABUSE = {"religious_hate", "sectarian_hate", "hate_speech", "racism"}
# Signals that a genuine attack (not neutral discussion) is present. If the LLM
# flags a sensitive abuse type but reports none of these AND no reason, we drop
# it — protecting neutral religious/sectarian discussion from false positives.
_ATTACK_SIGNAL = {"toxic", "profanity", "insult", "harassment", "bullying",
                  "threat", "violence", "extremism", "sexual"}

# Root cause of a NEGATIVE comment (why the pilgrim is unhappy). Multi-label.
CAUSES = ["crowding", "transport", "cleanliness", "accommodation", "prices",
          "organization", "permits", "services", "security", "weather",
          "apps", "delays", "mistreatment", "food", "health"]
CAUSE_LABELS_AR = {
    "crowding": "الزحام", "transport": "المواصلات", "cleanliness": "النظافة",
    "accommodation": "السكن", "prices": "الأسعار", "organization": "التنظيم",
    "permits": "التصاريح", "services": "الخدمات", "security": "الأمن",
    "weather": "الطقس", "apps": "التطبيقات", "delays": "التأخير",
    "mistreatment": "سوء المعاملة", "food": "الطعام", "health": "الصحة",
}
CAUSE_LABELS_EN = {
    "crowding": "Crowding", "transport": "Transport", "cleanliness": "Cleanliness",
    "accommodation": "Accommodation", "prices": "Prices", "organization": "Organization",
    "permits": "Permits", "services": "Services", "security": "Security",
    "weather": "Weather", "apps": "Apps", "delays": "Delays",
    "mistreatment": "Mistreatment", "food": "Food", "health": "Health",
}
_CAUSE_KEYWORDS = {
    "crowding": {"crowd", "crowded", "packed", "overcrowd", "stampede", "queue", "ازدحام", "زحمة", "تكدس", "تدافع", "زحام"},
    "transport": {"bus", "transport", "taxi", "shuttle", "traffic", "train", "نقل", "مواصلات", "حافلة", "باص", "قطار", "زحمة مرورية"},
    "cleanliness": {"clean", "dirty", "hygiene", "toilet", "trash", "smell", "نظافة", "متسخ", "قمامة", "روائح", "دورات المياه"},
    "accommodation": {"hotel", "room", "accommodation", "housing", "tent", "فندق", "غرفة", "سكن", "خيمة", "مخيم", "إقامة"},
    "prices": {"price", "expensive", "cost", "overprice", "rip off", "سعر", "أسعار", "غالي", "مبالغ", "استغلال", "تكلفة"},
    "organization": {"organization", "organize", "chaos", "management", "تنظيم", "فوضى", "إدارة", "سوء تنظيم", "عشوائية"},
    "permits": {"permit", "visa", "tasreeh", "nusuk", "تصريح", "تصاريح", "تأشيرة", "نسك", "تفويج"},
    "services": {"service", "services", "staff", "خدمة", "خدمات", "موظف", "مرافق"},
    "security": {"security", "police", "safety", "theft", "أمن", "شرطة", "سرقة", "أمان", "حماية"},
    "weather": {"heat", "hot", "weather", "sun", "cold", "حر", "حرارة", "طقس", "شمس", "برد", "قيظ"},
    "apps": {"app", "application", "website", "online", "system", "تطبيق", "موقع", "النظام", "الكتروني", "أبشر", "نسك"},
    "delays": {"delay", "late", "wait", "slow", "تأخير", "تأخر", "انتظار", "بطيء", "متأخر"},
    "mistreatment": {"rude", "ignored", "mistreat", "disrespect", "سوء معاملة", "قلة احترام", "تجاهل", "وقاحة", "إهمال"},
    "food": {"food", "meal", "eat", "hungry", "restaurant", "طعام", "أكل", "وجبة", "مطعم", "جوع"},
    "health": {"health", "sick", "hospital", "medical", "clinic", "صحة", "مريض", "مستشفى", "إسعاف", "طبي", "عيادة"},
}

# Question detection for the non-LLM fallback tier (the LLM decides on its own).
_QUESTION_WORDS = {
    # Arabic
    "كيف", "أين", "وين", "متى", "هل", "كم", "لماذا", "ليش", "ماذا", "ايش", "أيش",
    "وش", "من", "أي", "هل يجوز", "ما حكم", "ممكن", "هل يمكن", "وينه", "متي",
    # English
    "how", "where", "when", "what", "why", "who", "which", "can i", "is it",
    "are there", "do i", "does", "should i", "could", "would",
}


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "؟" in t or "?" in t:
        return True
    head = t[:40]
    return any(head.startswith(w) or (" " + w + " ") in (" " + t + " ") for w in _QUESTION_WORDS)


def _detect_causes(text: str, text_en: str = "") -> list:
    hay = ((text or "") + " " + (text_en or "")).lower()
    return [c for c, words in _CAUSE_KEYWORDS.items() if any(w in hay for w in words)]


def _sanitize_abuse(abuse_types: list, reason: str) -> list:
    """Guard against false positives on religion/sect: a sensitive abuse type
    (religious/sectarian/hate/racism) is kept ONLY when a real attack signal is
    also present or the model gave an explicit reason. Neutral discussion or
    description that merely names a religion or madhhab is NOT abuse."""
    types = [a for a in (abuse_types or []) if a in ABUSE_TYPES]
    if not types:
        return []
    has_attack = any(a in _ATTACK_SIGNAL for a in types)
    has_reason = bool((reason or "").strip())
    cleaned = []
    for a in types:
        if a in _SENSITIVE_ABUSE and not has_attack and not has_reason:
            continue  # drop lone religion/sect flag with no attack evidence
        cleaned.append(a)
    return list(dict.fromkeys(cleaned))

_CATEGORY_KEYWORDS = {
    "customer_service": {"staff", "employee", "employees", "customer service", "service desk",
                          "rude", "friendly", "helpful", "unprofessional", "welcoming", "hospitality",
                          "attitude", "ignored", "reception desk",
                          "موظف", "موظفين", "الموظفين", "التعامل", "أسلوب التعامل", "خدمة العملاء",
                          "استقبال", "مهذب", "متعاون", "متعاونين", "ترحيب", "ضيافة", "احترام",
                          "قلة احترام", "وقح", "وقحين", "تجاهل", "تجاهلونا", "اهتمام"},
    "service_quality": {"quality", "service", "professional", "standard", "level of service",
                         "excellent service", "poor service",
                         "جودة", "الخدمة", "الخدمات", "احترافي", "احترافية", "مستوى", "مستوى الخدمة",
                         "جودة عالية", "جودة رديئة", "إتقان", "خدمة ممتازة", "خدمة سيئة"},
    "transportation": {"bus", "transport", "transportation", "taxi", "shuttle", "traffic", "driver",
                        "vehicle", "route", "train",
                        "نقل", "مواصلات", "حافلة", "الحافلات", "باص", "تاكسي", "سائق", "قطار",
                        "خط سير", "محطة", "توصيل", "مسار", "ازدحام مروري", "زحمة مرورية",
                        "مواقف", "مواقف سيارات"},
    "accommodation": {"hotel", "room", "accommodation", "stay", "bed", "housing", "residence",
                       "check-in", "check-out", "suite",
                       "فندق", "غرفة", "غرف", "سكن", "إقامة", "سرير", "سكن الحجاج", "شقة",
                       "استراحة", "تسجيل الدخول", "فندقي"},
    "cleanliness": {"clean", "dirty", "hygiene", "hygienic", "unclean", "toilet", "trash", "garbage",
                     "smell",
                     "نظافة", "نظافة عامة", "نظيف", "متسخ", "دورات المياه", "قمامة", "زبالة",
                     "روائح", "رائحة كريهة", "صرف صحي", "تنظيف"},
    "crowd_management": {"crowd", "crowded", "packed", "overcrowding", "queue", "line", "waiting",
                          "organized", "chaos", "stampede",
                          "ازدحام", "ازدحام شديد", "زحمة", "تكدس", "تنظيم", "فوضى", "طابور",
                          "طوابير طويلة", "تدافع", "زحمة خانقة", "انتظار طويل"},
    "accessibility": {"wheelchair", "elderly", "disability", "disabled", "access", "special needs",
                       "ramp", "elevator", "senior citizens",
                       "كبار السن", "ذوي الإعاقة", "عربات", "سهولة الوصول", "إعاقة", "كرسي متحرك",
                       "منحدر", "مصعد", "ذوي الاحتياجات الخاصة", "المسنين"},
    "haram_experience": {"grand mosque", "kaaba", "tawaf", "haram", "black stone", "zamzam",
                          "المسجد الحرام", "الكعبة", "طواف", "الحرم المكي", "المطاف",
                          "الحجر الأسود", "زمزم", "ساحات الحرم"},
    "nabawi_experience": {"prophet's mosque", "nabawi", "rawdah", "green dome", "prophet's tomb",
                           "المسجد النبوي", "الروضة", "الحرم النبوي", "القبة الخضراء",
                           "الحجرة الشريفة", "باب السلام"},
    "hajj_experience": {"hajj", "arafah", "mina", "muzdalifah", "jamarat", "stoning", "pilgrims",
                         "الحج", "عرفة", "الوقوف بعرفة", "منى", "مزدلفة", "الجمرات", "رمي الجمرات",
                         "الحجاج", "طواف الإفاضة"},
    "umrah_experience": {"umrah", "umrah package", "umrah visa",
                          "العمرة", "المعتمرين", "معتمر", "عمرتي", "برنامج العمرة", "تأشيرة العمرة"},
}


def classify_category(text: str, text_en: str = "") -> str:
    """Heuristic keyword-scored classification (TIER B — no LLM key
    needed). Picks the category with the most keyword hits; "general" when
    nothing scores. The LLM tier overrides this with real understanding
    when a key is configured (see _LLM_PROMPT)."""
    hay = ((text or "") + " " + (text_en or "")).lower()
    best_cat, best_score = "general", 0
    for cat, words in _CATEGORY_KEYWORDS.items():
        score = sum(1 for w in words if w in hay)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat

_LLM_PROMPT = """You are a strict JSON content-analysis service for a Hajj & Umrah pilgrimage feedback website. Read the comment and understand its FULL meaning — context, negation ("not good" = negative), sarcasm, mixed positive/negative, Arabic dialects, slang, spelling mistakes, and emoji (😊❤️ positive, 😡😢 negative). Reply with ONLY a JSON object, no other text:
{
 "language": "<ISO 639-1 code of the comment's language>",
 "sentiment": "positive"|"negative"|"neutral",
 "sentiment_confidence": <0-100>,
 "sentiment_reason": "<short reason in the comment's language for the sentiment>",
 "category": "<one of: """ + ",".join(CATEGORIES) + """>",
 "category_tags": [<zero or more of: """ + ",".join(CATEGORY_TAGS) + """>],
 "is_question": true|false,
 "ai_answer": "<if is_question is true: a concise, accurate, helpful answer to the question IN THE SAME LANGUAGE as the comment, based on general Hajj/Umrah knowledge; otherwise empty string>",
 "abuse_types": [<zero or more of: """ + ",".join(ABUSE_TYPES) + """>],
 "causes": [<if sentiment is negative, zero or more root causes from: """ + ",".join(CAUSES) + """; else empty>],
 "flags": [<zero or more of: "profanity","insult","hate_speech","harassment","racism","sexual","violence","spam">],
 "relevant": true|false,
 "reason": "<short reason in Arabic if flagged or irrelevant, else empty string>"
}
Rules:
- sentiment: always pick exactly ONE dominant label. Pure factual statements with no opinion are "neutral". Brevity is not neutrality: a single clear word ("ممتاز","excellent","سيئ","bad") gets that sentiment with high confidence.
- category = the single best topic. category_tags = what the writer is DOING (a comment can be several, e.g. complaint + issue_report, or question + inquiry).
- is_question/ai_answer: set is_question true if the comment asks anything (question words or ؟/?). Then answer it helpfully in the SAME language. If not a question, ai_answer = "".
- abuse_types: be PRECISE and name the specific type, not just "negative". CRITICAL FAIRNESS RULE: simply mentioning, naming, describing, or discussing a religion, a school of thought (madhhab), or a sect is NOT abuse. Only assign religious_hate / sectarian_hate / hate_speech / racism when there is a REAL insult, contempt, dehumanization, or incitement AGAINST people for their religion/sect/race. Neutral, factual, or scholarly discussion — even disagreement — is NOT hate. When unsure, do NOT flag.
- causes: only for negative comments; list every applicable root cause.
- Flag ONLY clear violations; ordinary criticism, even harsh, is NOT a violation.
- relevant = true when the comment concerns the Hajj/Umrah journey in ANY way (rituals, holy sites, crowds, organization, transport, accommodation, a Hajj/Umrah company/campaign, food, or services during pilgrimage). Off-topic content (random ads, unrelated products) = false.
Comment:
\"\"\"{TEXT}\"\"\""""


# ------------------------------------------------------------------ #
# TIER A — one LLM call analyzes everything (user's own key)
# ------------------------------------------------------------------ #
# v15.7: same API-key hygiene as assistant.py — strip whitespace/quotes and
# remove any CR/LF/TAB that would make an HTTP header value invalid
# ("Invalid header value b'Bearer ...'") and silently push every comment down
# to the non-LLM tier. Each provider keeps using its OWN key only.
def _clean_key(raw) -> str:
    if not raw:
        return ""
    key = str(raw).strip().strip('"').strip("'").strip()
    return key.replace("\r", "").replace("\n", "").replace("\t", "")


def _env_key(name: str, *fallbacks: str) -> str:
    for n in (name, *fallbacks):
        v = _clean_key(os.environ.get(n))
        if v:
            return v
    return ""


def _gemini_generate(g_key: str, prompt: str) -> str:
    """Call Gemini for the classification prompt and return the raw text.
    Model-agnostic: attempt 1 turns thinking off (cheap, correct on
    gemini-2.5-flash); attempt 2 drops thinkingConfig and widens the budget
    because newer 3.x models reject thinkingBudget=0 with an HTTP 400 — the
    exact failure that used to knock every comment down to the weaker
    non-LLM tier. Raises on total failure so _llm_call() logs + falls back.
    """
    model = os.environ.get("LLM_MODEL", DEFAULT_GEMINI_MODEL)
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    attempts = [
        {"maxOutputTokens": 900, "responseMimeType": "application/json",
         "thinkingConfig": {"thinkingBudget": 0}},
        {"maxOutputTokens": 1600, "responseMimeType": "application/json"},
    ]
    last_detail = "no response"
    for i, gen_config in enumerate(attempts):
        r = requests.post(
            url,
            headers={"x-goog-api-key": g_key, "content-type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": gen_config,
                  "safetySettings": GEMINI_SAFETY_SETTINGS},
            timeout=LLM_TIMEOUT)
        if not r.ok:
            last_detail = f"HTTP {r.status_code}: {r.text[:300]}"
            print(f"[pipeline] Gemini API error {r.status_code} "
                  f"(attempt {i + 1}/{len(attempts)}): {r.text[:500]}")
            continue
        candidates = r.json().get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text = "".join(p.get("text", "") for p in parts).strip()
        if text:
            return text
        last_detail = "empty text"
        print(f"[pipeline] Gemini returned empty text (attempt {i + 1}/{len(attempts)})")
    raise RuntimeError(f"Gemini call failed: {last_detail}")


def _llm_call(text: str):
    """Returns the parsed JSON dict from Anthropic, OpenAI or Gemini, or None."""
    prompt = _LLM_PROMPT.replace("{TEXT}", text[:4000])
    a_key = _env_key("ANTHROPIC_API_KEY")
    o_key = _env_key("OPENAI_API_KEY")
    g_key = _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    try:
        if a_key:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": a_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": os.environ.get("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL),
                      "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=LLM_TIMEOUT)
            if not r.ok:
                print(f"[pipeline] Anthropic API error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            raw = "".join(b.get("text", "") for b in r.json().get("content", []))
        elif o_key:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {o_key}"},
                json={"model": os.environ.get("LLM_MODEL", DEFAULT_OPENAI_MODEL),
                      "max_tokens": 800,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=LLM_TIMEOUT)
            if not r.ok:
                print(f"[pipeline] OpenAI API error {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        elif g_key:
            raw = _gemini_generate(g_key, prompt)
        else:
            return None
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
        data = json.loads(raw)
        if data.get("sentiment") not in SENTIMENT_LABELS:
            return None
        if data.get("category") not in CATEGORIES:
            data["category"] = None  # process() falls back to the heuristic classifier
        return data
    except Exception as e:
        print(f"[pipeline] LLM analysis failed, falling back: {e}")
        return None


def llm_configured() -> str:
    if _env_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _env_key("OPENAI_API_KEY"):
        return "openai"
    if _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return "gemini"
    return ""


def providers_configured() -> dict:
    """v15.2: unlike llm_configured() (the single WINNING provider under the
    Anthropic -> OpenAI -> Gemini priority order), this reports each
    provider's key independently — e.g. so the admin panel can show "Gemini:
    configured" even while Anthropic is the one actually answering, the same
    way the External Sources panel shows YouTube/Reddit/X/Google Maps side
    by side regardless of which ones are actually active."""
    return {
        "anthropic": bool(_env_key("ANTHROPIC_API_KEY")),
        "openai": bool(_env_key("OPENAI_API_KEY")),
        "gemini": bool(_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    }


# ------------------------------------------------------------------ #
# TIER B moderation — HF toxicity model + local wordlists / heuristics
# ------------------------------------------------------------------ #
HF_TOX_MODEL = "unitary/multilingual-toxic-xlm-roberta"


def _hf_toxicity(text: str):
    token = _env_key("HF_API_TOKEN")
    if not token:
        return None
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_TOX_MODEL}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text[:1500], "options": {"wait_for_model": True}},
            timeout=20)
        r.raise_for_status()
        data = r.json()
        cand = data[0] if data and isinstance(data[0], list) else data
        for item in cand:
            if str(item.get("label", "")).lower() in ("toxic", "toxicity", "label_1"):
                return float(item.get("score", 0))
        return 0.0
    except Exception as e:
        print(f"[pipeline] toxicity model failed: {e}")
        return None


# Local moderation wordlists — a safety net that always runs, extensible via
# the EXTRA_BANNED_WORDS env var (comma-separated). Kept intentionally
# conservative: normal harsh criticism must never be flagged.
_PROFANITY = {
    # English
    "fuck", "fucking", "shit", "bitch", "asshole", "bastard", "dick", "cunt",
    "whore", "slut", "motherfucker", "porn", "nude",
    # Arabic (common explicit insults)
    "كلب", "حمار", "حقير", "قذر", "وسخ يا", "يلعن", "تفو", "زبالة", "خنزير",
    "حيوان يا", "غبي يا", "عاهرة", "قحبة", "زانية", "ابن الكلب", "يا خول",
}
_HATE_VIOLENCE = {
    "kill you", "i will kill", "deserve to die", "exterminate", "terrorist scum",
    "سأقتلك", "اقتلوهم", "يستاهلون الموت", "ابادة", "اذبحوهم",
}
_SPAM_PATTERNS = [
    re.compile(r"(https?://\S+.*){2,}", re.S),          # 2+ links
    re.compile(r"(whatsapp|واتساب|واتس اب).{0,30}\+?\d{8,}", re.I),
    re.compile(r"(اربح|ربح مضمون|win money|crypto|forex|promo code|discount code)", re.I),
    re.compile(r"(.)\1{9,}"),                            # aaaaaaaaaa spam
]


def _local_moderation(text: str, text_en: str):
    """Returns (flags, reason) from wordlists + heuristics."""
    flags, reason = [], ""
    hay = (text + " " + (text_en or "")).lower()
    extra = {w.strip().lower() for w in os.environ.get("EXTRA_BANNED_WORDS", "").split(",") if w.strip()}
    if any(w in hay for w in _PROFANITY | extra):
        flags.append("profanity")
        reason = "ألفاظ غير لائقة"
    if any(w in hay for w in _HATE_VIOLENCE):
        flags.append("violence")
        reason = "تهديد أو تحريض على العنف"
    for pat in _SPAM_PATTERNS:
        if pat.search(text):
            flags.append("spam")
            reason = reason or "محتوى دعائي/سبام"
            break
    return list(dict.fromkeys(flags)), reason


# ------------------------------------------------------------------ #
# TIER B relevance — topic heuristics (EXTERNAL comments only)
# ------------------------------------------------------------------ #
# v15: the topic-word list is now shared with assistant.py's scope guard,
# defined once in knowledge_base.TOPIC_WORDS.
def _heuristic_relevant(text: str, text_en: str) -> bool:
    return knowledge_base.in_scope(text) or knowledge_base.in_scope(text_en or "")


# v15: place types that make a Google Maps review relevant by definition,
# per the product spec (hajj/umrah experience, pilgrim hotel, pilgrim
# transport company, hajj/umrah company, hajj campaign, the Grand Mosque,
# the Prophet's Mosque, the sacred sites, crowd management, government
# pilgrim services) — bypasses the free-text heuristic, which can miss a
# short review like "Great stay!" that has no Hajj/Umrah keyword in it even
# though the PLACE itself is unambiguously pilgrim-related.
RELEVANT_PLACE_TYPES = {
    "hajj_experience", "umrah_experience", "pilgrim_hotel", "pilgrim_transport",
    "hajj_umrah_company", "hajj_campaign", "grand_mosque", "prophet_mosque",
    "sacred_sites", "crowd_management", "government_pilgrim_service",
}


def _place_type_relevant(place_type: str) -> bool:
    return bool(place_type) and place_type.strip().lower() in RELEVANT_PLACE_TYPES


# ------------------------------------------------------------------ #
# The pipeline
# ------------------------------------------------------------------ #
def _synthesize_scores(label: str, confidence: float) -> dict:
    """The LLM tier only returns the dominant label + a confidence number,
    not a full distribution — but the UI's percentage bars expect one for
    all four labels. Give the dominant label its confidence and split the
    remainder evenly across the other three so the bars are never blank."""
    confidence = max(0.0, min(100.0, confidence))
    remainder = round((100.0 - confidence) / 3, 1)
    scores = {lbl: remainder for lbl in SENTIMENT_LABELS}
    scores[label] = round(confidence, 1)
    return scores


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fallback_category_tags(label: str, is_question: bool, flags: list) -> list:
    tags = []
    if is_question:
        tags += ["question", "inquiry"]
    if "spam" in (flags or []):
        tags.append("spam")
    if label == "negative":
        tags.append("complaint")
    elif label == "positive":
        tags.append("praise")
    else:
        tags.append("feedback")
    return list(dict.fromkeys(tags))


def _fallback_answer_for_question(text: str, lang: str = "ar") -> str:
    """When there's no LLM to answer a detected question, offer a short answer
    grounded in the internal knowledge base if the question is on-topic; else
    return "" (the UI simply shows no AI answer). Never raises."""
    try:
        lang = "ar" if (lang or "ar").startswith("ar") else "en"
        if not knowledge_base.in_scope(text):
            return ""
        block = (knowledge_base.context_block(text, lang=lang, limit=1) or "").strip()
        if not block:
            return ""
        prefix = "بحسب قاعدة المعرفة: " if lang == "ar" else "From the knowledge base: "
        return (prefix + block)[:2000]
    except Exception:
        return ""


def process(text: str, ml_predict=None, is_external: bool = False, place_type: str = None) -> dict:
    """Run the FULL pipeline on one comment. Never raises.

    place_type (optional): for external reviews with a known place type
    (e.g. from Google Maps — see ai_pipeline.RELEVANT_PLACE_TYPES) — lets a
    pilgrim-related place count as relevant even when the review text alone
    doesn't mention Hajj/Umrah keywords ("Great stay!" at a pilgrim hotel).

    Returns:
      detected_language, text_ar, sentiment, confidence, scores, engine,
      category, moderation_status ('approved'|'flagged'|'rejected'),
      moderation_flags (list), moderation_reason (str), relevant (bool),
      content_fingerprint (str) — for cross-source de-duplication.
    """
    text = (text or "").strip()
    # -- translation first: the display copy AND the analysis input --
    tr_ar = translation.detect_and_translate(text, target="ar")
    detected = tr_ar["detected_lang"]
    text_ar = None
    if tr_ar["ok"] and detected and detected != "ar" and tr_ar["translated"] != text:
        text_ar = tr_ar["translated"]
    text_en = translation.to_english(text)

    llm = _llm_call(text)
    if llm is not None:
        # ---- TIER A: LLM understood the comment ----
        provider = llm_configured()
        detected = llm.get("language") or detected
        flags = [f for f in (llm.get("flags") or []) if f in MODERATION_FLAG_KEYS]
        relevant = bool(llm.get("relevant", True)) or _place_type_relevant(place_type)
        confidence = round(float(llm.get("sentiment_confidence", 80)), 1)
        # v15.9.1 SENTIMENT SAFETY-NET (sentiment-analysis fix only):
        # the LLM sometimes labels a short, clearly-polar comment as "neutral"
        # (e.g. a bare "ممتاز"), which the UI then shows as 0/0/100. When our
        # deterministic lexicon engine is confident the text is positive or
        # negative, adopt ITS label + real score distribution. This is fully
        # data-driven from the lexicon's own score for whatever the text is —
        # NOT a fixed per-word rule — and it only rescues a "neutral" verdict;
        # it never flips positive<->negative. Everything else below (category,
        # moderation, abuse, relevance, ai_answer) is left exactly as-is.
        sentiment_label = llm["sentiment"]
        scores = _synthesize_scores(sentiment_label, confidence)
        if sentiment_label == "neutral":
            _lex = sentiment.analyze(text, ml_predict=ml_predict)
            if _lex.get("label") in ("positive", "negative") and _lex.get("confidence", 0) >= 65:
                sentiment_label = _lex["label"]
                confidence = round(float(_lex["confidence"]), 1)
                scores = _lex.get("scores") or _synthesize_scores(sentiment_label, confidence)
        category = llm.get("category") or classify_category(text, text_en)
        # new multi-label fields (validated against the taxonomies)
        cat_tags = [t for t in (llm.get("category_tags") or []) if t in CATEGORY_TAGS]
        abuse = _sanitize_abuse(llm.get("abuse_types") or [], llm.get("reason"))
        causes = [c for c in (llm.get("causes") or []) if c in CAUSES] if sentiment_label == "negative" else []
        is_q = bool(llm.get("is_question")) or _looks_like_question(text)
        ai_answer = (llm.get("ai_answer") or "").strip() if is_q else ""
        result = {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "scores": scores,
            "engine": "llm-" + provider,
            "category": category,
            "moderation_flags": flags,
            "moderation_reason": (llm.get("reason") or "")[:300],
            "relevant": relevant,
            # --- v15.8 enriched, additive fields ---
            "sentiment_reason": (llm.get("sentiment_reason") or "")[:300],
            "category_tags": cat_tags,
            "abuse_types": abuse,
            "causes": causes,
            "is_question": is_q,
            "ai_answer": ai_answer[:2000],
            "analysis_mode": "llm",
            "model_used": (os.environ.get("LLM_MODEL")
                           or {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
                               "gemini": DEFAULT_GEMINI_MODEL}.get(provider)),
        }
    else:
        # ---- TIER B: LLM unavailable / 401 / 429 / quota — never stop, ----
        # ---- fall back to the traditional engine and fill the SAME shape --
        print("[pipeline] analysis_mode=fallback (LLM unavailable) — using traditional engine")
        s = sentiment.analyze(text, ml_predict=ml_predict)
        label = s["label"]
        flags, reason = _local_moderation(text, text_en)
        tox = _hf_toxicity(text)
        if tox is not None and tox >= 0.80 and "profanity" not in flags:
            flags.append("insult")
            reason = reason or "محتوى مسيء (نموذج كشف السمية)"
        relevant = (_heuristic_relevant(text, text_en) or _place_type_relevant(place_type)) if is_external else True
        is_q = _looks_like_question(text)
        # map local moderation flags -> abuse types; keep the religion/sect guard
        abuse = _sanitize_abuse(["profanity" if "profanity" in flags else None,
                                 "insult" if "insult" in flags else None,
                                 "violence" if "violence" in flags else None,
                                 "spam" if "spam" in flags else None], reason)
        causes = _detect_causes(text, text_en) if label == "negative" else []
        cat_tags = _fallback_category_tags(label, is_q, flags)
        result = {
            "sentiment": label,
            "confidence": s["confidence"],
            "scores": s.get("scores", {}),
            "engine": s.get("engine"),
            "category": classify_category(text, text_en),
            "moderation_flags": flags,
            "moderation_reason": reason,
            "relevant": relevant,
            # --- v15.8 enriched, additive fields (heuristic tier) ---
            "sentiment_reason": "",
            "category_tags": cat_tags,
            "abuse_types": abuse,
            "causes": causes,
            "is_question": is_q,
            # no LLM to answer with; offer a short KB-grounded answer if in scope
            "ai_answer": _fallback_answer_for_question(text, detected) if is_q else "",
            "analysis_mode": "fallback",
            "model_used": None,
        }

    result["analyzed_at"] = _now_iso()
    if not result["relevant"]:
        result["moderation_status"] = "rejected"
        result["moderation_reason"] = result["moderation_reason"] or "غير متعلق بالحج والعمرة"
    elif result["moderation_flags"] or result.get("abuse_types"):
        result["moderation_status"] = "flagged"
    else:
        result["moderation_status"] = "approved"
    result["detected_language"] = detected
    result["text_ar"] = text_ar
    # v15: fingerprint on the English translation so the SAME opinion posted
    # in different languages on different platforms still matches.
    result["content_fingerprint"] = dedup.fingerprint(text_en or text)
    return result


def pipeline_status() -> dict:
    provider = llm_configured()
    default_model = {"anthropic": DEFAULT_ANTHROPIC_MODEL, "openai": DEFAULT_OPENAI_MODEL,
                      "gemini": DEFAULT_GEMINI_MODEL}.get(provider)
    return {
        "llm_provider": provider,
        "llm_model": os.environ.get("LLM_MODEL") or default_model,
        "llm_providers_configured": providers_configured(),
        "toxicity_model_enabled": bool(os.environ.get("HF_API_TOKEN")),
        "categories": CATEGORIES,
        "sentiment_labels": SENTIMENT_LABELS,
        "category_tags": CATEGORY_TAGS,
        "abuse_types": ABUSE_TYPES,
        "causes": CAUSES,
        **sentiment.engine_status(),
    }
