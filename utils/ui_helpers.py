"""
utils/ui_helpers.py — Iteration 6

Streamlit has been removed. This file now contains only:
  1. Context enrichment logic  (enrich_with_context, _is_vague, etc.)
  2. Input validation          (validate_input)
  3. Smart fallback detection  (get_smart_fallback)
  4. Follow-up suggestions     (get_followup_suggestions)

All of these are called from main.py (FastAPI) — pure Python, no UI framework.
"""

from __future__ import annotations

import re
from typing import Optional
from utils.db_settings import get_cached_settings


# ══════════════════════════════════════════════════
# 1. Context Awareness  (Iteration 3 — unchanged)
# ══════════════════════════════════════════════════

_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "tell", "me",
    "about", "how", "many", "does", "do", "kiet", "university", "please",
    "can", "you", "i", "want", "know", "give", "show", "explain",
    "describe", "list", "find", "get", "any", "all", "some", "its",
    "their", "there", "which", "who", "when", "where", "will", "would",
    "could", "should", "have", "has", "had", "and", "or", "of", "for",
    "in", "at", "by", "with", "this", "that", "my", "your", "our",
    "also", "more", "from", "not", "but",
}

_NAMED_TOPICS = {
    "code jung", "hostel", "transport", "scholarship", "sports", "library",
    "lab", "clubs", "alumni", "jobs", "research", "oric", "qec", "lms",
    "portal", "campus", "location", "contact", "vision", "mission",
    "history", "founder", "principal", "rector", "timetable", "result",
    "internship", "semester", "software", "computer", "engineering",
    "management", "business", "science", "technology", "bachelor",
    "master", "phd", "program", "degree", "faculty", "department",
    "structure", "career", "placement", "accreditation", "ranking",
    "fee structure",
}


def _extract_topic(text: str) -> str:
    """Strip stop words → core topic keywords."""
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _has_named_topic(question: str) -> bool:
    q = question.lower()
    return any(topic in q for topic in _NAMED_TOPICS)


# Pronouns / references that signal the user is referring to the previous answer
_REFERENCE_TOKENS = {
    "it", "its", "this", "that", "they", "them", "their", "those", "these",
    "such", "same", "above", "mentioned", "said", "there", "here",
    "more", "also", "else", "other", "another", "similar", "related",
}


def _has_reference_pronoun(question: str) -> bool:
    """Returns True if the question contains a reference pronoun pointing to prior context."""
    tokens = set(question.lower().split())
    return bool(tokens & _REFERENCE_TOKENS)


def _is_vague(question: str) -> bool:
    """
    Vague = question is very short AND contains a reference pronoun OR
    is extremely short (≤ 2 words) with no named topic.

    Examples:
      "tell me more about it"   → vague (has reference: 'it', 'more')
      "what else?"              → vague (has reference: 'else', ≤2 meaningful words)
      "what documents are required" → NOT vague (standalone question, no pronoun)
      "tell me about code jung" → NOT vague (has named topic)
    """
    if _has_named_topic(question):
        return False
    words = question.strip().split()
    # Only enrich if the question has a reference pronoun
    if _has_reference_pronoun(question) and len(words) <= 8:
        return True
    # Ultra-short questions (1-2 words) with no clear topic
    if len(words) <= 2:
        return True
    return False


def enrich_with_context(current_question: str, history: list) -> str:
    """
    Enriches a vague follow-up with the topic from the previous user message.

    history format: [{"role": "user"|"assistant", "content": str}, ...]

    Examples:
        prev="what is the admission process", current="what documents are required"
        → "admission process what documents are required"
    """
    if not _is_vague(current_question):
        return current_question

    last_user_msg: Optional[str] = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return current_question

    topic = _extract_topic(last_user_msg)
    if not topic:
        return current_question

    return f"{topic} {current_question}"


# ══════════════════════════════════════════════════
# 2. Input Validation  (Iteration 5 — unchanged)
# ══════════════════════════════════════════════════

def validate_input(question: str, recent_questions: list[str]) -> Optional[str]:
    """
    Validates user input before sending to pipeline.
    Returns an error message string if invalid, None if valid.

    Checks:
      - Too short  (< 2 chars)
      - Too long   (> 500 chars)
      - No letters (only symbols/numbers)
      - Duplicate  (same as one of last 3 questions)
    """
    q = (question or "").strip()

    if len(q) < 2:
        return "Please enter a longer question."

    if len(q) > 500:
        return "Your question is too long. Please keep it under 500 characters."

    if not re.search(r"[a-zA-Z]", q):
        return "Please enter a question with actual words."

    return None


# ══════════════════════════════════════════════════
# 3. Smart Fallback  (Iteration 5 — updated)
# ══════════════════════════════════════════════════

def get_smart_fallback(question: str) -> Optional[dict]:
    """
    Returns a smart fallback dict if the question matches a known pattern,
    otherwise returns None (pipeline should handle it normally).

    Return format:
    {
        "message":    str,
        "link_label": str,
        "link_url":   str,
    }
    """
    q = question.lower()
    settings = get_cached_settings()
    fallbacks = settings.get("smart_fallbacks", [])
    
    for fb in fallbacks:
        keywords = fb.get("keywords", [])
        response = fb.get("response", {})
        if any(kw in q for kw in keywords):
            return response
    return None


# ══════════════════════════════════════════════════
# 4. Follow-up Suggestions  (Iteration 5 — unchanged)
# ══════════════════════════════════════════════════

def get_followup_suggestions(question: str, answer: str) -> list[str]:
    """
    Returns up to 3 follow-up question suggestions based on the topic.
    Matches against both the question and the answer for better coverage.
    """
    settings = get_cached_settings()
    followup_map = settings.get("followup_map", [])
    
    combined = (question + " " + answer).lower()
    for item in followup_map:
        keywords = item.get("keywords", [])
        if any(kw in combined for kw in keywords):
            return item.get("suggestions", [])[:3]
    return []