from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("eduassist.summarizing_agent")

# ══════════════════════════════════════════════════════════════
# Prompt Template
# ══════════════════════════════════════════════════════════════
# Only reaches the LLM when the answer is long AND unstructured.
# Prompt primed with "•" so the model continues in bullet format.

SUMMARIZE_PROMPT = """You are a helpful university assistant summarizing information for students.

Convert the following answer into a clear, student-friendly bullet-point summary.

Rules:
- Use simple, easy-to-understand language.
- Present information as short bullet points starting with "•" symbol.
- Each bullet point should cover ONE key fact.
- Do NOT add any information not present in the original answer.
- Do NOT use headers or bold text.
- Maximum 6 bullet points.

Original Answer:
{answer}

Bullet-Point Summary:
•"""


class SummarizingAgent:
    """
    Summarizing Agent — Iteration 7 (Refined)

    Decision tree — exactly one path is taken per answer:

    1. Empty / too short      → return as-is (nothing to do)
    2. "Not available"        → wrap in single bullet, return
    3. Already bulleted       → validate quality:
         a. Bullet count ≤ max AND avg length reasonable → return as-is (NO re-processing)
         b. Too many bullets OR bullets are paragraph-length → trim / reformat
    4. Short prose (≤ 60 words, ≤ 3 sentences) → light sentence-split, no LLM
    5. Medium prose (61–120 words)             → sentence-split, no LLM
    6. Long unstructured prose (> 120 words)   → LLM summarization
         └─ LLM fails / returns garbage        → sentence-split fallback

    Key principle:
        Already-structured answers are NEVER sent to the LLM.
        The LLM is called ONLY for genuinely long unstructured prose.
    """

    NOT_AVAILABLE_PHRASE = "the information is not available in the current data"

    # Bullet characters recognised as "already formatted"
    _BULLET_RE = re.compile(r"^[•\-\*\u2022\u2023\u25e6]\s*")
    _NUMBERED_RE = re.compile(r"^\d+[.)]\s+")
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        llm_agent,
        *,
        max_bullets: int = 6,
        # Words threshold above which LLM summarization is triggered
        long_answer_word_threshold: int = 120,
        # Words threshold below which sentence-split is used (no LLM)
        short_answer_word_threshold: int = 60,
        # Avg words per bullet above which bullets are considered "verbose" and trimmed
        verbose_bullet_threshold: int = 40,
        fallback_to_original: bool = True,
        debug: bool = False,
    ):
        self.llm = llm_agent
        self.max_bullets = max_bullets
        self.long_threshold  = long_answer_word_threshold
        self.short_threshold = short_answer_word_threshold
        self.verbose_bullet_threshold = verbose_bullet_threshold
        self.fallback_to_original = fallback_to_original
        self.debug = debug

    # ══════════════════════════════════════════════════════════
    # Public Entry Point
    # ══════════════════════════════════════════════════════════

    def summarize(self, answer: str) -> str:
        """
        Main entry point. Returns the answer in the most appropriate
        format without over-processing already-good answers.
        """
        answer = (answer or "").strip()

        # ── Path 1: Empty or too short to be meaningful ───────
        if not answer or len(answer) < 10:
            logger.debug("[Summarizer] Path 1: empty/trivial — returned as-is")
            return answer

        # ── Path 2: "Not available" — single bullet, done ────
        if self.NOT_AVAILABLE_PHRASE in answer.lower():
            logger.debug("[Summarizer] Path 2: not-available answer — single bullet")
            return f"• {answer}"

        # ── Path 3: Already structured (bullets / numbered) ──
        existing = self._extract_bullets(answer)
        if existing:
            return self._handle_existing_bullets(answer, existing)

        # ── Word count for prose paths ────────────────────────
        word_count = len(answer.split())

        # ── Path 4 & 5: Short / medium prose — no LLM needed ─
        if word_count <= self.long_threshold:
            logger.debug("[Summarizer] Path 4/5: prose %d words — sentence-split (no LLM)", word_count)
            return self._sentence_split(answer)

        # ── Path 6: Long unstructured prose — LLM summarize ──
        logger.debug("[Summarizer] Path 6: long prose %d words — LLM summarization", word_count)
        try:
            result = self._llm_summarize(answer)
            if result and len(self._extract_bullets(result)) >= 1:
                logger.debug("[Summarizer] Path 6: LLM summarization succeeded")
                return result
        except Exception as e:
            logger.warning("[Summarizer] LLM summarize error: %s", e)

        # ── Path 6 fallback: LLM failed, sentence-split ──────
        logger.debug("[Summarizer] Path 6 fallback: sentence-split")
        if self.fallback_to_original:
            return self._sentence_split(answer)
        return answer

    # ══════════════════════════════════════════════════════════
    # Path 3 Handler — Already Bulleted
    # ══════════════════════════════════════════════════════════

    def _handle_existing_bullets(self, original: str, bullets: list[str]) -> str:
        """
        Smart handling for answers that are already in bullet/list format.

        Cases:
          A. Bullet count ≤ max AND bullets are concise
             → Return as-is. Zero processing. This is the happy path.

          B. Bullet count > max
             → Trim to max_bullets. No LLM call.

          C. Bullets are verbose (avg words per bullet > threshold)
             → Trim each bullet to its first sentence. No LLM call.

          D. Mixed: both too many AND verbose
             → Trim count first, then trim verbose bullets.
        """
        bullet_count = len(bullets)
        avg_words = sum(len(b.split()) for b in bullets) / max(bullet_count, 1)

        too_many  = bullet_count > self.max_bullets
        verbose   = avg_words > self.verbose_bullet_threshold

        # ── Case A: already perfect — return unchanged ────────
        if not too_many and not verbose:
            logger.debug(
                "[Summarizer] Path 3A: %d bullets, avg %.1f words — returned as-is",
                bullet_count, avg_words
            )
            return original.strip()

        # ── Case B / C / D: needs light reformatting ─────────
        logger.debug(
            "[Summarizer] Path 3B/C/D: %d bullets, avg %.1f words — trimming",
            bullet_count, avg_words
        )

        cleaned = []
        for b in bullets:
            # Trim verbose bullet to its first sentence
            if verbose:
                b = self._first_sentence(b)
            # Normalise bullet character
            b = self._BULLET_RE.sub("", b).strip()
            b = self._NUMBERED_RE.sub("", b).strip()
            if b:
                cleaned.append(f"• {b}")

        # Trim count
        cleaned = cleaned[: self.max_bullets]

        return "\n".join(cleaned)

    # ══════════════════════════════════════════════════════════
    # LLM Summarization (Path 6 only)
    # ══════════════════════════════════════════════════════════

    def _llm_summarize(self, answer: str) -> Optional[str]:
        """
        Calls the LLM to summarize a long unstructured prose answer.
        Only reached for answers > long_answer_word_threshold words.
        """
        prompt = SUMMARIZE_PROMPT.format(answer=answer[:2000])
        raw = self.llm._generate(prompt)

        if not raw:
            return None

        if self.debug:
            logger.debug("[Summarizer] LLM raw output:\n%s", raw[:400])

        # Prepend the "•" that was the last character of the prompt
        raw = raw.strip()
        if not raw.startswith("•"):
            raw = "• " + raw

        bullets = self._extract_bullets(raw)
        if not bullets:
            return None

        return "\n".join(bullets[: self.max_bullets])

    # ══════════════════════════════════════════════════════════
    # Bullet Extraction
    # ══════════════════════════════════════════════════════════

    def _extract_bullets(self, text: str) -> list[str]:
        """
        Detects and extracts bullet/numbered list items from text.

        Handles:
          • standard bullet (•, -, *, unicode variants)
          • numbered lists (1. 2) 3 )
          • inline bullets on a single line separated by "•"

        Returns empty list if no structured list is found — this is
        the key signal used to decide whether an answer is "already
        formatted" or "raw prose".
        """
        if not text:
            return []

        # ── Inline bullets on a single line: "• point1 • point2" ──
        if "\n" not in text and text.count("•") >= 2:
            parts = text.split("•")
            bullets = []
            for p in parts:
                p = p.strip()
                if len(p) > 8:
                    bullets.append(f"• {p}")
            return bullets

        lines = text.splitlines()
        bullets = []

        for line in lines:
            line = line.strip()
            if not line or len(line) < 8:
                continue
            if self._BULLET_RE.match(line):
                clean = self._BULLET_RE.sub("", line).strip()
                if clean:
                    bullets.append(f"• {clean}")
            elif self._NUMBERED_RE.match(line):
                clean = self._NUMBERED_RE.sub("", line).strip()
                if clean:
                    bullets.append(f"• {clean}")

        return bullets

    # ══════════════════════════════════════════════════════════
    # Sentence-Split (Paths 4, 5, 6-fallback)
    # ══════════════════════════════════════════════════════════

    def _sentence_split(self, answer: str) -> str:
        """
        Splits prose into sentence-level bullets without any LLM call.

        Rules:
          - Splits on sentence boundaries (.  !  ?)
          - Skips fragments shorter than 12 characters
          - Merges very short sentences (< 6 words) with the next sentence
            to avoid orphan micro-bullets
          - Caps at max_bullets
        """
        sentences = self._SENTENCE_SPLIT_RE.split(answer.strip())

        merged: list[str] = []
        buffer = ""

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if buffer:
                # Merge short buffers with the next sentence
                buffer = buffer + " " + s
                if len(buffer.split()) >= 6:
                    merged.append(buffer.strip())
                    buffer = ""
            elif len(s.split()) < 4:
                # Very short fragment — buffer it
                buffer = s
            else:
                merged.append(s)

        # Flush any remaining buffer
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)

        # Build bullets, skip tiny fragments
        bullets = [f"• {s}" for s in merged if len(s) >= 12]

        if not bullets:
            return f"• {answer}"

        return "\n".join(bullets[: self.max_bullets])

    # ══════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════

    def _first_sentence(self, text: str) -> str:
        """
        Returns the first sentence of text.
        Used to trim verbose bullet points.
        """
        text = self._BULLET_RE.sub("", text).strip()
        text = self._NUMBERED_RE.sub("", text).strip()
        match = self._SENTENCE_SPLIT_RE.search(text)
        if match:
            return text[: match.start()].strip()
        return text.strip()