"""Text tools: question_parser, anchor_extractor, answer_formatter, text utilities."""

from __future__ import annotations

import re


# ── Question parser ──────────────────────────────────────────────────

def parse_question(question: str) -> dict:
    """Parse a question to determine language, answer kind, and format hints.

    Returns dict with keys: lang, answer_kind, format_hint
    """
    # Language detection: if >30% Chinese characters → zh
    total = len(question)
    if total == 0:
        return {"lang": "en", "answer_kind": "entity", "format_hint": ""}

    chinese_chars = sum(1 for c in question if '\u4e00' <= c <= '\u9fff')
    lang = "zh" if chinese_chars / total > 0.3 else "en"

    # Answer kind heuristic
    number_patterns = [
        r"多少", r"几个", r"几年", r"几月", r"第几", r"数量", r"数目",
        r"how many", r"how much", r"what year", r"what number",
        r"how old", r"how long", r"how far", r"how tall",
        r"what percentage", r"what is the number",
    ]
    answer_kind = "entity"
    for pat in number_patterns:
        if re.search(pat, question, re.IGNORECASE):
            answer_kind = "number"
            break

    # Format hint
    format_hint = ""
    if re.search(r"全称|full name|official name", question, re.IGNORECASE):
        format_hint = "use_full_name"
    elif re.search(r"简称|abbreviation|acronym", question, re.IGNORECASE):
        format_hint = "use_abbreviation"

    # Format example: extract explicit format patterns from the question
    format_example = ""
    # Chinese: "要求格式形如：XXX" / "格式如：XXX" / "回答格式形如：XXX"
    fmt_m = re.search(
        r"(?:要求|答案)?(?:回答)?格式(?:形如|如|为|：|:)\s*[:：]?\s*(.+?)(?:[。\.\n]|$)",
        question,
    )
    if fmt_m:
        format_example = fmt_m.group(1).strip().rstrip("。.")
    else:
        # English: "(e.g., XXX)" / "for example, XXX)" / "such as XXX)"
        fmt_m = re.search(
            r"(?:\(?\s*(?:e\.g\.|for example|such as)[,:]?\s*)([^)]+)\)?",
            question, re.IGNORECASE,
        )
        if fmt_m:
            format_example = fmt_m.group(1).strip().rstrip(".),")

    return {"lang": lang, "answer_kind": answer_kind, "format_hint": format_hint,
            "format_example": format_example}


# ── Answer formatter ─────────────────────────────────────────────────

_PREFIX_PATTERNS = [
    r'^答案[是为：:]\s*',
    r'^the answer is\s*[：:]?\s*',
    r'^answer\s*[：:]?\s*',
    r'^最终答案[是为：:]\s*',
    r'^final answer\s*[：:]?\s*',
]

_PREFIX_RE = re.compile('|'.join(_PREFIX_PATTERNS), re.IGNORECASE)


def format_answer(raw: str, answer_kind: str = "entity") -> str:
    """Clean LLM output artifacts from the raw answer.

    - Single line string
    - Remove common prefixes ("答案是", "The answer is", etc.)
    - Remove surrounding quotes
    - Remove Chinese period (。)
    Note: number integerization is left to the evaluation system.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Take only first line
    text = text.split('\n')[0].strip()

    # Remove common prefixes
    text = _PREFIX_RE.sub('', text).strip()

    # Remove surrounding quotes (various kinds)
    for q_open, q_close in [('"', '"'), ('"', '"'), ("'", "'"),
                            ('「', '」'), ('《', '》'), ('"', '"')]:
        if text.startswith(q_open) and text.endswith(q_close) and len(text) > 2:
            text = text[len(q_open):-len(q_close)].strip()

    # Remove trailing Chinese period
    text = text.rstrip('。')

    # Remove trailing parenthetical annotations (e.g., "玉米 (Corn/Maize)" → "玉米")
    text = re.sub(r'\s*[（(][^)）]*[)）]\s*$', '', text).strip()

    # De-duplicate "Operation X and Operation Y" → "Operation X and Y"
    text = re.sub(r'\b(Operation)\s+(.+?)\s+and\s+\1\s+', r'\1 \2 and ', text)

    return text


# ── Normalization for evaluation ─────────────────────────────────────

def normalize_for_eval(s: str, answer_kind: str = "entity") -> str:
    """Normalize answer for evaluation comparison."""
    text = s.strip()
    if answer_kind == "number" and re.fullmatch(r'\d+(\.\d+)?', text):
        return str(int(float(text)))
    return text


# ── Text utilities ───────────────────────────────────────────────────

def clean_snippet(text: str) -> str:
    """Clean a search result snippet."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    # Remove protocol
    domain = re.sub(r'^https?://', '', url)
    # Remove path
    domain = domain.split('/')[0]
    # Remove port
    domain = domain.split(':')[0]
    return domain
