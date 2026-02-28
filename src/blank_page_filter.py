"""
Project Akshara: Blank Page / LLM Commentary Filter
====================================================

Detects and filters out LLM-generated commentary that appears when
models encounter blank, empty, or non-textual pages. Without this filter,
lines like "There is no text on this page" leak into the final output.

Usage:
    from src.blank_page_filter import is_llm_commentary, filter_blank_page_text

    if is_llm_commentary(extracted_text):
        # Page is blank or LLM just described what it saw
        text = ""
    else:
        text = filter_blank_page_text(extracted_text)
"""

import re
from typing import List, Tuple

# Compiled regex patterns for LLM commentary detection.
# Each pattern targets a known class of LLM response to blank/non-text pages.
_COMMENTARY_PATTERNS: List[re.Pattern] = [
    # Direct "no text" / "blank page" statements
    re.compile(r"(?i)there\s+is\s+no\s+(?:visible\s+)?text\s+(?:on|in)\s+this\s+(?:page|image)"),
    re.compile(r"(?i)this\s+(?:page|image)\s+(?:is|appears?\s+to\s+be)\s+blank"),
    re.compile(r"(?i)this\s+(?:page|image)\s+(?:does\s+not|doesn'?t)\s+contain\s+(?:any\s+)?text"),
    re.compile(r"(?i)(?:the\s+)?page\s+(?:is|appears?)\s+(?:to\s+be\s+)?(?:entirely\s+)?(?:blank|empty)"),
    re.compile(r"(?i)no\s+(?:visible\s+|readable\s+)?text\s+(?:is\s+)?(?:present|visible|found|detected)"),
    re.compile(r"(?i)the\s+image\s+(?:shows?|contains?|displays?)\s+(?:a\s+)?blank\s+page"),

    # "I cannot" / "I can see" meta-commentary
    re.compile(r"(?i)i\s+(?:cannot|can'?t|am\s+unable\s+to)\s+(?:extract|transcribe|read|find|identify)\s+(?:any\s+)?text"),
    re.compile(r"(?i)i\s+(?:can\s+)?see\s+(?:that\s+)?(?:this|the)\s+(?:page|image)\s+(?:is|appears?)"),
    re.compile(r"(?i)i\s+(?:don'?t|do\s+not)\s+see\s+any\s+(?:readable\s+)?text"),

    # Description of non-text content
    re.compile(r"(?i)this\s+(?:page|image)\s+(?:contains?|shows?|displays?)\s+(?:only\s+)?(?:an?\s+)?(?:illustration|photograph|image|figure|plate|diagram|map|frontispiece)"),
    re.compile(r"(?i)(?:the\s+)?(?:page|image)\s+(?:contains?|shows?)\s+(?:a\s+)?(?:decorative|ornamental)\s+(?:border|element|design)"),

    # Library / stamp descriptions
    re.compile(r"(?i)(?:this\s+page\s+)?(?:contains?|shows?)\s+(?:only\s+)?(?:a\s+)?library\s+(?:stamp|card|marking|label)"),

    # Apologies and hedging
    re.compile(r"(?i)(?:i\s+)?apologi[sz]e,?\s+(?:but\s+)?(?:there|this|i)"),
    re.compile(r"(?i)unfortunately,?\s+(?:there\s+is\s+no|this\s+page|i\s+cannot)"),

    # "[This page intentionally left blank]" style markers
    re.compile(r"(?i)\[?\s*this\s+page\s+(?:is\s+)?(?:intentionally\s+)?(?:left\s+)?blank\s*\]?"),
]

# Short-text patterns: only checked if the entire text is very short (<150 chars)
_SHORT_TEXT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)^(?:blank\s+page|empty\s+page|no\s+text)\.?\s*$"),
    re.compile(r"(?i)^\[?\s*(?:blank|empty)\s*\]?\s*$"),
    re.compile(r"(?i)^(?:page\s+\d+\s+)?(?:is\s+)?blank\.?\s*$"),
    re.compile(r"(?i)^n/?a\.?\s*$"),
]


def is_llm_commentary(text: str) -> bool:
    """
    Check if the given text is LLM-generated commentary rather than
    actual page content.

    Returns True if the text matches known LLM commentary patterns,
    indicating the page is blank or non-textual.

    Args:
        text: Extracted text from a page

    Returns:
        True if the text is LLM commentary (page should be treated as blank)
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Very short text — check short-text patterns
    if len(stripped) < 150:
        for pattern in _SHORT_TEXT_PATTERNS:
            if pattern.search(stripped):
                return True

    # Check all main commentary patterns
    for pattern in _COMMENTARY_PATTERNS:
        if pattern.search(stripped):
            # Additional heuristic: if the text is short AND matches a pattern,
            # it's almost certainly commentary. If it's long, the pattern match
            # might be embedded in real content — only flag if the text is
            # mostly this commentary (< 300 chars total).
            if len(stripped) < 300:
                return True
            # For longer text, only flag if the commentary appears in the
            # first 200 characters (i.e., the LLM led with commentary)
            match = pattern.search(stripped)
            if match and match.start() < 200:
                # Check if the rest is also short / empty
                remaining = stripped[match.end():].strip()
                if len(remaining) < 100:
                    return True

    return False


def filter_blank_page_text(text: str) -> str:
    """
    Remove LLM commentary lines from text while preserving real content.

    Unlike is_llm_commentary() which checks the whole text, this function
    operates line-by-line to strip individual commentary lines from
    otherwise valid text.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text with commentary lines removed
    """
    if not text or not text.strip():
        return ""

    if is_llm_commentary(text):
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            cleaned_lines.append(line)
            continue

        is_commentary = False
        for pattern in _COMMENTARY_PATTERNS:
            if pattern.search(stripped_line):
                is_commentary = True
                break

        if not is_commentary and len(stripped_line) < 150:
            for pattern in _SHORT_TEXT_PATTERNS:
                if pattern.search(stripped_line):
                    is_commentary = True
                    break

        if not is_commentary:
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines).strip()
    return result


def classify_page_content(text: str) -> Tuple[str, str]:
    """
    Classify a page's content and return (classification, cleaned_text).

    Classifications:
        "BLANK"       — Empty or pure LLM commentary
        "CONTENT"     — Real textual content
        "MIXED"       — Some content with some commentary removed

    Args:
        text: Raw extracted text

    Returns:
        Tuple of (classification, cleaned_text)
    """
    if not text or not text.strip():
        return ("BLANK", "")

    if is_llm_commentary(text):
        return ("BLANK", "")

    cleaned = filter_blank_page_text(text)
    if not cleaned:
        return ("BLANK", "")

    if len(cleaned) < len(text.strip()) * 0.8:
        return ("MIXED", cleaned)

    return ("CONTENT", cleaned)
