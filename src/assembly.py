"""
Project Akshara: Pass 2/3 - Text Assembly
==========================================

Assembles extracted pages into chapters with:
- Chapter boundary detection (Bug Fix #3: stricter matching)
- Page type classification (Bug Fix #1: LLM commentary → BLANK)
- Footnote extraction and placement (Bug Fix #2: multi-line support)
- Cross-page word joining
- QA report generation

Usage:
    assembler = BookAssembler(gemini_client, claude_client)
    book = assembler.assemble(extractions)
"""

import re
import json
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field

from src.blank_page_filter import is_llm_commentary, filter_blank_page_text
from src.extraction import PageExtraction

# Model constant — used in logging and QA report
MODEL = "gemini-3-flash-preview"

SUPERSCRIPT_DIGIT_MAP = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
})
SUPERSCRIPT_DIGIT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
SUPERSCRIPT_LETTER_TO_ASCII = {
    "ᵃ": "a",
    "ᵇ": "b",
    "ᶜ": "c",
    "ᵈ": "d",
    "ᵉ": "e",
    "ᶠ": "f",
    "ᵍ": "g",
    "ʰ": "h",
    "ⁱ": "i",
    "ʲ": "j",
    "ᵏ": "k",
    "ˡ": "l",
    "ᵐ": "m",
    "ⁿ": "n",
    "ᵒ": "o",
    "ᵖ": "p",
    "ʳ": "r",
    "ˢ": "s",
    "ᵗ": "t",
    "ᵘ": "u",
    "ᵛ": "v",
    "ʷ": "w",
    "ˣ": "x",
    "ʸ": "y",
    "ᶻ": "z",
}
SUPERSCRIPT_LETTER_CHARS = "".join(SUPERSCRIPT_LETTER_TO_ASCII.keys())

# ─── Chapter Detection ───────────────────────────────────────────────────────
# Bug Fix #3: Only strong keywords trigger chapter detection.
# Previously, ANY short ALL CAPS line was treated as a chapter heading.

CHAPTER_KEYWORDS = re.compile(
    r'(?i)^(?:chapter|part|book|section|volume|preface|foreword|'
    r'introduction|prologue|epilogue|appendix|afterword|glossary|'
    r'bibliography|index|dedication|acknowledgement|acknowledgment|'
    r'contents|table\s+of\s+contents)\b'
)

# Roman numeral pattern (I through XXXIX covers most books)
ROMAN_NUMERAL = re.compile(
    r'^(?:X{0,3})(?:IX|IV|V?I{0,3})$'
)

LIBRARY_ARTIFACT_PATTERN = re.compile(
    r'(?i)\b('
    r'rmic\s+library|'
    r'ramakrishna\s+mission\s+institute\s+of\s+culture|'
    r'loan\s+dept\.?|'
    r'general\s+library|'
    r'univ\.?\s+of\s+california|'
    r'university\s+of\s+california|'
    r'berkeley\s+libraries|'
    r'return\s+to\s+desk|'
    r'due\s+on\s+the\s+last\s+date\s+stamped|'
    r'\bacc(?:ession)?\.?\s*no\.?|'
    r'\bclass\s*no\.?|'
    r'\bst\.\s*card\b|'
    r'\bbk\.\s*card\b|'
    r'\bchecked\b|'
    r'\blibrary\s*stamp\b'
    r')\b'
)

READER_FRONTMATTER_SCAN_ARTIFACTS = [
    re.compile(r'(?i)^\s*uc[- ]?nrlf\s*$'),
    re.compile(r'(?i)^\s*[ab]\s*\d[\d\s]{3,}\s*$'),
    re.compile(r'(?i)^\s*(?:repl|refel|reel)\s+\d+\s*$'),
    re.compile(r'(?i)^\s*reproduced\s+by\s+duopage\s+process\b'),
    re.compile(r'(?i)^\s*micro\s+photo\s+inc\.?\s*$'),
    re.compile(r'(?i)^\s*cleveland\s+\d+,\s*ohio\.?\s*$'),
    re.compile(r'(?i)^\s*in\s+the\s+united\s+states\s+of\s+america\s*$'),
    re.compile(r'(?i)^\s*[a-z]-\d{5,}\s*$'),
    re.compile(r'(?i)^\s*ds\s*\.?\s*\d{1,4}\s*[a-z]?\d*\s*$'),
    re.compile(r'(?i)^\s*r\d{1,4}\s*$'),
    # Scan/binder signature code (e.g. "5-VII-S")
    re.compile(r'(?i)^\s*\d{1,3}-[ivxlcdm]{1,8}-[a-z]\s*$'),
    # Isolated short folio/signature markers often found on title/imprint pages.
    re.compile(r'^\s*\d{1,3}\s*$'),
    re.compile(r'(?i)^\s*[ivxlcdm]{1,4}\s*$'),
]

TOC_JSON_TABLE_LINE = re.compile(r'^\s*\{"table"\s*:\s*\[.*\]\}\s*$')
TOC_ILLUSTRATION_ROW = re.compile(
    r'^\s*(\d+)\.\s+(.*?)\s+(\d+|Frontis(?:-\s*|\s*)piece)\s*$',
    flags=re.IGNORECASE,
)

FRONTMATTER_SECTION_HEADING_KEYS = {
    "an introductory note",
    "contents",
    "illustrations",
    "list of illustrations",
    "list of plates",
    "list of maps",
    "list of paintings",
    "list of authorities",
    "list of authorities consulted",
}

# LLM-assisted assembly prompts (operations only; no freeform rewriting)
MARGIN_OPS_PROMPT = """You are an archival text assembler.
Task: identify ONLY non-content edge lines (running headers, running footers, standalone page numbers).

STRICT RULES:
1) You must NOT remove body content sentences.
2) You may only select from the candidate lines provided.
3) If uncertain, return no removals.

Return JSON only:
{{
  "remove_candidate_ids": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Candidates:
{candidates}
"""

BOUNDARY_OPS_PROMPT = """You are an archival page-boundary analyzer.
Decide if page {prev_page} and page {next_page} should be joined at the boundary.

Rules:
1) Join when a word/sentence clearly continues across page break.
2) Do NOT join if a new chapter/section starts.
3) If joining, choose the correct join_mode for exact text preservation:
   - "keep_hyphen"    => keep visible hyphen across page break (e.g. Durga-Navaratri)
   - "drop_hyphen"    => remove line-break hyphenation (e.g. archaeo- / logical)
   - "space"          => insert a space between pages
   - "nospace"        => no space (rare; only if token clearly continues)
   - "auto"           => conservative fallback
4) If uncertain, choose false.

Return JSON only:
{{
  "should_join": true/false,
  "join_mode": "auto|keep_hyphen|drop_hyphen|space|nospace",
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

END OF PAGE {prev_page}:
{prev_tail}

START OF PAGE {next_page}:
{next_head}
"""

HEADING_PLAN_PROMPT = """You are an archival reader-edition assembly planner.
Task: assign markdown heading levels and emission behavior to already-detected section titles.

STRICT RULES:
1) Do NOT rewrite, normalize, or rename titles.
2) Only return planning metadata for the provided chapter indexes.
3) Preserve structure: BOOK/PART wrappers may be emitted even if body is empty.
4) If uncertain, choose conservative defaults (level 2, emit_if_empty=true for BOOK/PART; else false).

Return JSON only:
{{
  "chapters": [
    {{
      "chapter_index": int,
      "heading_level": 2|3|4,
      "emit_if_empty": true|false,
      "kind": "frontmatter|book|part|chapter|index|appendix|preface|introduction|other",
      "confidence": 0.0-1.0
    }}
  ],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Chapters:
{chapters}
"""

TOP_STACK_OPS_PROMPT = """You are an archival reader-edition page cleanup planner.
Task: identify only top-of-page non-content lines (running headers, library stamps, page labels) from the provided candidates.

STRICT RULES:
1) Never remove body prose.
2) Only select from the candidate IDs provided.
3) Do NOT remove chapter opener headings on true chapter-start pages.
4) If uncertain, return no removals.

Return JSON only:
{{
  "remove_candidate_ids": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- page_number: {page_number}
- page_type: {page_type}

Top candidates:
{candidates}
"""

FRONTMATTER_STRUCTURE_OPS_PROMPT = """You are an archival reader-edition frontmatter structuring planner.
Task: propose ONLY line-level structural operations for a merged frontmatter section.

STRICT RULES:
1) Do NOT rewrite any text.
2) Do NOT paraphrase, correct spelling, or change punctuation.
3) You may only reference the provided line IDs.
4) Remove only clear running headers/page labels/library artifacts (not prose).
5) Promote only true section headings (e.g. CONTENTS, ILLUSTRATIONS, AN INTRODUCTORY NOTE, LIST OF AUTHORITIES).
6) If uncertain, do nothing.

Return JSON only:
{{
  "remove_line_ids": [int, ...],
  "promote_heading_line_ids": [int, ...],
  "authorities_table_heading_line_ids": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}
- repeated_short_lines: {repeated_short_lines}

Lines:
{lines}
"""

CHAPTER_LINE_ROLE_OPS_PROMPT = """You are an archival reader-edition chapter cleanup planner.
Task: classify ONLY provided line IDs for structural cleanup in an already assembled chapter body.

STRICT RULES:
1) Do NOT rewrite any text.
2) Do NOT paraphrase, correct spelling, or change punctuation.
3) Only reference the provided line IDs.
4) Remove only clear running headers / library artifacts / page labels.
5) Promote only true section headings already present in the text.
6) If uncertain, do nothing.

Return JSON only:
{{
  "remove_line_ids": [int, ...],
  "promote_heading_line_ids": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}
- chapter_kind: {chapter_kind}

Candidate lines:
{lines}
"""

BACKMATTER_STRUCTURE_OPS_PROMPT = """You are an archival reader-edition backmatter formatting planner.
Task: classify ONLY provided line IDs for structural cleanup in already assembled backmatter text (index, bibliography, appendix, references).

STRICT RULES:
1) Do NOT rewrite any text.
2) Do NOT paraphrase, correct spelling, or change punctuation.
3) Only reference the provided line IDs.
4) Remove only clear running headers / page labels / printer colophon / library artifacts.
5) Promote only true section headings already present (e.g. INDEX, BIBLIOGRAPHY, APPENDIX, REFERENCES).
6) Promote letter dividers only when they are true index alphabetical dividers (A, B, C, ...).
7) If uncertain, do nothing.

Return JSON only:
{{
  "remove_line_ids": [int, ...],
  "promote_heading_line_ids": [int, ...],
  "promote_letter_divider_line_ids": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}
- backmatter_kind: {backmatter_kind}

Candidate lines:
{lines}
"""

FOOTNOTE_LINK_OPS_PROMPT = """You are an archival reader-edition footnote linking planner.
Task: match ambiguous footnote references in chapter prose to the correct chapter-end footnote definitions.

STRICT RULES:
1) Do NOT rewrite any text.
2) Do NOT invent footnotes or delete footnotes.
3) Only reassign the provided reference occurrence numbers to one of the provided definition IDs.
4) Only map within the same footnote base ID family (e.g. 2 -> 2-p56, 2-p57).
5) If uncertain, return no reassignments.

Return JSON only:
{{
  "reassign_ref_occurrences": [
    {{
      "occurrence": int,
      "target_def_id": "string"
    }}
  ],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}

Ambiguous definitions (verbatim starts only):
{definitions}

Ambiguous reference occurrences:
{references}
"""

FOOTNOTE_MARKER_INSERT_OPS_PROMPT = """You are an archival reader-edition footnote marker insertion planner.
Task: identify OCR-surviving inline single-letter note markers in chapter prose (e.g. `b Beekapore`)
that should become markdown footnote references (e.g. `[^b] Beekapore`).

STRICT RULES:
1) Do NOT rewrite prose.
2) Only select from the provided candidate occurrence numbers.
3) Candidates already include the detected marker letter; you must not change it.
4) Only choose occurrences that are clearly note markers, not ordinary articles/letters.
5) If uncertain, return no insertions.

Return JSON only:
{{
  "insert_marker_occurrences": [int, ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}
- missing_marker_letters: {missing_letters}

Unreferenced footnote definitions (preview):
{definitions}

Candidate marker occurrences:
{candidates}
"""

FOOTNOTE_RESIDUAL_OPS_PROMPT = """You are an archival reader-edition residual footnote resolver.
Task: resolve remaining unreferenced chapter-end footnote definitions by selecting safe inline insertion points,
and optionally dropping only clearly spurious residual definitions.

STRICT RULES:
1) Do NOT rewrite prose.
2) Do NOT invent new insertion locations; only use provided occurrence numbers.
3) Each occurrence already targets a specific definition ID; do not retarget it.
4) Drop only when a definition is clearly non-content/noise and should not appear in reader output.
5) If uncertain, return no insertions and no drops.

Return JSON only:
{{
  "insert_ref_occurrences": [
    {{
      "occurrence": int,
      "target_def_id": "string"
    }}
  ],
  "drop_def_ids": ["string", ...],
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Context:
- chapter_title: {chapter_title}

Unreferenced definitions:
{definitions}

Candidate insertion occurrences:
{candidates}

Drop candidates (subset of unresolved defs):
{drop_candidates}
"""


@dataclass
class Chapter:
    """A detected chapter."""
    number: int
    title: str
    pages: List[int]  # Page numbers in this chapter
    body: str
    footnotes: List[Dict[str, str]]  # [{"id": "1", "text": "..."}, ...]
    word_count: int = 0
    has_indic_script: bool = False
    has_poetry: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.body.strip()


def _chapter_title_kind(title: str) -> str:
    s = (title or "").strip()
    if not s:
        return "other"
    lower = s.lower()
    if lower == "front matter":
        return "frontmatter"
    if re.match(r'(?i)^book\b', s):
        return "book"
    if re.match(r'(?i)^part\b', s):
        return "part"
    if re.match(r'(?i)^chapter\b', s):
        return "chapter"
    if re.match(r'(?i)^index\b', s):
        return "index"
    if re.match(r'(?i)^appendix\b', s):
        return "appendix"
    if re.match(r'(?i)^preface\b', s):
        return "preface"
    if re.match(r'(?i)^introduction\b', s):
        return "introduction"
    return "other"


def _should_emit_structural_empty_chapter(title: str) -> bool:
    return _chapter_title_kind(title) in {"book", "part", "frontmatter", "index", "appendix", "preface", "introduction"}


def _default_heading_level_for_title(title: str) -> int:
    kind = _chapter_title_kind(title)
    if kind in {"frontmatter", "preface", "introduction", "index", "appendix"}:
        return 2
    if kind == "book":
        return 2
    if kind == "part":
        return 3
    if kind == "chapter":
        return 3
    return 2


@dataclass
class AssembledBook:
    """Fully assembled book."""
    title: str
    author: str
    chapters: List[Chapter]
    total_pages: int
    total_words: int
    blank_pages: List[int]
    warnings: List[str]
    pages_needing_review: List[int]
    master_markdown: str  # The final assembled text
    markdown: str  # Alias for export compatibility
    avg_fidelity: float = 0.0
    assembly_ops: List[Dict[str, Any]] = field(default_factory=list)


# ─── Page Type Detection ─────────────────────────────────────────────────────

def detect_page_type(extraction: PageExtraction, strict_chapter_detection: bool = True) -> str:
    """
    Classify a page's type based on its content and metadata.

    Bug Fix #1: Treats LLM commentary as BLANK.
    Bug Fix #3: Stricter chapter heading detection.

    Returns one of: "blank", "chapter_start", "frontmatter", "backmatter",
                     "illustration", "toc", "text"
    """
    text = extraction.text.strip()

    # Check for LLM commentary (Bug Fix #1)
    if not text or is_llm_commentary(text):
        return "blank"

    # If extraction already classified as blank
    if extraction.page_type == "blank":
        return "blank"

    # Illustration pages
    if extraction.page_type == "illustration" or (
        text.startswith("[Illustration") and len(text) < 200
    ):
        return "illustration"

    # TOC detection
    if extraction.page_type == "toc":
        return "toc"

    # Chapter start detection (Bug Fix #3: strict keyword matching)
    if _is_chapter_start(text, extraction, strict_chapter_detection=strict_chapter_detection):
        return "chapter_start"

    # Front matter (title pages, copyright, etc.)
    if extraction.page_type in ("title", "frontmatter"):
        return "frontmatter"

    if extraction.page_type == "backmatter":
        return "backmatter"

    return "text"


def _is_chapter_start(
    text: str,
    extraction: PageExtraction,
    strict_chapter_detection: bool = True,
) -> bool:
    """
    Determine if a page is the start of a new chapter.

    Bug Fix #3: Previously any ALL CAPS short line triggered this.
    Now requires chapter keywords for strong match.
    """
    lines = text.split('\n')
    non_empty = [line.strip() for line in lines if line.strip()]
    first_nonempty = non_empty[0] if non_empty else ""
    second_nonempty = non_empty[1] if len(non_empty) > 1 else ""

    # Running headers are often short ALL CAPS lines followed by a lowercase continuation.
    if _looks_like_running_header(first_nonempty, second_nonempty):
        return False
    if _looks_like_running_header_block(text):
        return False

    # Optional lenient mode for difficult scans/layouts.
    if not strict_chapter_detection:
        for line in lines[:5]:
            stripped = line.strip()
            if stripped.startswith('#') and stripped.lstrip('#').strip():
                return True
        for line in lines[:3]:
            stripped = line.strip()
            if (stripped and len(stripped) > 2 and len(stripped) < 60
                    and stripped == stripped.upper()
                    and any(c.isalpha() for c in stripped)):
                return True

    # Check for markdown headings with chapter keywords
    for line in lines[:5]:
        stripped = line.strip()
        if stripped.startswith('#'):
            heading_text = stripped.lstrip('#').strip()
            if _looks_like_standalone_chapter_heading_line(heading_text):
                return True

    # Check for chapter keywords in plain text (first 5 lines)
    for line in lines[:5]:
        stripped = line.strip()
        # Suppress running header variants like "INTRODUCTION. 19" on long pages.
        if (
            (_looks_like_running_header_with_page_number(stripped) or _looks_like_page_header_combo(stripped))
            and len(text) > 250
        ):
            continue
        if _looks_like_standalone_chapter_heading_line(stripped):
            return True

    # ALL CAPS check — BUT only if the page is very short (<200 chars)
    # AND contains chapter-like structure
    if len(text) < 200:
        for line in lines[:3]:
            stripped = line.strip()
            if (stripped and len(stripped) > 2 and len(stripped) < 60
                    and stripped == stripped.upper()
                    and any(c.isalpha() for c in stripped)):
                # Only match if it looks like a title (keywords or Roman numeral)
                if _looks_like_standalone_chapter_heading_line(stripped):
                    return True
                # Roman numeral alone on a short page
                clean = stripped.rstrip('.')
                if ROMAN_NUMERAL.match(clean):
                    return True

    # Check extraction metadata, but gate it to avoid turning running headers into chapters.
    if extraction.chapter_heading:
        heading = extraction.chapter_heading.strip()
        if heading:
            # Accept metadata-only heading only when the first visible line matches and
            # page layout looks like a true heading block (blank line after the heading).
            heading_norm = re.sub(r'[^a-z0-9]+', ' ', heading.lower()).strip()
            first_norm = re.sub(r'[^a-z0-9]+', ' ', first_nonempty.lower()).strip()
            heading_matches_first = bool(heading_norm and first_norm and heading_norm == first_norm)
            top_join_norm = _normalize_headingish(" ".join(non_empty[:4]))
            headingish_norm = _normalize_headingish(heading)
            heading_overlaps_top = bool(
                headingish_norm
                and top_join_norm
                and (headingish_norm in top_join_norm or top_join_norm in headingish_norm)
            )
            top_has_ordinal_marker = _is_standalone_ordinal_marker(first_nonempty)
            heading_tokens = heading_norm.split()
            first_tokens = first_norm.split()
            heading_starts_with_ordinal = bool(
                top_has_ordinal_marker
                and heading_tokens
                and first_tokens
                and heading_tokens[0] == first_tokens[0]
            )
            heading_text_for_keyword = heading.strip().lstrip('#').strip()
            heading_has_chapter_keyword = bool(_looks_like_standalone_chapter_heading_line(heading_text_for_keyword))
            short_allcaps_plain = bool(
                heading
                and heading == heading.upper()
                and any(c.isalpha() for c in heading)
                and len(heading.split()) <= 4
                and not heading_has_chapter_keyword
                and not top_has_ordinal_marker
            )
            second_looks_title = bool(
                second_nonempty
                and any(c.isalpha() for c in second_nonempty)
                and len(second_nonempty) <= 220
                and not _looks_like_running_header_with_page_number(second_nonempty)
                and not _looks_like_page_header_combo(second_nonempty)
            )

            blank_after_first = False
            for idx, line in enumerate(lines):
                if line.strip():
                    if idx + 1 < len(lines) and not lines[idx + 1].strip():
                        blank_after_first = True
                    break

            # Allow a real chapter opener hidden under a short artifact/header stack
            # (e.g. "UNIV. OF / CALIFORNIA / INDIAN SHIPPING. / INTRODUCTION. / I.-...").
            heading_line_idx: Optional[int] = None
            for idx, line in enumerate(lines[:20]):
                if _normalize_simple(line) == heading_norm:
                    heading_line_idx = idx
                    break
            if heading_line_idx is not None:
                prior_nonempty = [ln.strip() for ln in lines[:heading_line_idx] if ln.strip()]
                prefix_artifacts = True
                for ln in prior_nonempty:
                    norm_ln = _normalize_simple(ln.rstrip(".:"))
                    alpha = [c for c in ln if c.isalpha()]
                    upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
                    looks_caps_stub = bool(
                        len(ln.split()) <= 4
                        and len(ln) <= 40
                        and upper_ratio >= 0.8
                    )
                    if not (
                        _looks_like_running_header_with_page_number(ln)
                        or _looks_like_page_header_combo(ln)
                        or LIBRARY_ARTIFACT_PATTERN.search(ln)
                        or re.match(r'(?i)^univ\.?\s+of$', ln)
                        or re.match(r'(?i)^california\.?$', ln)
                        or ROMAN_NUMERAL.match(ln.rstrip('.'))
                        or looks_caps_stub
                    ):
                        prefix_artifacts = False
                        break

                blank_after_heading = bool(
                    heading_line_idx + 1 < len(lines) and not lines[heading_line_idx + 1].strip()
                )
                next_after_heading = ""
                for probe in lines[heading_line_idx + 1:heading_line_idx + 8]:
                    if probe.strip():
                        next_after_heading = probe.strip()
                        break
                next_alpha = [c for c in next_after_heading if c.isalpha()]
                next_upper_ratio = (sum(1 for c in next_alpha if c.isupper()) / len(next_alpha)) if next_alpha else 0.0
                next_title_like = bool(
                    next_after_heading
                    and (
                        _looks_like_standalone_chapter_heading_line(next_after_heading)
                        or _is_standalone_ordinal_marker(next_after_heading)
                        or (
                            len(next_after_heading) <= 180
                            and len(next_after_heading.split()) <= 18
                            and next_upper_ratio >= 0.55
                            and not re.match(r'^[a-z]', next_after_heading)
                        )
                    )
                )

                if (
                    prior_nonempty
                    and len(prior_nonempty) <= 5
                    and prefix_artifacts
                    and blank_after_heading
                    and (
                        heading_has_chapter_keyword
                        or next_title_like
                    )
                    and not _looks_like_running_header_with_page_number(heading)
                    and not _looks_like_page_header_combo(heading)
                ):
                    return True

            if (
                heading_matches_first
                and blank_after_first
                and not short_allcaps_plain
                and not _looks_like_running_header(first_nonempty, second_nonempty)
                and not _looks_like_running_header_block(text)
                and not _looks_like_running_header_with_page_number(first_nonempty)
            ):
                return True

            # Common chapter opening layout in older books:
            #   I.
            #   Long chapter title...
            # Use extraction metadata to validate the title block and avoid false splits
            # on running headers like "*THE SHEPHERD.* 61".
            if (
                top_has_ordinal_marker
                and second_looks_title
                and (heading_overlaps_top or heading_starts_with_ordinal)
                and len(text) > 120
                and not _looks_like_running_header(first_nonempty, second_nonempty)
                and not _looks_like_running_header_block(text)
                and not _looks_like_running_header_with_page_number(first_nonempty)
                and not _looks_like_page_header_combo(first_nonempty)
            ):
                return True

    return False


def _looks_like_running_header(first_line: str, second_line: str) -> bool:
    """
    Heuristic: short all-caps-ish top line followed by lowercase continuation
    is usually a running header, not a chapter start.
    """
    first = first_line.strip()
    second = second_line.strip()
    if not first or not second:
        return False
    if len(first) > 80:
        return False
    if len(first.split()) > 12:
        return False
    if CHAPTER_KEYWORDS.match(first):
        # e.g. "INTRODUCTION. 19" repeated on continued pages
        if not re.search(r'\b\d{1,4}\b\s*\.?$', first):
            return False
    if ROMAN_NUMERAL.match(first.rstrip('.')):
        return False

    alpha_chars = [c for c in first if c.isalpha()]
    if not alpha_chars:
        return False
    uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    if uppercase_ratio < 0.75:
        return False

    return bool(re.match(r'^[a-z0-9\"\'(\[]', second))


def _looks_like_running_header_with_page_number(line: str) -> bool:
    """Detect header-like line carrying a chapter label + page number suffix."""
    s = line.strip().strip('*').strip()
    if not s:
        return False
    # Bibliographic citations can look like "Part i., No. 1, pp. 64-96."
    if re.search(r'(?i)\b(?:vol|no|pp?)\.\s*\d', s):
        return False
    if not CHAPTER_KEYWORDS.match(s):
        return False
    # Exclude legitimate standalone chapter labels like "CHAPTER I." or "PART II"
    if re.match(
        r'(?i)^(?:chapter|part|book|section|volume)\s+'
        r'(?:\d+|[ivxlcdm]+)\.?\s*$',
        s,
    ):
        return False
    if re.search(r'\b\d{1,4}\b\s*\.?$', s):
        return True
    return bool(re.search(r'\b[ivxlcdm]{1,8}\b\s*\.?$', s, flags=re.IGNORECASE))


def _looks_like_page_header_combo(line: str) -> bool:
    """
    Detect one-line running-header/page-number combos, including prefixed numbers:
    e.g., "10 INTRODUCTION.", "*INTRODUCTION.* 15", "INTRODUCTION. 21 ."
    """
    s = re.sub(r'\s+', ' ', line.strip())
    if not s:
        return False
    plain = s.replace('*', '').strip()
    if re.match(
        r'(?i)^(?:chapter|part|book|section|volume)\s+'
        r'(?:\d+|[ivxlcdm]+)\.?\s*$',
        plain,
    ):
        return False
    if not (
        re.search(r'\b\d{1,4}\b', plain)
        or re.search(r'\b[ivxlcdm]{1,8}\b', plain, flags=re.IGNORECASE)
    ):
        return False
    # Allow an optional leading page number before the chapter-like label.
    # Header keywords must appear at the start (optionally after a leading page number),
    # otherwise prose like "I have indicated... Introduction..." gets misclassified.
    if not re.match(
        r'(?i)^(?:\d{1,4}\s+)?(?:chapter|part|book|section|volume|preface|foreword|'
        r'introduction|prologue|epilogue|appendix|afterword|glossary|'
        r'bibliography|index|dedication|acknowledgement|acknowledgment|'
        r'contents|table\s+of\s+contents)\b',
        plain,
    ):
        return False
    if len(plain.split()) > 8:
        return False
    # Bibliographic citations can look like "Part i., No. 1, pp. 64-96."
    # They are not running headers.
    if re.search(r'(?i)\b(?:vol|no|pp?)\.\s*\d', plain):
        return False
    if re.search(r'\b(?:contains|says|treats|describes|devoted)\b', plain, flags=re.IGNORECASE):
        return False
    return True


def _looks_like_standalone_chapter_heading_line(line: str) -> bool:
    """
    Heading-like chapter line, not prose that merely starts with a chapter keyword.
    """
    s = line.strip().lstrip('#').strip()
    if not s:
        return False
    if not CHAPTER_KEYWORDS.match(s):
        return False
    if re.match(r'(?i)^(?:part|book|section|volume)\b', s):
        if not re.match(
            r'(?i)^(?:part|book|section|volume)\s+(?:\d+|[ivxlcdm]+|[a-z])(?:\b|[.):-])',
            s,
        ):
            return False
    if len(s) > 90:
        return False
    if re.search(r'\b(?:contains|says|treats|describes|devoted|remarks?)\b', s, flags=re.IGNORECASE):
        return False
    return True


def _normalize_headingish(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r'[\u2010-\u2015]', '-', s)
    s = re.sub(r'[^a-z0-9\s-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _is_standalone_ordinal_marker(line: str) -> bool:
    """
    Detect a chapter-opening ordinal marker line such as "I.", "XV.", or "12.".
    """
    s = line.strip().strip('*').strip()
    if not s:
        return False
    s = s.rstrip('.').strip()
    if not s:
        return False
    if s.isdigit():
        return 1 <= int(s) <= 500
    return bool(ROMAN_NUMERAL.match(s.upper()))


def _looks_like_running_header_block(text: str) -> bool:
    """
    Detect multi-line header stacks at the top of a continued page, such as:
    - "20" + "INTRODUCTION."
    - "INTRODUCTION." + "17"
    """
    non_empty = [line.strip() for line in text.split('\n') if line.strip()]
    if len(non_empty) < 2:
        return False

    first = non_empty[0]
    second = non_empty[1]

    # page number only + chapter label on long pages
    if re.match(r'^\d{1,4}\*?\.?$', first):
        if CHAPTER_KEYWORDS.match(second.strip('*').strip()) and len(text) > 250:
            return True

    # chapter label + page number only on long pages
    if CHAPTER_KEYWORDS.match(first.strip('*').strip()):
        if re.match(r'^\d{1,4}\*?\.?$', second) and len(text) > 250:
            return True

    return False


# ─── Footnote Extraction ─────────────────────────────────────────────────────

_PLAIN_LETTER_NOTE_LINE_RE = re.compile(r'^\s*([a-z])\s{1,4}(.*\S)\s*$')


def _looks_like_ordinal_suffix_marker_id(
    marker_id: str,
    *,
    prev_char: str = "",
    next_char: str = "",
) -> bool:
    """
    Guard against OCR/TeX fragments like `3$^d$` / `2$^d$` being normalized into
    false footnote refs (`3[^d]`, `2[^d]`).
    """
    mid = str(marker_id or "").strip().lower()
    if mid not in {"d", "st", "nd", "rd", "th"}:
        return False
    if not (prev_char and prev_char.isdigit()):
        return False
    if next_char and (next_char.isalnum() or next_char in "_"):
        return False
    return True


def _match_plain_letter_note_line(line: str) -> Optional[Tuple[str, str]]:
    s = (line or "").rstrip("\n")
    stripped = s.strip()
    if not stripped:
        return None
    if stripped.startswith(("#", "|", ">", "[^")):
        return None
    if _looks_like_footnote_definition_line(stripped):
        return None
    m = _PLAIN_LETTER_NOTE_LINE_RE.match(s)
    if not m:
        return None
    fid = m.group(1).strip()
    txt = m.group(2).strip()
    if not fid or not txt:
        return None
    # Reject obvious page/header artifacts and accidental prose list items.
    if re.fullmatch(r'[\-—–_*=·•♦◈\s]+', txt):
        return None
    if re.fullmatch(r'[ivxlcdm]{1,8}[.]?$', txt, flags=re.IGNORECASE):
        return None
    return fid, txt


def _parse_plain_letter_footnote_run(
    lines: List[str],
    start_idx: int,
) -> Optional[Tuple[int, List[Dict[str, str]]]]:
    """
    Parse OCR footnote blocks emitted as raw lettered lines:
      a text...
      b text...
      c wrapped
        continuation...

    Conservative acceptance:
    - must contain at least 2 distinct lettered notes
    - must start in the lower half of the page or after a heading/blank separator
    """
    if start_idx < 0 or start_idx >= len(lines):
        return None

    first = _match_plain_letter_note_line(lines[start_idx])
    if not first:
        return None

    total = len(lines)
    # Footnote blocks usually appear in lower page regions unless clearly separated.
    prev_nonempty = ""
    for k in range(start_idx - 1, -1, -1):
        if lines[k].strip():
            prev_nonempty = lines[k].strip()
            break
    lower_halfish = start_idx >= max(6, total // 3)
    separated = (
        start_idx == 0
        or not lines[start_idx - 1].strip()
        or prev_nonempty.startswith("#")
        or bool(re.match(r'(?i)^notes?\.?$', prev_nonempty))
        or bool(re.fullmatch(r'[A-Z][A-Z\s]{0,24}', prev_nonempty))
    )
    if not (lower_halfish or separated):
        return None

    # Require at least one more marker nearby before we consume anything.
    probe_letters: List[str] = []
    probe_steps = 0
    j = start_idx
    while j < len(lines) and probe_steps < 10:
        probe_steps += 1
        mm = _match_plain_letter_note_line(lines[j])
        if mm:
            probe_letters.append(mm[0])
            if len(set(probe_letters)) >= 2:
                break
        j += 1
    if len(set(probe_letters)) < 2:
        return None

    out: List[Dict[str, str]] = []
    current_id: Optional[str] = None
    current_lines: List[str] = []
    idx = start_idx
    distinct_letters: List[str] = []

    def _flush_current() -> None:
        nonlocal current_id, current_lines
        if not current_id:
            return
        txt = "\n".join(ln.strip() for ln in current_lines if ln.strip()).strip()
        if txt:
            out.append({"id": current_id, "text": txt})
        current_id = None
        current_lines = []

    while idx < len(lines):
        raw = lines[idx]
        stripped = raw.strip()
        if not stripped:
            _flush_current()
            idx += 1
            break

        marker = _match_plain_letter_note_line(raw)
        if marker:
            fid, txt = marker
            if current_id is None:
                current_id = fid
                current_lines = [txt]
                distinct_letters.append(fid)
                idx += 1
                continue

            if fid != current_id:
                # Prefer ascending sequences; stop on obvious reset after a valid run.
                if fid in distinct_letters and len(set(distinct_letters)) >= 2:
                    _flush_current()
                    break
                if ord(fid) + 1 < ord(current_id) and len(set(distinct_letters)) >= 2:
                    _flush_current()
                    break
                _flush_current()
                current_id = fid
                current_lines = [txt]
                distinct_letters.append(fid)
                idx += 1
                continue

        # Continuation line for current note.
        if current_id is None:
            break
        if stripped.startswith("#") or _looks_like_running_header_with_page_number(stripped) or _looks_like_page_header_combo(stripped):
            _flush_current()
            break
        current_lines.append(_strip_footnote_leading_wrappers(raw))
        idx += 1

    _flush_current()

    uniq_letters = []
    for fn in out:
        fid = str(fn.get("id", "")).strip()
        if fid and fid not in uniq_letters:
            uniq_letters.append(fid)
    if len(uniq_letters) < 2:
        return None

    # Filter obvious heading contamination inside the run. Keep note-like prose only.
    cleaned: List[Dict[str, str]] = []
    for fn in out:
        fid = str(fn.get("id", "")).strip()
        txt = str(fn.get("text", "")).strip()
        if not fid or not txt:
            continue
        if _looks_like_running_header_with_page_number(txt) or _looks_like_page_header_combo(txt):
            continue
        alpha = [c for c in txt if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        if len(txt.split()) <= 1 and txt.isupper():
            continue
        if alpha and upper_ratio >= 0.95 and len(txt.split()) <= 8:
            continue
        cleaned.append({"id": fid, "text": txt})
    if len({str(fn["id"]) for fn in cleaned}) < 2:
        return None

    return idx, cleaned


def _convert_inline_bare_letter_markers_to_refs(
    text: str,
    valid_letter_ids: List[str],
) -> str:
    """
    Convert bare inline note markers like `... the c Khootbah ...` to markdown refs
    (`... the [^c] Khootbah ...`) using only letters known to be note IDs on that page.

    Conservative gate:
    - only convert inside paragraphs that contain >=2 distinct candidate markers
    """
    if not text or not valid_letter_ids:
        return text

    valid = {str(x).strip().lower() for x in valid_letter_ids if re.fullmatch(r'[a-z]', str(x).strip().lower())}
    if not valid:
        return text

    marker_pat = re.compile(r'(^|[^A-Za-z\]\[])' r'([a-z])' r'(?=\s+[A-Z][A-Za-z])')

    def _convert_para(para: str) -> str:
        matches = [m for m in marker_pat.finditer(para) if m.group(2).lower() in valid]
        if not matches:
            return para

        distinct = {m.group(2).lower() for m in matches}
        if len(distinct) < 2:
            return para

        def repl(m: re.Match) -> str:
            prefix = m.group(1)
            fid = m.group(2).lower()
            if fid not in valid:
                return m.group(0)
            return f"{prefix}[^{fid}]"

        return marker_pat.sub(repl, para)

    parts = re.split(r'(\n\s*\n)', text)
    changed = False
    for i, part in enumerate(parts):
        if not part or re.fullmatch(r'\n\s*\n', part):
            continue
        # Skip existing markdown blocks/footnote defs-heavy paragraphs.
        first_line = next((ln.strip() for ln in part.splitlines() if ln.strip()), "")
        if first_line.startswith(("#", "|", ">", "[^")) and "[^" not in part:
            continue
        new_part = _convert_para(part)
        if new_part != part:
            parts[i] = new_part
            changed = True
    return "".join(parts) if changed else text


def extract_footnotes(
    text: str,
    *,
    remove_from_body: bool = True,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extract footnotes from page text.

    Bug Fix #2: Supports multi-line footnotes and lettered/symbol footnotes.
    Previously only single-line [^N]: footnotes were captured.

    Returns:
        Tuple of (body_text, list_of_footnotes)
        Each footnote is {"id": "...", "text": "..."}
        If remove_from_body=False, the original body text is preserved and
        footnotes are only collected (useful for archival/master output).
    """
    if not text:
        return (text, [])

    text = _normalize_tex_text_superscript_footnote_markers(text)
    text = _normalize_math_trailing_superscript_footnote_markers(text)
    text = _normalize_embedded_dollar_caret_footnote_markers(text)
    text = _normalize_plain_caret_footnote_markers(text)
    # Normalize visible superscript note markers (¹²³...) to markdown refs so
    # downstream detection/placement can keep notes clickable in reader output.
    text = _normalize_superscript_footnote_markers(text)

    lines = text.split('\n')
    footnotes = []
    body_lines = []
    seen_defs = set()
    plain_letter_note_ids_seen: List[str] = []

    i = 0
    in_footnote = False
    current_footnote_id = None
    current_footnote_lines = []

    while i < len(lines):
        line = lines[i]

        # Some OCR pages pack multiple markdown footnote defs on one line:
        # "[^2]: ...    [^3] ...    [^4] ..."
        packed_defs = _split_packed_markdown_footnote_def_line(line)
        if packed_defs:
            lines[i:i+1] = packed_defs
            line = lines[i]

        # Some OCR pages pack multiple TeX-like footnote defs on one line:
        # "$^2$ ... \t$^3$ ... \t$^4$ ..."
        # Split them into independent definition lines before parsing.
        if re.match(r'^\s*\$\^\{?[\w*†‡§\d-]+\}?\$\s*', line):
            dollar_markers = list(re.finditer(r'\$\^\{?[\w*†‡§\d-]+\}?\$', line))
            if len(dollar_markers) > 1:
                split_defs = []
                for idx_m, match in enumerate(dollar_markers):
                    start = match.start()
                    end = dollar_markers[idx_m + 1].start() if idx_m + 1 < len(dollar_markers) else len(line)
                    segment = line[start:end].strip()
                    if segment:
                        split_defs.append(segment)
                if split_defs:
                    lines[i:i+1] = split_defs
                    line = lines[i]

        # OCR can emit raw lettered note blocks without any explicit footnote marker:
        #   a text...
        #   b text...
        # Parse and collect them before generic body handling.
        plain_letter_run = _parse_plain_letter_footnote_run(lines, i)
        if plain_letter_run:
            run_end, parsed_defs = plain_letter_run
            for fn in parsed_defs:
                fid = str(fn.get("id", "")).strip()
                ftxt = str(fn.get("text", "")).strip()
                key = (fid, ftxt)
                if fid and ftxt and key not in seen_defs:
                    footnotes.append({"id": fid, "text": ftxt})
                    seen_defs.add(key)
                    if re.fullmatch(r'[a-z]', fid) and fid not in plain_letter_note_ids_seen:
                        plain_letter_note_ids_seen.append(fid)
            i = run_end
            continue

        # Match footnote definition:
        # [^N]: text   or   [^N] text
        # $^N$ text    or   $^{N}$ text
        match_line = _strip_footnote_leading_wrappers(line)
        fn_match = re.match(r'^\s*\[\^([\w*†‡§\d-]+)\](?::|\s)\s*(.*)', match_line)
        if not fn_match:
            fn_match = re.match(r'^\s*\$\^\{?([\w*†‡§\d-]+)\}?\$\s*(.*)', match_line)

        if fn_match:
            # Save previous footnote if any
            if in_footnote and current_footnote_id:
                fn_text = '\n'.join(current_footnote_lines).strip()
                key = (current_footnote_id, fn_text)
                if fn_text and key not in seen_defs:
                    footnotes.append({"id": current_footnote_id, "text": fn_text})
                    seen_defs.add(key)

            # Start new footnote
            current_footnote_id = fn_match.group(1)
            current_footnote_lines = [fn_match.group(2).strip()]
            in_footnote = True
            i += 1
            continue

        # Continuation of a multi-line footnote.
        # OCR often emits wrapped footnote lines without indentation; keep consuming
        # non-empty lines until a blank line or a new footnote marker is encountered.
        if in_footnote:
            if not line.strip():
                if current_footnote_id:
                    fn_text = '\n'.join(current_footnote_lines).strip()
                    key = (current_footnote_id, fn_text)
                    if fn_text and key not in seen_defs:
                        footnotes.append({"id": current_footnote_id, "text": fn_text})
                        seen_defs.add(key)
                in_footnote = False
                current_footnote_id = None
                current_footnote_lines = []
                i += 1
                continue

            current_footnote_lines.append(_strip_footnote_leading_wrappers(line))
            i += 1
            continue

        body_lines.append(line)
        i += 1

    # Don't forget the last footnote
    if in_footnote and current_footnote_id:
        fn_text = '\n'.join(current_footnote_lines).strip()
        key = (current_footnote_id, fn_text)
        if fn_text and key not in seen_defs:
            footnotes.append({"id": current_footnote_id, "text": fn_text})

    body_text = '\n'.join(body_lines) if remove_from_body else text
    if remove_from_body and body_text:
        tex_ref_pat = re.compile(r'\$\^\{?([\w*†‡§\d-]+)\}?\$')

        def _tex_ref_repl(match: re.Match) -> str:
            fid = str(match.group(1) or "").strip()
            prev_char = body_text[match.start() - 1] if match.start() > 0 else ""
            next_char = body_text[match.end()] if match.end() < len(body_text) else ""
            if _looks_like_ordinal_suffix_marker_id(fid, prev_char=prev_char, next_char=next_char):
                return fid
            return f"[^{fid}]"

        body_text = tex_ref_pat.sub(_tex_ref_repl, body_text)
        if plain_letter_note_ids_seen:
            body_text = _convert_inline_bare_letter_markers_to_refs(body_text, plain_letter_note_ids_seen)
    return (body_text, footnotes)


# ─── Text Cleaning ───────────────────────────────────────────────────────────

def clean_page(text: str, archival_mode: bool = False) -> str:
    """
    Clean a page's text for assembly.

    Bug Fix #1: Strips any remaining LLM commentary from body text.
    """
    if not text:
        return ""

    # Filter blank page commentary
    text = filter_blank_page_text(text)

    # Normalize whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if not archival_mode:
            # Reader edition cleanup: drop common scan separator artifacts.
            if re.match(r'^\s*[_\-–—]{5,}\s*$', line):
                continue
            if re.match(r'^\s*[\\/]\s*$', line):
                continue
        # Remove trailing whitespace
        cleaned.append(line.rstrip())

    return '\n'.join(cleaned)


# ─── Content Formatting ──────────────────────────────────────────────────────

def format_content(text: str, preserve_nonchapter_headings: bool = True) -> str:
    """
    Format page content, handling markdown headings.

    Chapter detection is handled separately. Preserve body headings by default
    so subheadings/sections survive assembly.
    """
    if not text:
        return ""

    lines = text.split('\n')
    formatted = []

    for line in lines:
        stripped = line.strip()

        # Check if this is a markdown heading
        if stripped.startswith('#'):
            heading_text = stripped.lstrip('#').strip()

            if preserve_nonchapter_headings:
                formatted.append(line)
            # Keep if it matches chapter keywords
            elif CHAPTER_KEYWORDS.match(heading_text):
                formatted.append(line)
            else:
                # Strip the # markers — treat as regular text
                formatted.append(heading_text)
        else:
            formatted.append(line)

    return '\n'.join(formatted)


def _is_markdown_structural_line(line: str) -> bool:
    """Identify markdown structural lines that should not be prose-styled."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(('#', '---', '|', '>')):
        return True
    if stripped.startswith('[^'):
        return True
    if re.match(r'^[-+*]\s+', stripped):
        return True
    if re.match(r'^\d+\.\s+', stripped):
        return True
    return False


def _strip_soft_indent(line: str) -> str:
    """
    Remove leading indentation from prose lines so markdown renderers do not
    interpret wrapped text as code blocks.
    """
    if not line:
        return line
    stripped = line.lstrip()
    if not stripped:
        return ""
    if _is_markdown_structural_line(stripped):
        return stripped
    return stripped


def _restore_likely_paragraph_breaks(text: str) -> str:
    """
    Re-insert paragraph breaks when OCR joins a closing token directly into a
    quoted paragraph start (common around page boundaries).
    """
    out = text
    out = re.sub(
        r'(\[\^[\w*†‡§\d-]+\])(?=[\'"“‘][A-Z])',
        r'\1\n\n',
        out,
    )
    out = re.sub(
        r'([.!?]["”’\')\]])(?=[\'"“‘][A-Z])',
        r'\1\n\n',
        out,
    )
    return out


def _looks_like_letter_block(lines: List[str]) -> bool:
    """Avoid styling letter excerpts as poetry."""
    patterns = (
        r'^(?:dear|my dear|sir|gentlemen)\b',
        r'^(?:to|from)\b',
        r'^(?:yours sincerely|yours faithfully|respect and regard)\b',
        r'\bcalcutta\b',
        r'\besq\.?\b',
    )
    checks = [line.strip().lower() for line in lines if line.strip()]
    if not checks:
        return False
    for line in checks[:8]:
        for pat in patterns:
            if re.search(pat, line):
                return True
    return False


def _is_verse_like_block(raw_lines: List[str], normalized_lines: List[str], prefer_poetry: bool) -> bool:
    """Heuristic detection for verse blocks."""
    lines = [line for line in normalized_lines if line.strip()]
    if len(lines) < 3:
        return False
    if any(_is_markdown_structural_line(line) for line in lines):
        return False
    if _looks_like_letter_block(lines):
        return False

    prose_continuations = 0
    for i in range(len(lines) - 1):
        current = lines[i].strip()
        nxt = lines[i + 1].strip()
        if not current or not nxt:
            continue
        if re.search(r'[.!?;:]["”’\')\]]*$', current):
            continue
        if re.match(r'^[a-z]', nxt):
            prose_continuations += 1
    if prose_continuations >= max(1, len(lines) // 3):
        return False

    word_counts = [len(line.split()) for line in lines]
    avg_words = sum(word_counts) / max(1, len(word_counts))
    if avg_words > 9.5:
        return False
    if max(word_counts, default=0) > 16:
        return False

    short_lines = sum(1 for wc in word_counts if wc <= 12)
    short_ratio = short_lines / len(lines)
    indented = sum(1 for line in raw_lines if line[:1].isspace())
    indent_ratio = indented / max(1, len(raw_lines))

    if short_ratio >= 0.75 and (prefer_poetry or indent_ratio >= 0.25):
        return True
    if prefer_poetry and short_ratio >= 0.65 and len(lines) >= 4:
        return True
    return False


def _is_quote_like_block(raw_lines: List[str], normalized_lines: List[str]) -> bool:
    """Heuristic detection for quoted prose blocks."""
    lines = [line for line in normalized_lines if line.strip()]
    if not lines:
        return False
    if any(_is_markdown_structural_line(line) for line in lines):
        return False

    first = lines[0].strip()
    last = lines[-1].strip()
    word_counts = [len(line.split()) for line in lines]
    avg_words = sum(word_counts) / max(1, len(word_counts))
    starts_with_quote = bool(first) and first[0] in ('"', "'", '“', '‘')
    ends_with_quote = bool(last) and last[-1] in ('"', "'", '”', '’')
    indented = sum(1 for line in raw_lines if line[:1].isspace())
    prose_continuations = 0
    for i in range(len(lines) - 1):
        current = lines[i].strip()
        nxt = lines[i + 1].strip()
        if not current or not nxt:
            continue
        if re.search(r'[.!?;:]["”’\')\]]*$', current):
            continue
        if re.match(r'^[a-z]', nxt):
            prose_continuations += 1

    if starts_with_quote and len(lines) <= 40:
        return True
    if starts_with_quote and ends_with_quote:
        return True
    if (
        indented >= max(3, len(lines) // 2)
        and len(lines) <= 20
        and avg_words <= 9.0
        and prose_continuations == 0
    ):
        return True
    return False


def _as_blockquote(lines: List[str], italics: bool = False) -> str:
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if italics and "*" not in stripped:
            out.append(f"> *{stripped}*")
        else:
            out.append(f"> {stripped}")
    return '\n'.join(out)


def normalize_publish_markdown(
    text: str,
    *,
    allow_blockquotes: bool = True,
    prefer_poetry: bool = False,
) -> str:
    """
    Final deterministic cleanup for publication-safe markdown.
    - removes accidental leading indentation that creates code blocks
    - restores likely paragraph breaks for merged quote starts
    - styles verse as blockquote+italics and quoted prose as blockquote
    """
    if not text:
        return ""

    normalized = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', '    ')
    normalized = _restore_likely_paragraph_breaks(normalized)

    blocks = re.split(r'\n{2,}', normalized)
    out_blocks: List[str] = []

    for block in blocks:
        raw_lines = [line.rstrip() for line in block.split('\n')]
        if not any(line.strip() for line in raw_lines):
            continue

        cleaned_lines = [_strip_soft_indent(line) for line in raw_lines]
        content_lines = [line for line in cleaned_lines if line.strip()]

        if allow_blockquotes and _is_verse_like_block(raw_lines, cleaned_lines, prefer_poetry):
            out_blocks.append(_as_blockquote(content_lines, italics=True))
            continue

        if allow_blockquotes and _is_quote_like_block(raw_lines, cleaned_lines):
            out_blocks.append(_as_blockquote(content_lines, italics=False))
            continue

        out_blocks.append('\n'.join(cleaned_lines).strip('\n'))

    return '\n\n'.join(block for block in out_blocks if block.strip())


MARGINALIA_ERA_DATE_LINE = re.compile(
    r'''(?ix)^
    (?:a\s*[.;:]?\s*d\s*\.? | b\s*[.;:]?\s*c\s*\.? | a\s*[.;:]?\s*h\s*\.?)
    \s*
    \d{2,4}
    (?:\s*[-–—/]\s*\d{1,4})?
    [a-z]?
    (?:\s*[.,;:])?
    $'''
)

MARGINALIA_CALENDAR_YEAR_LABEL_LINE = re.compile(
    r'''(?ix)^
    (?:
      hijri | hijrah | hejira |
      saka | śaka | saka\ era |
      samvat | vikrama(?:\s+samvat)? |
      fasli
    )
    \s*
    \d{2,4}
    (?:\s*[-–—/]\s*\d{1,4})?
    [a-z]?
    (?:\s*[.,;:])?
    $'''
)

MARGINALIA_REGNAL_YEAR_LABEL_LINE = re.compile(
    r'''(?ix)^
    (?:
      regnal\s+year\s+\d{1,3} |
      (?:year\s+)?of\s+reign\s+\d{1,3} |
      \d{1,3}(?:st|nd|rd|th)\s+year(?:\s+of\s+(?:reign|his\s+reign))? |
      (?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|
         eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|
         seventeenth|eighteenth|nineteenth|twentieth)
      \s+year(?:\s+of\s+(?:reign|his\s+reign))?
    )
    (?:\s*[.,;:])?
    $'''
)

MARGINALIA_CIRCA_FLORUIT_LINE = re.compile(
    r'''(?ix)^
    (?:
      c(?:irca)? | ca | fl
    )\.?
    \s*
    \d{2,4}
    (?:\s*[-–—/]\s*\d{1,4})?
    (?:\s*[.,;:])?
    $'''
)

MARGINALIA_ERA_DATE_TOKEN = re.compile(
    r'''(?ix)
    (?:
      a\s*[.;:]?\s*d\s*\.? |
      b\s*[.;:]?\s*c\s*\.? |
      a\s*[.;:]?\s*h\s*\.?
    )
    \s*
    \d{2,4}
    (?:\s*[-–—/]\s*\d{1,4})?
    [a-z]?
    (?:\s*[.,;:])?
    '''
)


def _is_strong_marginalia_label_text(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    return bool(
        MARGINALIA_ERA_DATE_LINE.match(t)
        or MARGINALIA_CALENDAR_YEAR_LABEL_LINE.match(t)
        or MARGINALIA_REGNAL_YEAR_LABEL_LINE.match(t)
        or MARGINALIA_CIRCA_FLORUIT_LINE.match(t)
    )


def _split_inline_era_date_marginalia_line(line: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Split strong inline era-date marginalia that OCR flattened into a prose line.
    Returns (mode, left, label, right), where mode is "prefix" or "suffix".

    Examples:
    - "A.D. 1412.    In 815 ..."   -> ("prefix", "", "A.D. 1412.", "In 815 ...")
    - "... the roy of     A.D. 1417." -> ("suffix", "... the roy of", "A.D. 1417.", "")
    """
    s = (line or "").strip()
    if not s:
        return None
    if s.startswith(("#", "|", ">", "[^", "[Illustration", "[Marginal note:")):
        return None
    if _looks_like_footnote_definition_line(s):
        return None
    if _is_strong_marginalia_label_text(s):
        return None  # standalone case handled elsewhere

    # Prefix form: marginalia label starts the line, then prose continues.
    mp = MARGINALIA_ERA_DATE_TOKEN.match(s)
    if mp and mp.start() == 0:
        label = (mp.group(0) or "").strip()
        tail = s[mp.end():]
        gm = re.match(r'^(?P<gap>\s+)(?P<rest>\S.*)$', tail)
        gap = (gm.group("gap") if gm else "")
        rest = (gm.group("rest").strip() if gm else "")
        if (
            rest
            and _is_strong_marginalia_label_text(label)
            and (
                len(gap) >= 2
                or bool(re.match(r'(?i)^(?:in|the|he|she|they|when|towards|on|at|\d)\b', rest))
                or bool(re.match(r'^[a-z]', rest))  # broken word continuation
            )
            and len(rest.split()) >= 2
        ):
            return ("prefix", "", label, rest)

    # Suffix form: prose line ends with a margin date label after a wide gap.
    suffix_match = None
    for m in MARGINALIA_ERA_DATE_TOKEN.finditer(s):
        if m.end() != len(s):
            continue
        suffix_match = m
    if suffix_match is not None:
        label = (suffix_match.group(0) or "").strip()
        left_raw = s[:suffix_match.start()]
        gm = re.search(r'(?P<left>.*?\S)(?P<gap>\s{2,})$', left_raw)
        if gm:
            left = gm.group("left").strip()
            if _is_strong_marginalia_label_text(label):
                if len(left.split()) >= 3 and _looks_like_prose_line_for_marginalia_context(left):
                    return ("suffix", left, label, "")

    return None


def _looks_like_short_marginalia_side_heading(
    line: str,
    *,
    isolated_by_blanks: bool,
    prev_prose: bool,
    next_prose: bool,
) -> bool:
    """
    Conservative detector for marginal side-headings like "Battle of X" or
    "Accession of Y" when OCR flattens margin text into the body stream.
    """
    s = (line or "").strip()
    if not s:
        return False
    if not (isolated_by_blanks and prev_prose and next_prose):
        return False
    if s.startswith(("#", "|", ">", "[^", "[Illustration", "[Marginal note:")):
        return False
    if _looks_like_footnote_definition_line(s):
        return False
    if re.search(r'\d', s):
        return False
    if len(s) > 48:
        return False
    words = s.split()
    if not (2 <= len(words) <= 4):
        return False
    if re.search(r'[!?;]{1,}', s):
        return False
    if re.search(r'(?i)\b(chapter|book|part|index|preface|introduction|notes?)\b', s):
        return False
    alpha = [c for c in s if c.isalpha()]
    if len(alpha) < 6:
        return False
    # Prefer title-case phrases, allow connector words.
    connector_words = {"of", "the", "and", "to", "in", "for", "on"}
    good = 0
    for w in words:
        token = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w)
        if not token:
            continue
        if token.lower() in connector_words:
            good += 1
            continue
        if re.match(r"^[A-Z][a-z'’-]{1,}$", token):
            good += 1
    if good < len(words):
        return False
    # All-caps short labels are more often headers/running heads.
    upper_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))
    if upper_ratio >= 0.8:
        return False
    return True


def _looks_like_prose_line_for_marginalia_context(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if s.startswith(("#", "|", ">", "[^", "[Illustration")):
        return False
    if _looks_like_footnote_definition_line(s):
        return False
    if re.fullmatch(r'[\-—–_*=·•♦◈\s]{3,}', s):
        return False
    words = s.split()
    if len(words) < 3:
        return False
    alpha = [c for c in s if c.isalpha()]
    if len(alpha) < 5:
        return False
    # Pure uppercase short labels are more likely headers than prose.
    upper_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))
    if upper_ratio >= 0.85 and len(words) <= 10:
        return False
    return True


def _tag_reader_marginalia_lines(text: str) -> Tuple[str, List[str]]:
    """
    Reader-edition only: preserve likely side-margin notes (especially era/date
    markers like "A.D. 1407.") as explicit inline markers instead of leaving
    them as ambiguous stray lines in prose flow.

    This is intentionally conservative and currently only targets strong
    date-label marginalia patterns.
    """
    if not text:
        return text, []

    lines = text.split("\n")
    out: List[str] = []
    tagged: List[str] = []

    def _nearest_nonempty(idx: int, step: int, limit: int = 4) -> str:
        j = idx + step
        hops = 0
        while 0 <= j < len(lines) and hops < limit:
            if lines[j].strip():
                return lines[j]
            j += step
            hops += 1
        return ""

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            out.append(line)
            continue

        if s.startswith("[Marginal note:"):
            out.append(line)
            continue

        inline_split = _split_inline_era_date_marginalia_line(s)
        if inline_split is not None:
            mode, left_txt, label_txt, right_txt = inline_split
            if mode == "prefix":
                if out and out[-1].strip():
                    out.append("")
                out.append(f"[Marginal note: {label_txt}]")
                out.append("")
                out.append(right_txt)
                tagged.append(label_txt)
                continue
            if mode == "suffix":
                out.append(left_txt)
                out.append("")
                out.append(f"[Marginal note: {label_txt}]")
                tagged.append(label_txt)
                continue

        immediate_prev_blank = (i == 0) or (not lines[i - 1].strip())
        immediate_next_blank = (i + 1 >= len(lines)) or (not lines[i + 1].strip())
        isolated_by_blanks = immediate_prev_blank and immediate_next_blank

        is_chronology_marginalia = _is_strong_marginalia_label_text(s)
        if not is_chronology_marginalia and not isolated_by_blanks:
            out.append(line)
            continue

        prev_line = _nearest_nonempty(i, -1)
        next_line = _nearest_nonempty(i, +1)
        prev_prose = _looks_like_prose_line_for_marginalia_context(prev_line)
        next_prose = _looks_like_prose_line_for_marginalia_context(next_line)

        is_short_heading_marginalia = _looks_like_short_marginalia_side_heading(
            s,
            isolated_by_blanks=isolated_by_blanks,
            prev_prose=prev_prose,
            next_prose=next_prose,
        )
        if not (is_chronology_marginalia or is_short_heading_marginalia):
            out.append(line)
            continue

        # Require prose context on at least one side to avoid reclassifying
        # true headings or standalone chronology tables.
        if not (prev_prose or next_prose):
            out.append(line)
            continue

        if out and out[-1].strip() and (prev_prose or next_prose):
            out.append("")
        out.append(f"[Marginal note: {s}]")
        if i + 1 < len(lines) and next_line.strip() and (prev_prose or next_prose):
            out.append("")
        tagged.append(s)

    return "\n".join(out), tagged


def _render_structural_heading_lines(title: str, heading_level: int) -> List[str]:
    """
    Render heading lines, optionally splitting composite BOOK/PART wrapper titles
    into clearer hierarchical headings for reader output.
    """
    title = (title or "").strip()
    if not title:
        return []

    # Example: "BOOK I. HINDU PERIOD. PART I. Indications..."
    m = re.match(r'(?is)^(book\b.*?)(?:(?:\s+[—-]\s*)?(part\b.*))$', title)
    if m and "chapter" not in title.lower():
        book_part = m.group(1).strip()
        part_part = m.group(2).strip()
        if book_part.lower().startswith("book") and part_part.lower().startswith("part"):
            book_part = re.sub(r'[\s—-]+$', '', book_part).strip()
            book_prefix = "#" * max(2, min(6, heading_level))
            part_prefix = "#" * max(2, min(6, heading_level + 1))
            return [f"{book_prefix} {book_part}", "", f"{part_prefix} {part_part}"]

    prefix = "#" * max(2, min(6, heading_level))
    return [f"{prefix} {title}"]


def format_chapter(
    chapter: Chapter,
    emit_notes_section: bool = True,
    heading_level: int = 2,
) -> str:
    """Format a chapter as markdown."""
    parts = []

    # Chapter heading
    parts.extend(_render_structural_heading_lines(chapter.title, heading_level))
    parts.append("")

    # Body text
    if chapter.body.strip():
        parts.append(chapter.body.strip())
        parts.append("")

    # Footnotes (Bug Fix #2: only emit section if there are footnotes)
    if emit_notes_section and chapter.footnotes:
        parts.append("---")
        parts.append("")
        parts.append("##### Notes")
        parts.append("")
        for fn in chapter.footnotes:
            fid = str(fn.get("id", "")).strip()
            ftext = str(fn.get("text", "")).strip()
            if not fid or not ftext:
                continue
            flines = ftext.splitlines() or [ftext]
            parts.append(f"[^{fid}]: {flines[0].strip()}")
            for cont in flines[1:]:
                cont = cont.rstrip()
                if cont:
                    parts.append(f"    {cont}")
                else:
                    parts.append("")
        parts.append("")

    return '\n'.join(parts)


def _first_nonempty_line(text: str) -> str:
    for line in text.split('\n'):
        if line.strip():
            return line.strip()
    return ""


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.split('\n')):
        if line.strip():
            return line.strip()
    return ""


def _strip_repeated_boundary_catchword(left_text: str, right_text: str) -> Tuple[str, bool]:
    """
    Collapse repeated page-boundary catchwords after a join decision is made.

    Examples:
    - `... againſt` + `againſt Warunkul ...` -> drop right duplicate
    - `... command,` + `command, and ...` -> drop right duplicate
    """
    left = (left_text or "").rstrip()
    right = (right_text or "")
    if not left or not right or left.endswith("-"):
        return right_text, False

    left_m = re.search(r'([A-Za-zſ][A-Za-zſ\'’\-]{3,})([.,;:!?]*)\s*$', left)
    right_m = re.match(r'^([\"\'“‘(\[]*)([A-Za-zſ][A-Za-zſ\'’\-]{3,})\b([.,;:!?]*)(?:\s+|$)', right)
    if not left_m or not right_m:
        return right_text, False

    left_word = left_m.group(1)
    right_word = right_m.group(2)
    if not left_word or not right_word:
        return right_text, False

    norm_left = left_word.lower().replace("ſ", "s")
    norm_right = right_word.lower().replace("ſ", "s")
    if norm_left != norm_right:
        return right_text, False

    # Avoid deleting when the right side is just the same single token (rarely useful).
    remainder = right[right_m.end():]
    if not remainder.strip():
        return right_text, False

    return (right[:right_m.start()] + remainder.lstrip()), True


def _merge_page_boundary(
    prev_text: str,
    next_text: str,
    force: bool = False,
    force_mode: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Merge text across a page break when the boundary clearly indicates continuation.

    Returns:
        (merged_text, did_merge)
    """
    if not prev_text.strip() or not next_text.strip():
        return "", False

    prev_last = _last_nonempty_line(prev_text)
    next_first = _first_nonempty_line(next_text)
    if not prev_last or not next_first:
        return "", False

    # Don't merge into footnote definition starts.
    if _looks_like_footnote_definition_line(next_first):
        return "", False

    # Don't merge when next page begins with a standalone page marker/header fragment.
    if re.match(r'^\[?\(?\d{1,4}\)?\]?[.*]?$', next_first):
        return "", False
    if re.match(r'^[ivxlcdm]{1,8}[.]?$', next_first, flags=re.IGNORECASE):
        return "", False

    # Keep illustration markers as standalone blocks, not inline sentence fragments.
    if next_first.startswith("[Illustration") or prev_last.startswith("[Illustration"):
        return "", False

    terminal_punct = re.search(r'[.!?]["”’\')\]]*$', prev_last)
    continuation_start = bool(re.match(r"^[a-z0-9\"'(\[]", next_first))
    dash_artifact_continuation = bool(re.match(r'^-{2,}\s*[A-Za-z0-9"\'(\[]', next_first))
    requested_mode = str(force_mode or "").strip().lower()
    alias_modes = {
        "join_with_space": "space",
        "join_without_space": "nospace",
        "keephyphen": "keep_hyphen",
        "drophyphen": "drop_hyphen",
        "dehyphenate": "drop_hyphen",
        "none": "",
    }
    if requested_mode in alias_modes:
        requested_mode = alias_modes[requested_mode]
    valid_modes = {"", "auto", "space", "nospace", "keep_hyphen", "drop_hyphen"}
    if requested_mode not in valid_modes:
        requested_mode = ""
    explicit_mode = requested_mode if requested_mode and requested_mode != "auto" else ""

    should_merge = force or bool(explicit_mode)
    if prev_last.endswith('-'):
        should_merge = True
    elif not terminal_punct and (
        continuation_start
        or dash_artifact_continuation
        or prev_last.endswith((',', ';', ':', '—'))
    ):
        should_merge = True

    if not should_merge:
        return "", False

    left = prev_text.rstrip()
    right = next_text.lstrip()

    # OCR/layout cleanup: page starts sometimes retain visual continuation
    # markers ("--"/"---") that should not survive into merged prose.
    if re.match(r'^-{2,}\s*[A-Za-z0-9"\'(\[]', right):
        right = re.sub(r'^-{2,}\s*', '', right, count=1)

    # Historical books often repeat the last word of the prior page as a catchword
    # at the top of the next page. Remove the duplicate before final joining.
    right, _ = _strip_repeated_boundary_catchword(left, right)

    if explicit_mode == "space":
        no_space_prefix = ('.', ',', ';', ':', ')', ']', '}', '"', "'", '”', '’')
        spacer = "" if right.startswith(no_space_prefix) else " "
        if left.endswith('-'):
            left = re.sub(r'-\s*$', '', left)
        merged = left + spacer + right
        return merged, True

    if explicit_mode == "nospace":
        if left.endswith('-'):
            left = re.sub(r'-\s*$', '', left)
        merged = left + right
        return merged, True

    if left.endswith('-'):
        if explicit_mode == "keep_hyphen":
            return left + right, True
        if explicit_mode == "drop_hyphen":
            return re.sub(r'-\s*$', '', left) + right, True
        # Keep semantic compounds like "Durga-Navaratri" (uppercase continuation),
        # but de-hyphenate wrapped words like "archaeo-\nlogical".
        right_starts_upper = bool(re.match(r'^[A-Z]', right))
        if right_starts_upper:
            merged = left + right
        else:
            merged = re.sub(r'-\s*$', '', left) + right
    else:
        if left.endswith(':') and re.match(r'^[\'"“‘][A-Z]', right):
            # Common case at page breaks: prose introduces a quoted block/list.
            merged = left + "\n\n" + right
            return merged, True
        no_space_prefix = ('.', ',', ';', ':', ')', ']', '}', '"', "'", '”', '’')
        spacer = "" if right.startswith(no_space_prefix) else " "
        merged = left + spacer + right

    return merged, True


def _normalize_simple(text: str) -> str:
    """Normalize text for fuzzy heading/header comparisons."""
    norm = text.lower().strip()
    norm = re.sub(r'[^a-z0-9\s#]', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm)
    return norm.strip()


def _superscript_digits_to_ascii(text: str) -> str:
    return (text or "").translate(SUPERSCRIPT_DIGIT_MAP)


def _normalize_superscript_footnote_markers(text: str) -> str:
    """
    Convert raw unicode superscript digit markers (¹²³...) into markdown
    footnote references `[^{id}]`, then collapse obvious duplicate markers.

    This is reader-safe and materially improves note linking in OCR output where
    the extractor preserved visible superscripts instead of normalizing them.
    """
    if not text:
        return text

    superscript_re = re.compile(rf'([{re.escape(SUPERSCRIPT_DIGIT_CHARS)}]+)')

    def repl(match: re.Match) -> str:
        sup = match.group(1)
        digits = _superscript_digits_to_ascii(sup)
        if not digits:
            return sup
        return f"[^{digits}]"

    out = superscript_re.sub(repl, text)
    # Collapse `[^{id}]` immediately followed by the same marker (with optional
    # spacing), e.g. `[^1]¹` -> `[^1]`, after normalization becomes `[^1][^1]`.
    out = re.sub(r'(\[\^([^\]]+)\])\s*(\[\^\2\])', r'\1', out)
    return out


def _normalize_superscript_letter_footnote_markers(
    text: str,
    allowed_letters: Optional[set] = None,
) -> str:
    """
    Convert isolated unicode superscript Latin letters (ᵃ ᵇ ᶜ ...) used as
    visible note markers into markdown footnote refs `[^{letter}]`.
    """
    if not text:
        return text
    if not any(ch in text for ch in SUPERSCRIPT_LETTER_CHARS):
        return text

    allowed_norm: Optional[set] = None
    if allowed_letters is not None:
        allowed_norm = {
            str(x).strip().lower()
            for x in allowed_letters
            if re.fullmatch(r"[a-z]", str(x).strip().lower())
        }
        if not allowed_norm:
            return text

    pat = re.compile(rf'(?<!\[\^)([{re.escape(SUPERSCRIPT_LETTER_CHARS)}])')

    def repl(match: re.Match) -> str:
        ch = match.group(1)
        letter = SUPERSCRIPT_LETTER_TO_ASCII.get(ch)
        if not letter:
            return ch
        if allowed_norm is not None and letter.lower() not in allowed_norm:
            return ch
        return f"[^{letter}]"

    out = pat.sub(repl, text)
    out = re.sub(r'(\[\^([a-z])\])\s*(\[\^\2\])', r'\1', out, flags=re.IGNORECASE)
    return out


def _normalize_embedded_dollar_caret_footnote_markers(text: str) -> str:
    """
    OCR can trap a trailing footnote marker inside a dollar-delimited span, e.g.
    `$Rāmāyaṇa,^3$`. Convert obviously textual cases to `$Rāmāyaṇa,$[^3]`.
    """
    if not text or "^" not in text or "$" not in text:
        return text

    pat = re.compile(r'\$([^$\n]*?)\^(\d{1,3}(?:-[A-Za-z0-9]+)?)\$')

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        fid = match.group(2)
        if not re.search(r'[A-Za-z\u00C0-\u024F]', prefix):
            return match.group(0)
        return f"${prefix}$[^{fid}]"

    return pat.sub(repl, text)


def _normalize_math_trailing_superscript_footnote_markers(text: str) -> str:
    """
    Convert footnote markers trapped as trailing LaTeX superscripts inside
    dollar spans, e.g. `$R\\bar{a}ja-Tara\\dot{n}gin\\bar{i}^{1}$` ->
    `$R\\bar{a}ja-Tara\\dot{n}gin\\bar{i}$[^1]`.

    This is intentionally conservative and only rewrites spans that look
    textual/transliteration-like (contain letters or LaTeX commands).
    """
    if not text or "^{" not in text or "$" not in text:
        return text

    pat = re.compile(r'\$([^$\n]*?)\^\{(\d{1,3}(?:-[A-Za-z0-9]+)?)\}\$')

    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        fid = match.group(2)
        if not re.search(r'[A-Za-z\\\u00C0-\u024F]', prefix):
            return match.group(0)
        return f"${prefix}$[^{fid}]"

    return pat.sub(repl, text)


def _normalize_tex_text_superscript_footnote_markers(text: str) -> str:
    """
    Convert TeX text-style superscript markers such as `$^\\text{a}$` or
    `$^{\\mathrm{1}}$` into markdown footnote refs.
    """
    if not text or "^" not in text or "$" not in text or "\\" not in text:
        return text

    pat = re.compile(
        r'\$'
        r'\^\s*'
        r'(?:\{\s*)?'
        r'\\(?:text|mathrm)\s*'
        r'\{\s*([A-Za-z0-9]{1,4}(?:-[A-Za-z0-9]+)?)\s*\}'
        r'(?:\s*\})?'
        r'\$'
    )

    def repl(match: re.Match) -> str:
        fid = str(match.group(1) or "").strip()
        prev_char = text[match.start() - 1] if match.start() > 0 else ""
        next_char = text[match.end()] if match.end() < len(text) else ""
        if _looks_like_ordinal_suffix_marker_id(fid, prev_char=prev_char, next_char=next_char):
            return fid
        return f"[^{fid}]"

    return pat.sub(repl, text)


def _normalize_plain_caret_footnote_markers(text: str) -> str:
    """
    Convert inline plain caret refs like `word.^1` or `^a` (outside `$...$`)
    into markdown footnote refs.
    """
    if not text or "^" not in text:
        return text

    out: List[str] = []
    i = 0
    in_dollar = False
    while i < len(text):
        ch = text[i]
        if ch == "$":
            in_dollar = not in_dollar
            out.append(ch)
            i += 1
            continue
        if not in_dollar and ch == "^":
            if i > 0 and text[i - 1] == "[":
                out.append(ch)
                i += 1
                continue
            if i > 0 and text[i - 1] == "\\":
                out.append(ch)
                i += 1
                continue
            m = re.match(r'\^([A-Za-z0-9]{1,4}(?:-[A-Za-z0-9]+)?)\b', text[i:])
            if m:
                fid = str(m.group(1) or "").strip()
                prev_char = text[i - 1] if i > 0 else ""
                next_idx = i + len(m.group(0))
                next_char = text[next_idx] if next_idx < len(text) else ""
                if _looks_like_ordinal_suffix_marker_id(fid, prev_char=prev_char, next_char=next_char):
                    out.append(fid)
                else:
                    out.append(f"[^{fid}]")
                i += len(m.group(0))
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_packed_markdown_footnote_def_line(line: str) -> Optional[List[str]]:
    """
    Split lines that contain multiple markdown footnote definitions packed
    together by OCR/layout loss, e.g.
    `[^2]: ...      [^3] ...      [^4] ...`
    """
    if not line:
        return None
    stripped = _strip_footnote_leading_wrappers(line)
    if not re.match(r'^\[\^[^\]]+\](?::|\s)', stripped):
        return None

    marker_re = re.compile(r'\[\^[\w*†‡§\d-]+\](?::|\s)')
    markers = list(marker_re.finditer(line))
    if len(markers) < 2:
        return None

    split_points = [0]
    for m in markers[1:]:
        # Only split if OCR inserted a clear spacing gap before the next marker.
        if re.search(r'(?:\s{2,}|\t+)$', line[:m.start()]):
            split_points.append(m.start())
    if len(split_points) < 2:
        return None

    split_points.append(len(line))
    segments: List[str] = []
    for a, b in zip(split_points, split_points[1:]):
        seg = line[a:b].strip()
        if seg:
            segments.append(seg)
    return segments or None


def _canonical_illustration_marker_from_caption(caption: str) -> str:
    c = (caption or "").strip()
    c = re.sub(r'\s+', ' ', c)
    c = re.sub(r'^\[?\s*illustration\s*:?\s*', '', c, flags=re.IGNORECASE)
    # Reader output should not expose library ownership prefixes inside captions.
    c = re.sub(
        r'(?i)^(?:univ(?:ersity)?\.?\s+of\s+california|u\.?\s*c\.?\s*berkeley\s+libraries|general\s+library)\b[\s:;,\-]*',
        '',
        c,
    )
    c = re.sub(r'(?i)^(?:univ\.?\s+of\s+california)\s+', '', c)
    c = c.strip("[] ")
    if not c:
        return "[Illustration]"
    return f"[Illustration: {c}]"


def _strip_footnote_leading_wrappers(line: str) -> str:
    """
    Ignore markdown quote/emphasis wrappers often wrapped around OCR notes,
    e.g. `> *¹ text*` so definition detection can still see the marker.
    """
    s = (line or "").lstrip()
    while s.startswith(">"):
        s = s[1:].lstrip()
    # Strip one leading emphasis run if it wraps the whole line.
    s = re.sub(r'^(?:\*\*|\*|__|_)\s*', '', s)
    s = re.sub(r'\s*(?:\*\*|\*|__|_)\s*$', '', s)
    return s.strip()


def _looks_like_footnote_definition_line(line: str) -> bool:
    stripped = _strip_footnote_leading_wrappers(line)
    if not stripped:
        return False
    if re.match(r'^\[\^[^\]]+\](?::|\s)', stripped):
        return True
    if re.match(r'^\$\^[^$]+\$\s+', stripped):
        return True
    if re.match(rf'^[{re.escape(SUPERSCRIPT_DIGIT_CHARS)}]+\s+', stripped):
        return True
    return False


def _dedup_chapter_footnotes(footnotes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    De-duplicate footnotes at chapter level (Fix #5).
    When pages are aggregated into chapters, the same footnote ID can appear
    from multiple pages. Keep the first occurrence.
    """
    seen = set()
    result = []
    for fn in footnotes:
        key = (fn.get("id"), (fn.get("text") or "").strip())
        if key not in seen:
            seen.add(key)
            result.append(fn)
    return result


def _normalize_footnote_text_for_compare(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def _remap_colliding_page_footnote_ids(
    body_text: str,
    footnotes: List[Dict[str, str]],
    existing_footnotes: List[Dict[str, str]],
    page_number: int,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    If a numeric/short footnote ID repeats within a chapter with different text,
    rewrite the page-local references/definitions to a stable page-scoped ID.

    This preserves all notes in reader chapter-end rendering while keeping
    markdown footnote IDs unique.
    """
    if not footnotes:
        return body_text, footnotes

    existing_by_id: Dict[str, List[str]] = {}
    for fn in existing_footnotes:
        fid = str(fn.get("id", ""))
        txt = _normalize_footnote_text_for_compare(str(fn.get("text", "")))
        existing_by_id.setdefault(fid, []).append(txt)

    updated_body = body_text
    used_ids = {str(fn.get("id", "")) for fn in existing_footnotes}
    remapped: List[Dict[str, str]] = []

    for fn in footnotes:
        fid = str(fn.get("id", ""))
        ftxt = str(fn.get("text", ""))
        norm_text = _normalize_footnote_text_for_compare(ftxt)
        prior_texts = existing_by_id.get(fid, [])

        new_id = fid
        if prior_texts and norm_text not in prior_texts:
            base = f"{fid}-p{page_number}"
            candidate = base
            suffix = 2
            while candidate in used_ids:
                candidate = f"{base}-{suffix}"
                suffix += 1
            new_id = candidate

            # Repoint only body references (definitions have already been removed).
            updated_body = re.sub(
                rf'\[\^{re.escape(fid)}\]',
                f"[^{new_id}]",
                updated_body,
            )

        remapped_fn = dict(fn)
        remapped_fn["id"] = new_id
        remapped.append(remapped_fn)
        used_ids.add(new_id)
        existing_by_id.setdefault(new_id, []).append(norm_text)
        if new_id != fid:
            existing_by_id.setdefault(fid, []).append(norm_text)

    return updated_body, remapped


def _footnote_sort_key(fn: Dict[str, str]) -> Tuple[int, Any]:
    fid = str(fn.get("id", "")).strip()
    if fid.isdigit():
        return (0, int(fid))
    return (1, fid.lower())


def _footnote_base_id(fid: str) -> str:
    s = str(fid or "").strip()
    m = re.match(r"^(.*?)-p\d+(?:-\d+)?$", s)
    return m.group(1) if m else s


def _is_page_scoped_footnote_id(fid: str) -> bool:
    s = str(fid or "").strip()
    return bool(re.search(r"-p\d+(?:-\d+)?$", s))


def _footnote_inline_marker_letter(fid: str) -> Optional[str]:
    """
    Infer visible inline marker letter for chapter footnote ids when present.
    """
    s = str(fid or "").strip().lower()
    if re.fullmatch(r"[a-z]", s):
        return s
    m = re.fullmatch(r"([a-z])-p\d+(?:-\d+)?", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"c\d+-([a-z])(?:-\d+)?", s)
    if m:
        return m.group(1)
    return None


def _collect_resolvable_letter_markers(footnotes: List[Dict[str, str]]) -> set:
    """
    Marker letters that can be safely represented as plain refs (`[^a]`) and
    reconciled to chapter defs by existing logic.
    Supported families:
    - a
    - a-p123 / a-p123-2
    Excludes cN-a style IDs, which are not base-compatible with plain `[^a]`.
    """
    out: set = set()
    for fn in footnotes or []:
        fid = str(fn.get("id", "")).strip().lower()
        if re.fullmatch(r"[a-z]", fid):
            out.add(fid)
            continue
        m = re.fullmatch(r"([a-z])-p\d+(?:-\d+)?", fid)
        if m:
            out.add(m.group(1))
    return out


def _looks_like_discourse_cue_not_heading(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    # Common OCR/prose transition cues that are not section headings.
    return bool(re.match(r'(?i)^(again|thus|hence|further|moreover)\s*[:—–-]+\s*$', s))


def _reconcile_chapter_footnote_references(
    body_text: str,
    footnotes: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Reconcile chapter body refs with page-scoped footnote IDs produced during
    assembly (`1-p120`, `2-p56`, ...).

    Failure mode addressed:
    - ref appears on one page as `[^2]`
    - note definition appears later and gets remapped to `[^2-p56]: ...`
    - the earlier ref is never updated, leaving `2-p56` unreferenced

    We deterministically reassign later plain refs (`[^2]`) to missing page-
    scoped variants in reading order, preserving the first plain occurrence for
    the plain base ID when present.
    """
    if not body_text or not footnotes:
        return body_text, footnotes

    ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
    matches = list(ref_pat.finditer(body_text))
    if not matches:
        return body_text, footnotes

    # Ordered unique definition IDs per base, preserving page order.
    defs_by_base: Dict[str, List[str]] = {}
    for fn in footnotes:
        fid = str(fn.get("id", "")).strip()
        if not fid:
            continue
        base = _footnote_base_id(fid)
        bucket = defs_by_base.setdefault(base, [])
        if fid not in bucket:
            bucket.append(fid)

    if not defs_by_base:
        return body_text, footnotes

    ref_ids = [m.group(1) for m in matches]
    ref_counts = Counter(ref_ids)
    planned_by_occurrence: Dict[int, str] = {}

    for base, def_ids in defs_by_base.items():
        if len(def_ids) <= 1:
            continue

        # We only patch missing page-scoped variants; leave already-linked refs alone.
        missing_variant_ids = [
            fid for fid in def_ids
            if fid != base and ref_counts.get(fid, 0) == 0
        ]
        if not missing_variant_ids:
            continue

        plain_occ_idxs = [i for i, rid in enumerate(ref_ids) if rid == base]
        if not plain_occ_idxs:
            continue

        # Preserve one plain reference for the base ID when a plain definition exists.
        start_at = 1 if base in def_ids else 0
        candidate_occ_idxs = [
            occ_i for occ_i in plain_occ_idxs[start_at:]
            if occ_i not in planned_by_occurrence
        ]
        if not candidate_occ_idxs:
            continue

        assign_count = min(len(candidate_occ_idxs), len(missing_variant_ids))
        for occ_i, new_id in zip(candidate_occ_idxs[:assign_count], missing_variant_ids[:assign_count]):
            planned_by_occurrence[occ_i] = new_id
            ref_counts[new_id] = ref_counts.get(new_id, 0) + 1
            if ref_counts.get(base, 0) > 0:
                ref_counts[base] -= 1

    if not planned_by_occurrence:
        return body_text, footnotes

    pieces: List[str] = []
    last_end = 0
    occ_idx = 0
    for m in matches:
        pieces.append(body_text[last_end:m.start()])
        rid = m.group(1)
        new_id = planned_by_occurrence.get(occ_idx, rid)
        pieces.append(f"[^{new_id}]")
        last_end = m.end()
        occ_idx += 1
    pieces.append(body_text[last_end:])

    return "".join(pieces), footnotes


def _chapter_footnote_integrity_counts(
    body_text: str,
    footnotes: List[Dict[str, str]],
) -> Dict[str, int]:
    """
    Lightweight chapter-local footnote integrity counts used for fail-safe
    gating of experimental assembly ops.
    """
    ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
    refs = {m.group(1) for m in ref_pat.finditer(body_text or "")}
    defs = {
        str(fn.get("id", "")).strip()
        for fn in (footnotes or [])
        if str(fn.get("id", "")).strip()
    }
    orphaned = refs - defs
    unreferenced = defs - refs
    return {
        "refs": len(refs),
        "defs": len(defs),
        "orphaned": len(orphaned),
        "unreferenced": len(unreferenced),
        "mismatches": len(orphaned) + len(unreferenced),
    }


def _heal_paragraph_breaks_after_footnote_refs(text: str) -> str:
    """
    Rejoin paragraphs split by removed footnote-definition blocks.

    Typical failure:
    - prose ... `word[^1]`
    - (footnote block removed)
    - continuation starts in next paragraph `and ...`

    We only join when the first paragraph ends with a footnote ref and the next
    paragraph looks like continuation (lowercase / opening quote+lowercase /
    punctuation-leading continuation), to avoid damaging real paragraph breaks.
    """
    if not text or "[^" not in text:
        return text

    paras = text.split("\n\n")
    if len(paras) < 2:
        return text

    out: List[str] = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        if i + 1 >= len(paras):
            out.append(cur)
            break

        nxt = paras[i + 1]
        cur_last = _last_nonempty_line(cur)
        nxt_first = _first_nonempty_line(nxt)
        if not cur_last or not nxt_first:
            out.append(cur)
            i += 1
            continue

        cur_ends_with_ref = bool(re.search(r'\[\^[^\]]+\]["”’\')\]]*\s*$', cur_last))
        # Continuation signals: lowercase, punctuation-leading, or quote/paren + lowercase.
        continuationish = bool(re.match(r"^(?:[a-z]|['\"“‘(\\[]\s*[a-z]|[,:;)\]])", nxt_first))
        if cur_ends_with_ref and continuationish:
            left = cur.rstrip()
            right = nxt.lstrip()
            spacer = "" if right[:1] in ",.;:)]}\"”’" else " "
            out.append(left + spacer + right)
            i += 2
            continue

        out.append(cur)
        i += 1

    return "\n\n".join(out)


def _uniquify_footnote_ids_across_chapters(chapters: List["Chapter"]) -> List[Dict[str, Any]]:
    """
    Markdown footnote labels must be unique within the full assembled file.
    Reuse of `[^1]` across chapters is fine conceptually but breaks markdown
    linkage and strict QA. Rewrite duplicate labels to chapter-scoped IDs.

    Returns a list of change summaries for provenance logging.
    """
    if not chapters:
        return []

    global_counts: Counter = Counter()
    for ch in chapters:
        for fn in ch.footnotes or []:
            fid = str(fn.get("id", "")).strip()
            if fid:
                global_counts[fid] += 1

    duplicate_ids = {fid for fid, count in global_counts.items() if count > 1}
    if not duplicate_ids:
        return []

    used_ids = set(global_counts.keys())
    change_summaries: List[Dict[str, Any]] = []
    ref_pat_tpl = r"\[\^{fid}\]"

    for ch in chapters:
        if not ch.footnotes:
            continue
        mapping: Dict[str, str] = {}

        for fn in ch.footnotes:
            old_id = str(fn.get("id", "")).strip()
            if not old_id or old_id not in duplicate_ids:
                continue
            if old_id in mapping:
                continue

            base = f"c{max(1, ch.number)}-{old_id}"
            candidate = base
            suffix = 2
            while candidate in used_ids:
                candidate = f"{base}-{suffix}"
                suffix += 1
            mapping[old_id] = candidate
            used_ids.add(candidate)

        if not mapping:
            continue

        # Rewrite chapter body refs.
        new_body = ch.body
        for old_id, new_id in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
            new_body = re.sub(
                ref_pat_tpl.format(fid=re.escape(old_id)),
                f"[^{new_id}]",
                new_body,
            )
        ch.body = new_body

        # Rewrite note IDs.
        for fn in ch.footnotes:
            fid = str(fn.get("id", "")).strip()
            if fid in mapping:
                fn["id"] = mapping[fid]

        change_summaries.append(
            {
                "chapter_number": ch.number,
                "chapter_title": ch.title,
                "rewritten_ids": mapping,
            }
        )

    return change_summaries


def _render_markdown_table(rows: List[List[str]]) -> str:
    """
    Render rows into a markdown table.
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    normalized_rows = []
    for row in rows:
        norm_row = [str(c).strip() for c in row] + [""] * (width - len(row))
        normalized_rows.append(norm_row)

    header = normalized_rows[0]
    separators = ["---"] * width
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    for row in normalized_rows[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


@dataclass
class StructuredTextLine:
    """Lightweight line object used by structured assembly helpers."""
    index: int
    raw: str

    @property
    def text(self) -> str:
        return self.raw.strip()

    @property
    def is_blank(self) -> bool:
        return not self.text


@dataclass
class StructuredTextBlock:
    """
    A contiguous block of lines with a coarse type.
    Used to avoid repeatedly reparsing flattened strings in table/list assembly.
    """
    kind: str  # blank | markdown_table | text
    start: int
    end: int
    lines: List[StructuredTextLine]

    def to_lines(self) -> List[str]:
        return [ln.raw for ln in self.lines]


@dataclass
class StructuredListEntry:
    """
    Reader-facing structured list row with source line provenance.
    `label` supports cases like TOC numbering or list item prefixes.
    """
    line_indexes: List[int]
    text: str
    page: str = ""
    label: str = ""
    kind: str = "entry"


def _to_structured_lines(text: str) -> List[StructuredTextLine]:
    return [StructuredTextLine(index=i, raw=ln) for i, ln in enumerate((text or "").split("\n"))]


def _segment_structured_text_blocks(text: str) -> List[StructuredTextBlock]:
    """
    Segment text into coarse blocks so structured assembly passes can operate on
    block types (tables vs prose) instead of regex over whole chapter strings.
    """
    lines = _to_structured_lines(text)
    if not lines:
        return []

    blocks: List[StructuredTextBlock] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.is_blank:
            start = i
            chunk: List[StructuredTextLine] = []
            while i < len(lines) and lines[i].is_blank:
                chunk.append(lines[i])
                i += 1
            blocks.append(StructuredTextBlock(kind="blank", start=start, end=i - 1, lines=chunk))
            continue

        if line.text.startswith("|"):
            start = i
            chunk = []
            while i < len(lines) and lines[i].text.startswith("|"):
                chunk.append(lines[i])
                i += 1
            blocks.append(StructuredTextBlock(kind="markdown_table", start=start, end=i - 1, lines=chunk))
            continue

        start = i
        chunk = []
        while i < len(lines) and (not lines[i].is_blank) and (not lines[i].text.startswith("|")):
            chunk.append(lines[i])
            i += 1
        blocks.append(StructuredTextBlock(kind="text", start=start, end=i - 1, lines=chunk))

    return blocks


def _parse_markdown_table_block(
    block: StructuredTextBlock,
) -> Optional[Tuple[str, str, List[str]]]:
    """
    Parse a markdown table block into (header_line, separator_line, body_rows).
    Returns None if the block is not a normal markdown table.
    """
    if block.kind != "markdown_table":
        return None
    rows = [ln.text for ln in block.lines if ln.text]
    if len(rows) < 2:
        return None
    header = rows[0]
    sep = rows[1]
    if not (header.startswith("|") and sep.startswith("|") and re.search(r"-{3,}", sep)):
        return None
    body_rows = [r for r in rows[2:] if r and r != header and r != sep]
    return header, sep, body_rows


def _is_markdown_table_separator_row(line: str) -> bool:
    s = (line or "").strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{3,}:?", c or "")) for c in cells)


def _looks_like_markdown_table_row_line(line: str) -> bool:
    s = (line or "").strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    if s.count("|") < 2:
        return False
    return True


def _markdown_table_cell_count(line: str) -> int:
    s = (line or "").strip()
    if not _looks_like_markdown_table_row_line(s):
        return 0
    return len([c for c in s.strip("|").split("|")])


def _render_markdown_table_separator(cols: int) -> str:
    cols = max(1, int(cols or 0))
    return "| " + " | ".join(["---"] * cols) + " |"


def _looks_like_table_header_restart_row(line: str, seen_headers: set) -> bool:
    s = (line or "").strip()
    if not _looks_like_markdown_table_row_line(s):
        return False
    if _is_markdown_table_separator_row(s):
        return False

    if s in seen_headers:
        return True

    cells = [c.strip() for c in s.strip("|").split("|")]
    if not cells or len(cells) > 6:
        return False
    if any(not c for c in cells):
        return False
    if sum(len(c) for c in cells) > 80:
        return False
    # Header rows are usually short labels ("Entry", "Page", "Illustration", ...).
    if any(len(c) > 32 for c in cells):
        return False
    if any(re.search(r"[.!?]{2,}", c) for c in cells):
        return False
    alpha_count = sum(1 for c in "".join(cells) if c.isalpha())
    return alpha_count >= 3


def _repair_markdown_table_header_restarts(text: str) -> str:
    """
    Repair malformed markdown table restarts where a header row is repeated but
    the required separator row is missing (common in split TOCs/list pages).

    Example:
      | Entry | Page |
      <blank>
      | CHAPTER V ... | 148 |

    becomes:
      | Entry | Page |
      | --- | --- |
      | CHAPTER V ... | 148 |
    """
    if not text or "|" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    seen_headers: set = set()

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        # Track well-formed table headers we've already seen in this text.
        if _looks_like_markdown_table_row_line(s):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _is_markdown_table_separator_row(lines[j]):
                seen_headers.add(s)

        if not _looks_like_markdown_table_row_line(s) or _is_markdown_table_separator_row(s):
            out.append(line)
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines):
            out.append(line)
            i += 1
            continue

        next_nonempty = lines[j].strip()
        if _is_markdown_table_separator_row(next_nonempty):
            out.append(line)
            i += 1
            continue

        if not _looks_like_markdown_table_row_line(next_nonempty):
            out.append(line)
            i += 1
            continue

        cur_cols = _markdown_table_cell_count(s)
        next_cols = _markdown_table_cell_count(next_nonempty)
        if cur_cols <= 0 or next_cols <= 0 or cur_cols != next_cols:
            out.append(line)
            i += 1
            continue

        if not _looks_like_table_header_restart_row(s, seen_headers):
            out.append(line)
            i += 1
            continue

        out.append(line)
        out.append(_render_markdown_table_separator(cur_cols))
        seen_headers.add(s)
        # Drop blank lines between the repaired header and the next row so
        # markdown parsers keep the table contiguous.
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

    return "\n".join(out)


def _render_structured_blocks(blocks: List[StructuredTextBlock]) -> str:
    out_lines: List[str] = []
    for block in blocks:
        out_lines.extend(block.to_lines())
    return "\n".join(out_lines)


def _collect_wrapped_entry_page_rows(
    source_lines: List[StructuredTextLine],
    *,
    page_tail_pattern: re.Pattern,
    ignore_line: Optional[Callable[[str], bool]] = None,
    normalize_page: Optional[Callable[[str], str]] = None,
    allow_label_rows: bool = False,
    label_predicate: Optional[Callable[[str], bool]] = None,
) -> List[StructuredListEntry]:
    """
    Deterministically group wrapped OCR list entries ending with a page token.
    This helper preserves source line indexes and does not rewrite entry text.
    """
    entries: List[StructuredListEntry] = []
    pending_lines: List[StructuredTextLine] = []

    def _joined_pending() -> str:
        if not pending_lines:
            return ""
        parts: List[str] = []
        for ln in pending_lines:
            s = ln.text
            if not s:
                continue
            if not parts:
                parts.append(s)
            elif parts[-1].endswith("-"):
                parts[-1] = parts[-1] + s
            else:
                parts.append(s)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _flush() -> bool:
        nonlocal pending_lines
        joined = _joined_pending()
        if not joined:
            pending_lines = []
            return False

        m = page_tail_pattern.match(joined)
        if m:
            entry_text = re.sub(r"(?:\s*[.·•])+\s*$", "", (m.group(1) or "")).strip()
            page = str(m.group(2) or "").strip()
            if normalize_page:
                page = normalize_page(page)
            if entry_text:
                entries.append(
                    StructuredListEntry(
                        line_indexes=[ln.index for ln in pending_lines],
                        text=entry_text,
                        page=page,
                    )
                )
                pending_lines = []
                return True

        if label_predicate and label_predicate(joined):
            entries.append(
                StructuredListEntry(
                    line_indexes=[ln.index for ln in pending_lines],
                    text=joined,
                    page="",
                    kind="label",
                )
            )
            pending_lines = []
            return True
        if allow_label_rows and joined.endswith(":"):
            entries.append(
                StructuredListEntry(
                    line_indexes=[ln.index for ln in pending_lines],
                    text=joined,
                    page="",
                    kind="label",
                )
            )
            pending_lines = []
            return True
        return False

    for ln in source_lines:
        s = ln.text
        if not s:
            _flush()
            continue
        if ignore_line and ignore_line(s):
            continue
        pending_lines.append(ln)
        _flush()

    _flush()
    return entries


# ─── Book Assembler ──────────────────────────────────────────────────────────

class BookAssembler:
    """
    Pass 2/3: Assemble extracted pages into a coherent book.

    Handles:
    - Chapter boundary detection
    - Cross-page paragraph merging
    - Footnote collection and placement
    - Page type classification
    """

    def __init__(
        self,
        gemini_client=None,
        claude_client=None,
        shadow_planner_client=None,
        strict_chapter_detection: bool = True,
        llm_assembly_ops: bool = True,
        reader_assembly_mode: str = "deterministic",
        archival_mode: bool = False,
        preserve_nonchapter_headings: bool = True,
        aggressive_publish_normalization: bool = True,
        llm_planner_routing: Optional[Dict[str, str]] = None,
        llm_planner_models: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ):
        self.gemini = gemini_client
        self.claude = claude_client
        self.shadow_planner = shadow_planner_client
        self.strict_chapter_detection = strict_chapter_detection
        self.llm_assembly_ops = llm_assembly_ops
        self.reader_assembly_mode = (reader_assembly_mode or "deterministic").strip().lower()
        self.archival_mode = archival_mode
        self.preserve_nonchapter_headings = preserve_nonchapter_headings
        self.aggressive_publish_normalization = aggressive_publish_normalization
        self.llm_planner_routing = {
            str(k).strip(): str(v).strip().lower()
            for k, v in (llm_planner_routing or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.llm_planner_models = {
            str(k).strip(): str(v).strip()
            for k, v in (llm_planner_models or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.verbose = verbose
        self.assembly_ops: List[Dict[str, Any]] = []
        # Shadow/challenger calls are useful for audits, but can be expensive and
        # rate-limit prone. Throttle low-value calls and fail closed.
        self._shadow_calls_run = 0
        self._shadow_skips_run = 0
        self._shadow_last_call_ts = 0.0
        self._shadow_consecutive_errors = 0
        self._shadow_circuit_open_until = 0.0
        self._shadow_circuit_notice_emitted = False
        self._shadow_max_calls_per_run = 60
        self._shadow_min_interval_sec = 1.0
        self._shadow_low_priority_sample_rate = 0.15
        self._shadow_error_threshold = 3
        self._shadow_cooldown_sec = 300.0

    def _log(self, message: str):
        if self.verbose:
            print(f"[Assembly] {message}")

    def _planner_backend_for_op(self, op_name: str) -> Tuple[Optional[Any], str, Optional[str]]:
        """
        Select planner backend for a specific assembly op.
        Routing values: auto | gemini | claude
        """
        op_key = str(op_name or "").strip()
        route = (
            self.llm_planner_routing.get(op_key)
            or self.llm_planner_routing.get("*")
            or self.llm_planner_routing.get("default")
            or "auto"
        )
        model_override = (
            self.llm_planner_models.get(op_key)
            or self.llm_planner_models.get("*")
            or self.llm_planner_models.get("default")
        )

        if route == "claude":
            if self.claude:
                return self.claude, "claude", model_override
            if self.gemini:
                return self.gemini, "gemini", None
            return None, "none", None
        if route == "gemini":
            if self.gemini:
                return self.gemini, "gemini", None
            if self.claude:
                return self.claude, "claude", model_override
            return None, "none", None

        # auto: prefer Gemini for speed/cost, fall back to Claude if needed
        if self.gemini:
            return self.gemini, "gemini", None
        if self.claude:
            return self.claude, "claude", model_override
        return None, "none", None

    def _planner_json_generate(
        self,
        *,
        op_name: str,
        prompt: str,
        thinking: Optional[Any] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float, str]:
        """
        Provider-agnostic JSON planner call.
        Returns (json_dict_or_none, duration_ms, provider_name).
        """
        client, provider, model_override = self._planner_backend_for_op(op_name)
        if not client:
            return None, 0.0, "none"
        t0 = time.time()
        if provider == "gemini":
            resp = client.generate(
                prompt=prompt,
                thinking=thinking if thinking is not None else getattr(client, "default_thinking", None),
                json_mode=True,
            )
        else:
            resp = client.generate(
                prompt=prompt,
                model=model_override,
                json_mode=True,
            )
        elapsed_ms = (time.time() - t0) * 1000.0
        if not getattr(resp, "success", False):
            err = str(getattr(resp, "error", "") or "failed")
            raise RuntimeError(f"{provider} planner failed: {err}")
        data = getattr(resp, "json_data", None) or self._parse_json_response(getattr(resp, "text", ""))
        return (data if isinstance(data, dict) else None), elapsed_ms, provider

    def assemble(
        self,
        extractions: List[PageExtraction],
        title: str = "Untitled",
        author: str = "Unknown",
    ) -> AssembledBook:
        """
        Assemble extracted pages into a book.

        Args:
            extractions: List of PageExtraction results (in page order)
            title: Book title
            author: Book author

        Returns:
            AssembledBook with chapters and master markdown
        """
        self.assembly_ops = []
        self._log(f"Assembling {len(extractions)} pages...")

        # Step 1: Classify all pages
        page_types = {}
        blank_pages = []
        warnings = []
        review_pages = []

        for ext in extractions:
            ptype = detect_page_type(
                ext,
                strict_chapter_detection=self.strict_chapter_detection
            )
            page_types[ext.page_number] = ptype

            if ptype == "blank":
                blank_pages.append(ext.page_number)
            if ext.warnings:
                review_pages.append(ext.page_number)
                for w in ext.warnings:
                    warnings.append(f"Page {ext.page_number}: {w}")

        self._log(f"Page types: {len(blank_pages)} blank, "
                  f"{sum(1 for v in page_types.values() if v == 'chapter_start')} chapter starts")

        # Step 2: Detect running headers/footers from frequent top/bottom lines
        margin_rules = self._detect_running_margins(extractions, page_types)

        # Step 3: Cross-page word joining via LLM decision map
        join_overrides = set()
        if self.gemini or self.claude:
            join_overrides = self._join_cross_page_words(extractions, page_types)

        # Step 4: Group pages into chapters
        chapters = self._group_into_chapters(
            extractions,
            page_types,
            join_overrides=join_overrides,
            margin_rules=margin_rules,
        )

        # Step 5: Process each chapter
        self._log(f"Processing {len(chapters)} chapters...")
        for chapter_idx, chapter in enumerate(chapters, start=1):
            chapter_label = chapter.title.strip().replace("\n", " ")
            if len(chapter_label) > 80:
                chapter_label = chapter_label[:77] + "..."
            self._log(f"Chapter {chapter_idx}/{len(chapters)}: {chapter_label}")
            # Clean and format body text
            chapter.body = format_content(
                chapter.body,
                preserve_nonchapter_headings=self.preserve_nonchapter_headings,
            )
            if not self.archival_mode:
                chapter.body, _ = self._strip_inline_signature_prefixes(chapter.body, "text")
                chapter.body = self._strip_leading_duplicate_chapter_title(chapter.body, chapter.title)
                chapter.body = self._strip_leading_chapter_running_header(chapter.body, chapter.title)
                chapter.body = self._strip_leading_split_chapter_label_title(chapter.body, chapter.title)
                chapter.body = self._strip_leading_chapter_opening_wrapper_stack(chapter.body, chapter.title)
                chapter.body = self._strip_leading_composite_title_component_lines(chapter.body, chapter.title)
            chapter.has_poetry = any(
                ext.has_poetry for ext in extractions
                if ext.page_number in chapter.pages
            )
            chapter_kind = _chapter_title_kind(chapter.title)
            chapter_is_frontmatter = _normalize_simple(chapter.title) == "front matter"
            chapter_is_frontmatter_like = chapter_kind in {"frontmatter", "preface"}
            if self.aggressive_publish_normalization:
                chapter.body = normalize_publish_markdown(
                    chapter.body,
                    allow_blockquotes=not chapter_is_frontmatter_like,
                    prefer_poetry=chapter.has_poetry,
                )
            else:
                chapter.body = chapter.body.replace('\r\n', '\n').replace('\r', '\n')
            if not self.archival_mode:
                chapter.body, _ = self._strip_inline_signature_prefixes(chapter.body, "text")

            if (not self.archival_mode) and chapter_kind not in {"frontmatter", "preface", "index", "appendix"}:
                chapter.body, tagged_marginalia = _tag_reader_marginalia_lines(chapter.body)
                if tagged_marginalia:
                    self._record_op(
                        "reader_marginalia_tagging",
                        {
                            "chapter_title": chapter.title,
                            "count": len(tagged_marginalia),
                            "samples": tagged_marginalia[:10],
                        },
                    )

            if chapter_is_frontmatter_like and not self.archival_mode:
                self._log(f"  LLM frontmatter structuring: {chapter_label}")
                chapter.body = self._apply_llm_frontmatter_structure_ops(chapter.body, chapter.title)
                chapter.body = self._merge_adjacent_frontmatter_tables(chapter.body)
                chapter.body = self._promote_reader_frontmatter_section_headings(chapter.body)
                chapter.body = self._merge_repeated_frontmatter_headed_tables(chapter.body)
                chapter.body = self._normalize_frontmatter_authorities_tables(chapter.body)
                chapter.body = self._normalize_reader_frontmatter_body(chapter.body, chapter.title)
            elif not self.archival_mode:
                self._log(f"  LLM chapter line roles: {chapter_label}")
                chapter.body = self._apply_llm_chapter_line_role_ops(chapter.body, chapter.title)
                chapter.body = self._promote_reader_body_heading_blocks(chapter.body, chapter.title)
                chapter.body = self._strip_leading_library_artifact_headings(chapter.body)
                chapter.body = self._strip_reader_library_artifact_blocks(chapter.body, chapter.title)
                self._log(f"  LLM backmatter structuring (if applicable): {chapter_label}")
                chapter.body = self._apply_llm_backmatter_structure_ops(chapter.body, chapter.title)
                chapter.body = self._normalize_reader_backmatter_body(chapter.body, chapter.title)
            if not self.archival_mode:
                _resolvable_markers = _collect_resolvable_letter_markers(chapter.footnotes)
                chapter.body = _normalize_superscript_letter_footnote_markers(
                    chapter.body,
                    allowed_letters=_resolvable_markers,
                )
                chapter.body, chapter.footnotes = _reconcile_chapter_footnote_references(
                    chapter.body,
                    chapter.footnotes,
                )
                _marker_before_body = chapter.body
                _marker_before_footnotes = list(chapter.footnotes)
                _marker_before_counts = _chapter_footnote_integrity_counts(chapter.body, chapter.footnotes)
                marker_inserted_body = self._apply_llm_chapter_footnote_marker_insertions(
                    chapter.body,
                    chapter.footnotes,
                    chapter.title,
                )
                marker_inserted_body, marker_inserted_footnotes = _reconcile_chapter_footnote_references(
                    marker_inserted_body,
                    chapter.footnotes,
                )
                _marker_after_counts = _chapter_footnote_integrity_counts(marker_inserted_body, marker_inserted_footnotes)
                marker_worsened = (
                    _marker_after_counts["orphaned"] > _marker_before_counts["orphaned"]
                    or _marker_after_counts["mismatches"] > _marker_before_counts["mismatches"]
                )
                if marker_worsened:
                    self._record_op(
                        "llm_footnote_marker_insert_revert",
                        {
                            "chapter_title": chapter.title,
                            "before": _marker_before_counts,
                            "after": _marker_after_counts,
                            "reason": "chapter footnote integrity worsened",
                        },
                    )
                    chapter.body = _marker_before_body
                    chapter.footnotes = _marker_before_footnotes
                else:
                    chapter.body = marker_inserted_body
                    chapter.footnotes = marker_inserted_footnotes
                self._log(f"  LLM footnote linking (if ambiguous): {chapter_label}")
                chapter.body, chapter.footnotes = self._apply_llm_chapter_footnote_link_ops(
                    chapter.body,
                    chapter.footnotes,
                    chapter.title,
                )
                chapter.body, chapter.footnotes = _reconcile_chapter_footnote_references(
                    chapter.body,
                    chapter.footnotes,
                )
                chapter.footnotes = self._apply_llm_chapter_footnote_def_cleanup_ops(
                    chapter.body,
                    chapter.footnotes,
                    chapter.title,
                )
                chapter.footnotes = self._merge_split_unreferenced_page_scoped_footnote_defs(
                    chapter.body,
                    chapter.footnotes,
                    chapter.title,
                )
                _res_before_body = chapter.body
                _res_before_footnotes = list(chapter.footnotes)
                _res_before_counts = _chapter_footnote_integrity_counts(chapter.body, chapter.footnotes)
                residual_body, residual_footnotes = self._apply_llm_chapter_footnote_residual_ops(
                    chapter.body,
                    chapter.footnotes,
                    chapter.title,
                )
                residual_body, residual_footnotes = _reconcile_chapter_footnote_references(
                    residual_body,
                    residual_footnotes,
                )
                residual_footnotes = self._merge_unreferenced_page_scoped_sibling_defs(
                    residual_body,
                    residual_footnotes,
                    chapter.title,
                )
                _res_after_counts = _chapter_footnote_integrity_counts(residual_body, residual_footnotes)
                residual_worsened = (
                    _res_after_counts["orphaned"] > _res_before_counts["orphaned"]
                    or _res_after_counts["mismatches"] > _res_before_counts["mismatches"]
                )
                if residual_worsened:
                    self._record_op(
                        "llm_footnote_residual_cleanup_revert",
                        {
                            "chapter_title": chapter.title,
                            "before": _res_before_counts,
                            "after": _res_after_counts,
                            "reason": "chapter footnote integrity worsened",
                        },
                    )
                    chapter.body = _res_before_body
                    chapter.footnotes = _res_before_footnotes
                else:
                    chapter.body = residual_body
                    chapter.footnotes = residual_footnotes
                chapter.body = _heal_paragraph_breaks_after_footnote_refs(chapter.body)

            # De-duplicate footnotes at chapter level (Fix #5)
            chapter.footnotes = _dedup_chapter_footnotes(chapter.footnotes)
            chapter.footnotes = sorted(chapter.footnotes, key=_footnote_sort_key)

            # Count words
            chapter.word_count = len(chapter.body.split()) if chapter.body.strip() else 0

            # Check for Indic script
            chapter.has_indic_script = any(
                ext.has_indic_script for ext in extractions
                if ext.page_number in chapter.pages
            )

        if not self.archival_mode:
            id_changes = _uniquify_footnote_ids_across_chapters(chapters)
            if id_changes:
                self._record_op(
                    "global_footnote_id_uniquify",
                    {
                        "chapters_changed": len(id_changes),
                        "sample": id_changes[:10],
                    },
                )

        # Step 6: Build master markdown
        master_parts = [f"# {title}", ""]
        self._log("Planning reader heading layout...")
        heading_plan = self._llm_plan_reader_heading_layout(chapters)
        self._log("Reader heading layout planned.")

        for idx, chapter in enumerate(chapters):
            plan = heading_plan.get(idx, {})
            emit_if_empty = bool(plan.get("emit_if_empty", _should_emit_structural_empty_chapter(chapter.title)))
            heading_level = int(plan.get("heading_level", _default_heading_level_for_title(chapter.title)))
            if (not chapter.is_empty) or emit_if_empty:
                master_parts.append(
                    format_chapter(
                        chapter,
                        emit_notes_section=not self.archival_mode,
                        heading_level=heading_level,
                    )
                )
                master_parts.append("")

        master_markdown = '\n'.join(master_parts).strip() + '\n'

        # Calculate stats
        total_words = sum(ch.word_count for ch in chapters)
        # avg_fidelity is a heuristic confidence score, NOT verified source fidelity
        avg_fidelity = 0.0
        fidelity_pages = [e for e in extractions if e.success and not e.is_blank]
        if fidelity_pages:
            avg_fidelity = sum(e.fidelity for e in fidelity_pages) / len(fidelity_pages)

        self._log(f"Assembly complete: {len(chapters)} chapters, "
                  f"{total_words:,} words, {len(blank_pages)} blank pages")

        return AssembledBook(
            title=title,
            author=author,
            chapters=chapters,
            total_pages=len(extractions),
            total_words=total_words,
            blank_pages=blank_pages,
            warnings=warnings,
            pages_needing_review=review_pages,
            master_markdown=master_markdown,
            markdown=master_markdown,
            avg_fidelity=avg_fidelity,
            assembly_ops=self.assembly_ops[:],
        )

    def _normalize_margin_line(self, line: str) -> str:
        """Normalize a line for running header/footer detection."""
        norm = line.strip()
        if not norm:
            return ""
        norm = re.sub(r'\s+', ' ', norm)
        norm = re.sub(r'\b\d+\b', '#', norm)
        norm = re.sub(r'\b[ivxlcdm]+\b', '#', norm, flags=re.IGNORECASE)
        norm = re.sub(r'[_\-–—]+', ' ', norm)
        norm = re.sub(r'[^a-z0-9\s#]', ' ', norm.lower())
        norm = re.sub(r'\s+', ' ', norm)
        return norm.strip()

    def _is_probable_margin_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) > 80:
            return False
        if _looks_like_footnote_definition_line(stripped):
            return False
        return len(stripped.split()) <= 12

    def _is_page_number_line(self, line: str) -> bool:
        """Detect standalone page number/footer markers."""
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r'^\[?\(?\d{1,4}\)?\]?[.*]?$', stripped):
            return True
        roman = stripped.rstrip('.').upper()
        if ROMAN_NUMERAL.match(roman):
            return True
        return False

    def _strip_top_page_header_stack(self, text: str, page_type: str) -> str:
        """
        Remove common continued-page top stacks like:
        - "20" + "INTRODUCTION."
        - "INTRODUCTION." + "17"
        - "INTRODUCTION. 19"
        """
        if page_type not in ("text", "frontmatter", "backmatter", "toc"):
            return text

        lines = text.split('\n')
        non_empty_idxs = [i for i, line in enumerate(lines) if line.strip()]
        if len(non_empty_idxs) < 2:
            return text

        top_idxs = non_empty_idxs[:3]
        top_lines = [lines[i].strip() for i in top_idxs]

        def _is_header_label(line: str) -> bool:
            s = line.strip().strip('*').strip()
            return bool(s and CHAPTER_KEYWORDS.match(s))

        # line 1: combined header + page number suffix
        first = top_lines[0]
        if _looks_like_running_header_with_page_number(first) or _looks_like_page_header_combo(first):
            lines[top_idxs[0]] = ""
            return '\n'.join(lines)

        # line1 page number, line2 repeated header label
        if self._is_page_number_line(first) and _is_header_label(top_lines[1]):
            lines[top_idxs[0]] = ""
            lines[top_idxs[1]] = ""
            return '\n'.join(lines)

        # line1 repeated header label, line2 page number
        if _is_header_label(first) and self._is_page_number_line(top_lines[1]):
            lines[top_idxs[0]] = ""
            lines[top_idxs[1]] = ""
            return '\n'.join(lines)

        # line1 page number, line2 combined header/page number
        if self._is_page_number_line(first) and _looks_like_page_header_combo(top_lines[1]):
            lines[top_idxs[0]] = ""
            lines[top_idxs[1]] = ""
            return '\n'.join(lines)

        return text

    def _detect_running_margins(
        self,
        extractions: List[PageExtraction],
        page_types: Dict[int, str],
    ) -> Dict[str, set]:
        """
        Detect repetitive running headers/footers from top/bottom lines on text pages.
        """
        text_pages = [
            ext for ext in extractions
            # Exclude frontmatter from frequency-based margin detection: repeated title-page
            # lines (e.g., book title on half-title + title page) are legitimate content.
            if page_types.get(ext.page_number) in ("text", "chapter_start", "backmatter")
            and ext.text.strip()
        ]
        if len(text_pages) < 2:
            return {"headers": set(), "footers": set()}

        header_counts: Counter = Counter()
        footer_counts: Counter = Counter()

        for ext in text_pages:
            non_empty_lines = [line for line in ext.text.split('\n') if line.strip()]
            if not non_empty_lines:
                continue

            for line in non_empty_lines[:2]:
                if self._is_probable_margin_line(line):
                    norm = self._normalize_margin_line(line)
                    if norm:
                        header_counts[norm] += 1

            for line in non_empty_lines[-2:]:
                if self._is_probable_margin_line(line):
                    norm = self._normalize_margin_line(line)
                    if norm:
                        footer_counts[norm] += 1

        # Small chunks often have alternating recto/verso running headers that appear
        # only twice, so use a lower threshold there.
        min_hits = 2 if len(text_pages) <= 30 else max(3, int(len(text_pages) * 0.12))
        headers = {k for k, v in header_counts.items() if v >= min_hits}
        footers = {k for k, v in footer_counts.items() if v >= min_hits}

        if headers or footers:
            self._log(
                f"Detected running margins: {len(headers)} headers, {len(footers)} footers "
                f"(threshold={min_hits})"
            )

        return {"headers": headers, "footers": footers}

    def _strip_detected_margins(self, text: str, margin_rules: Dict[str, set]) -> str:
        """Strip detected running headers/footers from a page's body text."""
        headers = margin_rules.get("headers", set())
        footers = margin_rules.get("footers", set())
        if not text:
            return text

        lines = text.split('\n')
        if not lines:
            return text

        non_empty_top = [i for i, line in enumerate(lines) if line.strip()][:2]
        for idx in non_empty_top:
            stripped = lines[idx].strip()
            if _looks_like_footnote_definition_line(stripped):
                continue
            if self._is_page_number_line(lines[idx]):
                lines[idx] = ""
                continue
            norm = self._normalize_margin_line(lines[idx])
            if norm and norm != "#" and norm in headers:
                lines[idx] = ""

        non_empty_bottom = [i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()][:2]
        for idx in non_empty_bottom:
            stripped = lines[idx].strip()
            if _looks_like_footnote_definition_line(stripped):
                continue
            if self._is_page_number_line(lines[idx]):
                lines[idx] = ""
                continue
            norm = self._normalize_margin_line(lines[idx])
            if norm and norm != "#" and norm in footers:
                lines[idx] = ""

        return '\n'.join(lines)

    def _strip_chapter_heading_lines(
        self,
        body_text: str,
        chapter_title: str,
    ) -> str:
        """
        Remove leading title lines from chapter-start page body to avoid duplication.
        """
        if not body_text.strip():
            return body_text

        title_norm = _normalize_simple(chapter_title)
        lines = body_text.split('\n')
        i = 0
        removed = 0

        while i < len(lines) and removed < 5:
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                removed += 1
                continue

            norm = _normalize_simple(stripped)
            looks_like_chapter_label = bool(CHAPTER_KEYWORDS.match(stripped))
            in_title = bool(norm and title_norm and (norm in title_norm or title_norm in norm))

            # Only remove lines that are clearly the chapter label/title itself.
            # Do not remove arbitrary short ALL-CAPS lines here; chapter-opening
            # pages often contain a real subheading immediately after the title.
            if looks_like_chapter_label or in_title:
                i += 1
                removed += 1
                continue
            break

        cleaned = '\n'.join(lines[i:]).lstrip('\n')
        return cleaned

    def _strip_repeated_continuation_chapter_header(
        self,
        body_text: str,
        chapter_title: str,
    ) -> str:
        """
        Remove a repeated chapter-title running header at the top of a continuation page
        before boundary merge (e.g. page starts with "INTRODUCTION" then continued prose).
        """
        if not body_text.strip() or not chapter_title.strip():
            return body_text
        lines = body_text.split('\n')
        title_norm = _normalize_simple(chapter_title.rstrip(".:"))
        first_idx = None
        for idx, line in enumerate(lines):
            if line.strip():
                first_idx = idx
                break
        if first_idx is None:
            return body_text
        first = lines[first_idx].strip()
        first_norm = _normalize_simple(first.rstrip(".:"))
        prefix_match = bool(
            first_norm
            and title_norm
            and first_norm in title_norm
            and len(first_norm) >= 8
        )
        if not first_norm or not (first_norm == title_norm or prefix_match):
            return body_text
        # Require the title line to be visually isolated (blank after) to avoid touching prose.
        if first_idx + 1 < len(lines) and lines[first_idx + 1].strip():
            return body_text
        lines[first_idx] = ""
        return '\n'.join(lines)

    def _strip_contextual_page_top_stack(
        self,
        text: str,
        *,
        chapter_title: str,
        page_type: str,
        is_chapter_start: bool = False,
    ) -> str:
        """
        Context-aware page-top stack stripper used during chapter grouping.

        This runs before footnote extraction and before chapter text is flattened,
        so it can remove obvious page-top wrappers while preserving real chapter
        opener lines and frontmatter section labels.

        It behaves like a small state machine: repeatedly inspect the first
        non-empty line(s), classify them, peel only safe wrapper/header lines,
        and stop as soon as likely content is reached.
        """
        if self.archival_mode or not text.strip():
            return text
        if page_type not in ("text", "chapter_start", "backmatter", "frontmatter"):
            return text
        if page_type == "toc":
            return text

        lines = text.split("\n")
        title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))

        def _non_empty_idxs() -> List[int]:
            return [i for i, ln in enumerate(lines) if ln.strip()]

        def _line_kind(s: str) -> str:
            ss = s.strip()
            if not ss:
                return "blank"
            if _looks_like_footnote_definition_line(ss):
                return "footnote"
            if self._is_page_number_line(ss):
                return "page_no"
            if re.fullmatch(r'[\-—–_*=·•♦◈\s]{3,}', ss):
                return "divider"
            # Library checkout/stamp artifacts are header-stack style in this pass.
            # Avoid matching long prose lines that happen to contain tokens like "checked".
            if len(ss) <= 90 and len(ss.split()) <= 10 and self._is_library_artifact_line(ss):
                return "library"
            if re.match(r'(?i)^univ\.?\s+of$', ss) or re.match(r'(?i)^california\.?$', ss):
                return "library_frag"
            if ss.startswith(("#", "|", ">", "[^")):
                return "struct"
            if _looks_like_standalone_chapter_heading_line(ss) or _is_standalone_ordinal_marker(ss):
                return "heading"
            if title_norm and _normalize_simple(ss.rstrip(".:")) == title_norm:
                return "title_dup"
            alpha = [c for c in ss if c.isalpha()]
            upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
            if len(ss) <= 90 and len(ss.split()) <= 8 and upper_ratio >= 0.75:
                return "short_upper"
            if re.match(r'^[a-z]', ss):
                return "prose"
            return "other"

        changed = False
        removed = 0
        # Bounded peeling loop. We inspect the top repeatedly after each removal.
        while removed < 8:
            non_empty = _non_empty_idxs()
            if not non_empty:
                break
            i0 = non_empty[0]
            if i0 > 20:
                break
            s0 = lines[i0].strip()
            k0 = _line_kind(s0)

            # Always peel obvious wrappers at the very top.
            if k0 in {"page_no", "divider", "library", "library_frag"}:
                lines[i0] = ""
                changed = True
                removed += 1
                continue

            # Duplicate exact chapter title at page top is a wrapper in reader mode.
            if k0 == "title_dup":
                lines[i0] = ""
                changed = True
                removed += 1
                continue

            # Lookahead for stacked wrappers/header patterns.
            i1 = non_empty[1] if len(non_empty) >= 2 else None
            s1 = lines[i1].strip() if i1 is not None else ""
            k1 = _line_kind(s1) if s1 else "blank"
            i2 = non_empty[2] if len(non_empty) >= 3 else None
            s2 = lines[i2].strip() if i2 is not None else ""
            k2 = _line_kind(s2) if s2 else "blank"

            if is_chapter_start or page_type == "chapter_start":
                # Chapter opener pages often have stacks like:
                # running header -> divider -> duplicate title -> ordinal/subheading.
                if k0 == "short_upper" and k1 in {"heading", "title_dup", "short_upper"}:
                    # Prefer preserving explicit heading/title line and peel the wrapper above it.
                    lines[i0] = ""
                    changed = True
                    removed += 1
                    continue
                if (
                    k0 == "short_upper"
                    and k1 == "divider"
                    and k2 in {"heading", "title_dup", "short_upper"}
                ):
                    lines[i0] = ""
                    changed = True
                    removed += 1
                    continue
                # Library stack fragments before a real heading chain.
                if k0 in {"other", "short_upper"} and k1 in {"library", "library_frag"}:
                    lines[i0] = ""
                    changed = True
                    removed += 1
                    continue
                # Once we reach actual heading/ordinal on chapter start, stop.
                if k0 in {"heading", "other"}:
                    break
                break

            # Continuation/body pages: peel leaked running header labels at top.
            if k0 == "short_upper":
                # Strong match to current chapter title fragment or repeated header style.
                n0 = _normalize_simple(s0.rstrip(".:"))
                title_related = bool(title_norm and n0 and (n0 == title_norm or n0 in title_norm or title_norm in n0))
                next_looks_content = (k1 in {"prose", "other", "heading", "title_dup"}) or (
                    s1 and re.match(r'^[a-z0-9"\'(\[]', s1)
                )
                if title_related or (next_looks_content and k1 not in {"footnote", "struct"}):
                    lines[i0] = ""
                    changed = True
                    removed += 1
                    # Also peel an immediate page number after the header label.
                    if i1 is not None and k1 == "page_no":
                        lines[i1] = ""
                        removed += 1
                    continue

            # Special continuation stack: header + page no / page no + header
            if k0 == "page_no" and k1 == "short_upper":
                lines[i0] = ""
                lines[i1] = ""
                changed = True
                removed += 2
                continue
            if k0 == "short_upper" and k1 == "page_no":
                lines[i0] = ""
                lines[i1] = ""
                changed = True
                removed += 2
                continue

            # If the third line is clearly prose and first line is short upper header-ish,
            # allow peeling the first line even when a blank/divider sits between them.
            if (
                k0 == "short_upper"
                and k1 in {"blank", "divider"}  # `blank` won't happen in non_empty list, keep for readability
                and i2 is not None
                and (k2 == "prose" or (s2 and re.match(r'^[a-z]', s2)))
            ):
                lines[i0] = ""
                changed = True
                removed += 1
                continue

            break

        return "\n".join(lines) if changed else text

    def _should_promote_text_page_to_chapter_start(self, ext: PageExtraction) -> bool:
        """
        Rescue heuristic for text pages that actually contain a chapter-opener stack.
        Used when page-type detection misses openers (common in historical scans with
        library/header wrappers above the real heading).
        """
        heading = (getattr(ext, "chapter_heading", "") or "").strip()
        raw_text = (getattr(ext, "text", "") or "")
        if not heading or not raw_text.strip():
            return False

        heading_norm = _normalize_simple(heading)
        heading_has_key = bool(re.search(r'(?i)\b(chapter|book|part|introduction|preface|conclusion|appendix|index)\b', heading))
        if not heading_has_key and len(heading_norm) < 20:
            return False

        non_empty: List[str] = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
        if not non_empty:
            return False

        visible: List[str] = []
        for s in non_empty[:16]:
            if self._is_page_number_line(s):
                continue
            if re.fullmatch(r'[\-—–_*=·•♦◈\s]{3,}', s):
                continue
            if len(s) <= 90 and len(s.split()) <= 10 and self._is_library_artifact_line(s):
                continue
            if re.match(r'(?i)^univ\.?\s+of$', s) or re.match(r'(?i)^california\.?$', s):
                continue
            visible.append(s)
        if len(visible) < 2:
            return False

        pre_prose: List[str] = []
        for s in visible:
            if re.match(r'^[a-z]', s):
                break
            # Long sentence-like prose lines can start uppercase; stop on those too.
            if len(s) > 90 or (len(s.split()) > 12 and re.search(r'[a-z].*[a-z].*[a-z]', s)):
                break
            pre_prose.append(s)
            if len(pre_prose) >= 8:
                break

        if len(pre_prose) < 2:
            return False

        def _is_headingish(line: str) -> bool:
            if _looks_like_standalone_chapter_heading_line(line) or _is_standalone_ordinal_marker(line):
                return True
            if re.search(r'(?i)\b(chapter|book|part|introduction|preface|conclusion|appendix|index)\b', line):
                return True
            alpha = [c for c in line if c.isalpha()]
            if alpha:
                upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
                if len(line) <= 100 and len(line.split()) <= 10 and upper_ratio >= 0.65:
                    return True
            return False

        headingish_count = sum(1 for s in pre_prose if _is_headingish(s))
        if headingish_count < 2:
            return False

        # Avoid promoting plain running-header pages: require a line matching/extending
        # the extracted heading signal (or a strong heading keyword in the stack).
        if heading_norm:
            for s in pre_prose:
                sn = _normalize_simple(s.rstrip(".:"))
                if sn and (sn == heading_norm or sn in heading_norm or heading_norm in sn):
                    return True
        return any(re.search(r'(?i)\b(chapter|book|part|introduction|preface|conclusion|appendix|index)\b', s) for s in pre_prose)

    def _strip_leading_duplicate_chapter_title(self, body_text: str, chapter_title: str) -> str:
        """
        Remove a leading body line/block that repeats the chapter title.
        Handles wrapped title duplicates on chapter opener pages while preserving
        subsequent subheadings.
        """
        if not body_text.strip() or not chapter_title.strip():
            return body_text
        lines = body_text.split('\n')
        title_norm = _normalize_simple(chapter_title.rstrip(".:"))
        if not title_norm:
            return body_text

        for idx, line in enumerate(lines[:20]):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return body_text
            first_norm = _normalize_simple(stripped.rstrip(".:"))
            if first_norm == title_norm:
                lines[idx] = ""
                return '\n'.join(lines)

            # Wrapped duplicate title block (common in scans):
            # e.g. chapter title already emitted as markdown heading, followed by
            # 2-4 centered ALL-CAPS lines that repeat the same title text.
            if len(stripped) > 120 or len(stripped.split()) > 14:
                return body_text
            if self._is_page_number_line(stripped) or self._is_library_artifact_line(stripped):
                return body_text

            block_idxs: List[int] = []
            block_lines: List[str] = []
            j = idx
            while j < min(len(lines), idx + 8):
                s = lines[j].strip()
                if not s:
                    break
                if s.startswith(("#", "|", ">", "[^")):
                    break
                if _looks_like_footnote_definition_line(s):
                    break
                if self._is_page_number_line(s) or self._is_library_artifact_line(s):
                    break
                if len(s) > 120 or len(s.split()) > 14:
                    break
                block_idxs.append(j)
                block_lines.append(s)
                j += 1
                if len(block_lines) >= 5:
                    break

            if not block_lines:
                return body_text

            joined = " ".join(block_lines)
            joined_norm = _normalize_simple(joined.rstrip(".:"))
            if not joined_norm:
                return body_text

            alpha = [c for c in joined if c.isalpha()]
            upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
            headingish_block = (
                bool(CHAPTER_KEYWORDS.match(block_lines[0]))
                or upper_ratio >= 0.55
                or all(len(bl.split()) <= 8 for bl in block_lines)
            )
            if not headingish_block:
                return body_text

            # Accept exact match or a substantial title fragment (subtitle-only duplicate).
            substantial_fragment = (
                joined_norm in title_norm and
                len(joined_norm) >= 16 and
                len(joined_norm.split()) >= 3
            )
            if joined_norm == title_norm or substantial_fragment:
                for bi in block_idxs:
                    lines[bi] = ""
                return '\n'.join(lines)

            return body_text
        return body_text

    def _strip_top_running_header(self, text: str, page_type: str) -> str:
        """
        Deterministically remove top running headers that escaped frequency rules.
        """
        if page_type not in ("text", "frontmatter", "backmatter", "chapter_start", "toc"):
            return text
        lines = text.split('\n')
        non_empty = [(i, lines[i].strip()) for i in range(len(lines)) if lines[i].strip()]
        if len(non_empty) < 2:
            return text
        first_idx, first_line = non_empty[0]
        _, second_line = non_empty[1]
        if page_type in ("frontmatter", "toc"):
            norm = _normalize_simple(first_line.rstrip(".:"))
            if norm in FRONTMATTER_SECTION_HEADING_KEYS:
                return text
        if _looks_like_running_header(first_line, second_line):
            lines[first_idx] = ""
        return '\n'.join(lines)

    def _strip_inline_signature_prefixes(self, text: str, page_type: str) -> Tuple[str, List[str]]:
        """
        Remove line-start signature/page-marker prefixes injected into prose lines,
        e.g. "I    B that ..." or "3     B 2 which ...".
        Reader-only cleanup.
        """
        if page_type not in ("text", "chapter_start", "backmatter"):
            return text, []
        lines = text.split("\n")
        removed_prefixes: List[str] = []
        out: List[str] = []
        pat = re.compile(
            r'^\s*((?:\d{1,4}|[IVXLCDM]{1,6})\s{2,}[A-Z](?:\s+\d+)?\s+)(?=[a-z])'
        )
        for line in lines:
            m = pat.match(line)
            if m:
                removed_prefixes.append(m.group(1).strip())
                out.append(line[m.end():])
            else:
                out.append(line)
        return "\n".join(out), removed_prefixes

    def _is_library_artifact_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if LIBRARY_ARTIFACT_PATTERN.search(stripped):
            return True
        if re.match(r'^\|\s*(?:Date|St\.?\s*Card|Class|Cat|Bk\.?\s*Card\.?|Checked)\s*\|', stripped, flags=re.IGNORECASE):
            return True
        if re.match(r'^\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|?$', stripped):
            return True
        return False

    def _strip_library_artifacts(self, text: str, page_type: str) -> Tuple[str, List[str]]:
        """
        Remove ownership/catalog artifacts (library stamps, accession-card rows)
        from non-body pages.
        """
        if page_type not in ("frontmatter", "illustration", "toc"):
            return text, []
        lines = text.split('\n')
        kept: List[str] = []
        removed: List[str] = []
        for line in lines:
            stripped = line.strip()
            if self._is_library_artifact_line(stripped):
                removed.append(stripped)
                continue
            # Remove empty catalog table shell lines.
            if page_type == "frontmatter" and stripped in ("| | |", "| |", "|"):
                removed.append(stripped)
                continue
            kept.append(line)
        return "\n".join(kept), removed

    def _strip_reader_frontmatter_scan_noise(self, text: str, page_type: str) -> Tuple[str, List[str]]:
        """
        Reader edition only: remove obvious scan/microfilm/library noise from
        frontmatter while preserving archival master output intact.
        """
        if page_type not in ("frontmatter", "title", "toc"):
            return text, []
        lines = text.split('\n')
        kept: List[str] = []
        removed: List[str] = []
        prev_nonempty: Optional[str] = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            # Preserve split page-reference continuation lines on TOC/illustration lists,
            # e.g. "... {235" followed by a standalone "236" on the next line.
            if (
                page_type == "toc"
                and re.fullmatch(r"\d{1,4}", stripped)
                and prev_nonempty
                and re.search(r'(?:[.·•]+\s*|\s)[\[{(]?\d{1,4}\s*$', prev_nonempty.strip())
            ):
                kept.append(line)
                prev_nonempty = stripped
                continue
            # Reader output should not show standalone roman page-number markers
            # that survive frontmatter page stitching.
            roman = stripped.rstrip('.').upper()
            if ROMAN_NUMERAL.match(roman):
                removed.append(stripped)
                continue
            # Long numeric blobs on title/imprint pages are typically scan IDs.
            if re.fullmatch(r"\d{5,}", stripped):
                removed.append(stripped)
                continue
            if any(p.search(stripped) for p in READER_FRONTMATTER_SCAN_ARTIFACTS):
                removed.append(stripped)
                continue
            kept.append(line)
            prev_nonempty = stripped
        return "\n".join(kept), removed

    def _normalize_reader_frontmatter_body(self, text: str, chapter_title: str) -> str:
        """
        Reader-only frontmatter/preface cleanup after LLM structuring:
        - split fused URL + list-entry lines
        - merge obvious split heading fragments
        - drop lingering numeric scan ID blobs
        """
        if self.archival_mode or not text.strip():
            return text
        kind = _chapter_title_kind(chapter_title)
        if kind not in {"frontmatter", "preface"}:
            return text

        lines = text.split("\n")
        out: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if s:
                m = re.match(r'^(https?://\S+)\s+(.+)$', s, flags=re.IGNORECASE)
                if m:
                    tail = m.group(2).strip()
                    out.append(m.group(1))
                    out.append("")
                    if tail:
                        out.append(tail)
                    i += 1
                    continue
                # Split fused frontmatter lines where a numbered list item got
                # appended to title/imprint text.
                m = re.match(r'^(.*?\S)\s+(\d+\.\s+["“].+)$', s)
                if m:
                    left, right = m.group(1).strip(), m.group(2).strip()
                    if left and right:
                        out.append(left)
                        out.append("")
                        out.append(right)
                        i += 1
                        continue
                if re.fullmatch(r"\d{5,}", s):
                    i += 1
                    continue

            # Merge over-split frontmatter heading fragments, e.g.
            # "### WITH AN." + "### INTRODUCTORY NOTE.".
            cur_heading = None
            if s.startswith("### "):
                cur_heading = s[4:].strip()
            elif re.match(r'(?i)^with\s+an\.?$', s):
                cur_heading = s
            if cur_heading is not None:
                cur_txt = cur_heading
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if nxt.startswith("### "):
                        nxt_txt = nxt[4:].strip()
                        if re.match(r'(?i)^with\s+an\.?$', cur_txt) and re.match(r'(?i)^introductory\s+note\.?$', nxt_txt):
                            merged = "### WITH AN INTRODUCTORY NOTE."
                            out.append(merged)
                            i += 2
                            continue

            out.append(line)
            i += 1

        return "\n".join(out)

    def _strip_reader_library_checkout_page(self, text: str, page_type: str) -> Tuple[str, List[str]]:
        """
        Reader edition only: drop library checkout-card / due-slip pages and barcode stamp
        pages in frontmatter/backmatter. Archival master retains them.
        """
        if page_type not in ("frontmatter", "backmatter"):
            return text, []
        if not text.strip():
            return text, []

        s = text.lower()
        markers = [
            "loan dept",
            "return to desk from which borrowed",
            "due on the last date stamped below",
            "renewed books are subject to immediate recall",
            "general library",
            "university of california",
            "u. c. berkeley libraries",
            "uc berkeley libraries",
        ]
        hit_count = sum(1 for m in markers if m in s)
        barcode_like = bool(re.search(r'\b[c]\d{8,}\b', s))
        non_empty_lines = [ln for ln in text.split('\n') if ln.strip()]

        # Strong due-slip page, or standalone library-barcode stamp page.
        if hit_count >= 2 or (hit_count >= 1 and barcode_like) or (barcode_like and len(non_empty_lines) <= 3):
            lines = [ln.strip() for ln in non_empty_lines]
            return "", lines[:12]

        return text, []

    def _is_frontmatter_section_heading_line(self, line: str) -> bool:
        stripped = line.strip().strip('#').strip()
        if not stripped:
            return False
        norm = _normalize_simple(stripped.rstrip(".:"))
        return norm in FRONTMATTER_SECTION_HEADING_KEYS

    def _promote_reader_frontmatter_section_headings(self, text: str) -> str:
        """
        Promote key frontmatter section labels (Contents, Illustrations, etc.)
        to markdown headings and drop exact duplicate repeats from continuation pages.
        """
        if not text:
            return text
        lines = text.split("\n")
        out: List[str] = []
        seen_keys: set = set()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            if stripped.startswith("|"):
                out.append(line)
                continue
            if stripped.startswith("#"):
                out.append(line)
                continue

            norm = _normalize_simple(stripped.rstrip(".:"))
            if norm in FRONTMATTER_SECTION_HEADING_KEYS:
                # Repeated continuation-page running headers are noise in reader output.
                if norm in seen_keys:
                    continue
                seen_keys.add(norm)
                heading_text = stripped
                if not heading_text.endswith((".", ":")):
                    heading_text = heading_text + "."
                out.append(f"### {heading_text}")
                continue

            out.append(line)

        return "\n".join(out)

    def _merge_repeated_frontmatter_headed_tables(self, text: str) -> str:
        """
        Merge repeated frontmatter sections with the same heading and table schema,
        e.g. CONTENTS spread across multiple pages after heading promotion.
        """
        if not text:
            return text
        lines = text.split("\n")

        def _heading_key_at(idx: int) -> Optional[str]:
            if idx >= len(lines):
                return None
            s = lines[idx].strip()
            if not s.startswith("#"):
                return None
            norm = _normalize_simple(re.sub(r'^#{1,6}\s*', '', s).rstrip(".:"))
            return norm if norm in FRONTMATTER_SECTION_HEADING_KEYS else None

        def _read_table(start: int) -> Tuple[Optional[int], List[str], str]:
            if start >= len(lines) or not lines[start].strip().startswith("|"):
                return None, [], ""
            if start + 1 >= len(lines):
                return None, [], ""
            header = lines[start].strip()
            sep = lines[start + 1].strip()
            if not sep.startswith("|") or not re.search(r"-{3,}", sep):
                return None, [], ""
            j = start + 2
            rows: List[str] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = lines[j].strip()
                if row and row != header and row != sep:
                    rows.append(row)
                j += 1
            return j, rows, header + "\n" + sep

        out: List[str] = []
        i = 0
        while i < len(lines):
            key = _heading_key_at(i)
            if key is None:
                out.append(lines[i])
                i += 1
                continue

            heading_line = lines[i]
            out.append(heading_line)
            i += 1
            while i < len(lines) and not lines[i].strip():
                out.append(lines[i])
                i += 1

            end_idx, rows, sig = _read_table(i)
            if end_idx is None:
                continue

            merged_rows = list(rows)
            next_i = end_idx
            while True:
                probe = next_i
                while probe < len(lines) and not lines[probe].strip():
                    probe += 1
                if _heading_key_at(probe) != key:
                    break
                probe += 1
                while probe < len(lines) and not lines[probe].strip():
                    probe += 1
                nxt_end, nxt_rows, nxt_sig = _read_table(probe)
                if nxt_end is None or nxt_sig != sig:
                    break
                merged_rows.extend(nxt_rows)
                next_i = nxt_end

            header_line, sep_line = sig.split("\n", 1)
            out.append(header_line)
            out.append(sep_line)
            out.extend(merged_rows)
            i = next_i

        return "\n".join(out)

    def _normalize_frontmatter_authorities_tables(self, text: str) -> str:
        """
        Convert 'LIST OF AUTHORITIES...' frontmatter sections into markdown tables.
        """
        if not text:
            return text
        lines = text.split("\n")
        out: List[str] = []
        i = 0

        def _is_authorities_noise_entry(entry: str) -> bool:
            norm = _normalize_simple(entry.rstrip(".:"))
            if norm in {"list of authorities", "list of authorities consulted", "indian shipping"}:
                return True
            if self._is_page_number_line(entry):
                return True
            if any(p.search(entry) for p in READER_FRONTMATTER_SCAN_ARTIFACTS):
                return True
            if re.match(r'(?i)^[ivxlcdm]+\s+\*?[a-z]\*?\s+\d+$', entry):
                return True
            return False

        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if s.startswith("#"):
                heading_norm = _normalize_simple(re.sub(r'^#{1,6}\s*', '', s).rstrip(".:"))
                if heading_norm in {"list of authorities consulted", "list of authorities"}:
                    out.append(line)
                    i += 1
                    while i < len(lines) and not lines[i].strip():
                        out.append(lines[i])
                        i += 1
                    if i < len(lines) and lines[i].strip().startswith("|"):
                        # Already structured as a markdown table: filter repeated
                        # continuation headers and scan/page-marker rows.
                        j = i
                        row_idx = 0
                        while j < len(lines) and lines[j].strip().startswith("|"):
                            row = lines[j].strip()
                            if row_idx < 2:
                                out.append(row)
                            else:
                                cell_text = row.strip("|").strip()
                                if not _is_authorities_noise_entry(cell_text):
                                    out.append(row)
                            j += 1
                            row_idx += 1
                        i = j
                        continue
                    entries: List[str] = []
                    while i < len(lines):
                        cur = lines[i]
                        cs = cur.strip()
                        if cs.startswith("#"):
                            break
                        if cs.startswith("|"):
                            break
                        if not cs:
                            i += 1
                            continue
                        if self._is_page_number_line(cs):
                            i += 1
                            continue
                        if _looks_like_running_header_with_page_number(cs) or _looks_like_page_header_combo(cs):
                            i += 1
                            continue
                        if _is_authorities_noise_entry(cs):
                            i += 1
                            continue
                        entries.append(re.sub(r'\s+', ' ', cs).strip())
                        i += 1
                    if entries:
                        out.append(_render_markdown_table([["Authority"]] + [[e] for e in entries]))
                        out.append("")
                    continue
            out.append(line)
            i += 1
        return "\n".join(out)

    def _llm_plan_frontmatter_structure(self, chapter_title: str, text: str) -> Dict[str, Any]:
        """
        LLM planner for merged frontmatter structure cleanup (reader-only).
        Returns line-id-based ops. Fail-closed.
        """
        empty = {
            "remove_ids": set(),
            "promote_ids": set(),
            "authorities_heading_ids": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty

        lines = text.split("\n")
        non_empty: List[Tuple[int, str]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                non_empty.append((idx, stripped))
        if not non_empty:
            return empty

        # Keep prompt size bounded but preserve entire frontmatter for typical books.
        max_lines = 420
        selected = non_empty[:max_lines]
        id_map: Dict[int, int] = {}
        prompt_lines: List[str] = []
        for lid, (idx, stripped) in enumerate(selected, start=1):
            id_map[lid] = idx
            prompt_lines.append(f"{lid}. {stripped}")

        short_counts: Counter = Counter()
        for _, stripped in non_empty:
            if len(stripped) <= 80 and len(stripped.split()) <= 8:
                short_counts[stripped] += 1
        repeated_short = [
            {"text": s, "count": c}
            for s, c in short_counts.items()
            if c >= 2
        ][:30]

        prompt = FRONTMATTER_STRUCTURE_OPS_PROMPT.format(
            chapter_title=chapter_title,
            repeated_short_lines=json.dumps(repeated_short, ensure_ascii=False),
            lines="\n".join(prompt_lines),
        )
        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="frontmatter_structure",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}

            def _collect_ids(key: str) -> set:
                vals = data.get(key, []) or []
                out_ids = set()
                for v in vals:
                    try:
                        lid = int(v)
                    except Exception:
                        continue
                    if lid in id_map:
                        out_ids.add(lid)
                return out_ids

            return {
                "remove_ids": _collect_ids("remove_line_ids"),
                "promote_ids": _collect_ids("promote_heading_line_ids"),
                "authorities_heading_ids": _collect_ids("authorities_table_heading_line_ids"),
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_line_id_to_idx": id_map,
            }
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _validate_frontmatter_remove_line(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        norm = _normalize_simple(s.rstrip(".:"))
        # Preserve section headings; these should be promoted, not removed.
        if norm in FRONTMATTER_SECTION_HEADING_KEYS:
            return False
        if _looks_like_footnote_definition_line(s):
            return False
        if self._is_page_number_line(s):
            return True
        if self._is_library_artifact_line(s):
            return True
        if len(s) > 80:
            return False
        words = s.split()
        if len(words) > 8:
            return False
        alpha = [c for c in s if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        return upper_ratio >= 0.85

    def _validate_frontmatter_heading_line(self, line: str) -> bool:
        s = line.strip()
        if not s or s.startswith("#"):
            return False
        if s.startswith("|"):
            return False
        if len(s) > 120:
            return False
        if re.search(r'(?i)\bpage\b', s):
            return False
        if re.match(r'^\s*(?:chapter\s+)?[ivxlcdm]+[.\-–—)]\s+\S', s, flags=re.IGNORECASE):
            return False
        if re.search(r'\b(?:frontis(?:piece)?|\d{1,4}|[ivxlcdm]{1,8})\s*$', s, flags=re.IGNORECASE):
            if len(s.split()) >= 3:
                return False
        if s.endswith(('.', ':')) or s == s.upper():
            return len(s.split()) <= 12
        return False

    def _apply_llm_frontmatter_structure_ops(self, text: str, chapter_title: str) -> str:
        """
        Apply LLM-planned line-level ops to merged frontmatter chapter text.
        Reader-only, validator-gated, no text rewriting.
        """
        if self.archival_mode:
            return text
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return text
        if not (self.gemini or self.claude):
            return text

        plan = self._llm_plan_frontmatter_structure(chapter_title, text)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.55:
            return text

        id_map = plan.get("_line_id_to_idx", {}) or {}
        lines = text.split("\n")
        removed_lines: List[str] = []
        promoted_lines: List[str] = []

        for lid in sorted(plan.get("remove_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            if not self._validate_frontmatter_remove_line(line):
                continue
            removed_lines.append(line.strip())
            lines[idx] = ""

        for lid in sorted(plan.get("promote_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            if not self._validate_frontmatter_heading_line(line):
                continue
            stripped = line.strip()
            if not stripped.endswith((".", ":")):
                stripped += "."
            lines[idx] = f"### {stripped}"
            promoted_lines.append(stripped)

        out = "\n".join(lines)
        # Deterministically honor LLM table intent for authorities sections.
        auth_ids = plan.get("authorities_heading_ids", set()) or set()
        if auth_ids:
            out = self._normalize_frontmatter_authorities_tables(out)

        if removed_lines or promoted_lines or auth_ids:
            self._record_op(
                "llm_frontmatter_structure",
                {
                    "chapter_title": chapter_title,
                    "removed": removed_lines[:20],
                    "promoted": promoted_lines[:20],
                    "authorities_table_heading_ids": sorted(int(x) for x in auth_ids)[:20],
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                },
            )
        return out

    def _llm_plan_chapter_line_roles(self, chapter_title: str, text: str) -> Dict[str, Any]:
        """
        LLM planner for merged chapter line-role cleanup (reader-only).
        Targets leaked running headers and isolated unformatted section headings.
        Returns line-id-based ops. Fail-closed.
        """
        empty = {
            "remove_ids": set(),
            "promote_ids": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty

        lines = text.split("\n")
        non_empty_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
        if not non_empty_idxs:
            return empty

        stripped_counts = Counter(
            lines[i].strip()
            for i in non_empty_idxs
            if lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", ">", "[^"))
        )
        early_cutoff = set(non_empty_idxs[:8])

        candidates: List[Dict[str, Any]] = []
        line_id_to_idx: Dict[int, int] = {}
        cid = 1

        def _near_nonempty(start: int, step: int) -> str:
            j = start + step
            limit = 0
            while 0 <= j < len(lines) and limit < 6:
                if lines[j].strip():
                    return lines[j].strip()
                j += step
                limit += 1
            return ""

        for idx in non_empty_idxs:
            s = lines[idx].strip()
            if not s:
                continue
            if s.startswith(("#", "|", ">", "[^")):
                continue
            if len(s) > 140 or len(s.split()) > 18:
                continue
            blank_before = (idx == 0) or (not lines[idx - 1].strip())
            blank_after = (idx + 1 >= len(lines)) or (not lines[idx + 1].strip())
            if not (blank_before and blank_after):
                continue

            alpha = [c for c in s if c.isalpha()]
            upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
            repeated = int(stripped_counts.get(s, 0))
            headingish = bool(
                CHAPTER_KEYWORDS.match(s)
                or _is_standalone_ordinal_marker(s)
                or (alpha and upper_ratio >= 0.7 and len(s.split()) <= 12)
            )
            removableish = bool(
                self._is_page_number_line(s)
                or self._is_library_artifact_line(s)
                or re.match(r'(?i)^univ\.?\s+of$', s)
                or re.match(r'(?i)^california\.?$', s)
                or (alpha and upper_ratio >= 0.9 and len(s.split()) <= 5)
            )
            if not (headingish or removableish or repeated >= 2):
                continue

            prev_line = _near_nonempty(idx, -1)
            next_line = _near_nonempty(idx, +1)
            candidates.append({
                "id": cid,
                "idx": idx,
                "text": s,
                "repeats": repeated,
                "prev": prev_line[:120],
                "next": next_line[:120],
                "is_early": idx in early_cutoff,
            })
            line_id_to_idx[cid] = idx
            cid += 1
            if len(candidates) >= 80:
                break

        if not candidates:
            return empty

        prompt_lines = []
        for c in candidates:
            prompt_lines.append(
                f"{c['id']}. text={json.dumps(c['text'], ensure_ascii=False)} | "
                f"repeats={c['repeats']} | prev={json.dumps(c['prev'], ensure_ascii=False)} | "
                f"next={json.dumps(c['next'], ensure_ascii=False)}"
            )

        chapter_kind = _chapter_title_kind(chapter_title)
        prompt = CHAPTER_LINE_ROLE_OPS_PROMPT.format(
            chapter_title=chapter_title,
            chapter_kind=chapter_kind,
            lines="\n".join(prompt_lines),
        )
        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="chapter_line_roles",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}

            def _collect_ids(key: str) -> set:
                vals = data.get(key, []) or []
                out_ids = set()
                valid_ids = set(line_id_to_idx.keys())
                for v in vals:
                    try:
                        lid = int(v)
                    except Exception:
                        continue
                    if lid in valid_ids:
                        out_ids.add(lid)
                return out_ids

            return {
                "remove_ids": _collect_ids("remove_line_ids"),
                "promote_ids": _collect_ids("promote_heading_line_ids"),
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_line_id_to_idx": line_id_to_idx,
                "_candidate_meta": {c["id"]: c for c in candidates},
            }
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _validate_chapter_line_remove(self, line: str, *, candidate_meta: Optional[Dict[str, Any]] = None) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith(("#", "|", ">", "[^")):
            return False
        if len(s) > 120 or len(s.split()) > 10:
            return False
        if _looks_like_footnote_definition_line(s):
            return False
        if self._is_page_number_line(s) or self._is_library_artifact_line(s):
            return True
        if re.match(r'(?i)^univ\.?\s+of$', s) or re.match(r'(?i)^california\.?$', s):
            return True

        alpha = [c for c in s if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        repeats = int((candidate_meta or {}).get("repeats", 0) or 0)
        is_early = bool((candidate_meta or {}).get("is_early", False))

        # Be conservative near chapter starts: preserve likely true headings there.
        if is_early and repeats < 2:
            return False
        if repeats >= 2 and upper_ratio >= 0.75 and len(s.split()) <= 6:
            return True
        if upper_ratio >= 0.9 and len(s.split()) <= 4 and not CHAPTER_KEYWORDS.match(s):
            return True
        return False

    def _validate_chapter_line_promote(self, line: str, *, candidate_meta: Optional[Dict[str, Any]] = None) -> bool:
        s = line.strip()
        if not s or s.startswith(("#", "|", ">", "[^")):
            return False
        if _looks_like_discourse_cue_not_heading(s):
            return False
        if len(s) > 140 or len(s.split()) > 16:
            return False
        if self._is_page_number_line(s) or self._is_library_artifact_line(s):
            return False
        if re.match(r'(?i)^univ\.?\s+of\s+california\b', s):
            return False
        if re.search(r'(?i)\bpage\b', s):
            return False
        if re.match(r'^[a-z]', s):
            return False
        alpha = [c for c in s if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        repeats = int((candidate_meta or {}).get("repeats", 0) or 0)
        if repeats >= 2 and upper_ratio >= 0.75:
            return False
        if CHAPTER_KEYWORDS.match(s) or _is_standalone_ordinal_marker(s):
            return True
        if upper_ratio >= 0.6 and len(s.split()) <= 12 and not s.endswith(","):
            return True
        return False

    def _apply_llm_chapter_line_role_ops(self, text: str, chapter_title: str) -> str:
        """
        Reader-only merged-chapter cleanup: remove leaked running headers and
        promote isolated section-title lines. Validator-gated, no text rewriting.
        """
        if self.archival_mode:
            return text
        chapter_kind = _chapter_title_kind(chapter_title)
        if chapter_kind in {"frontmatter", "preface"}:
            return text
        plan = self._llm_plan_chapter_line_roles(chapter_title, text)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.6:
            return text

        id_map = plan.get("_line_id_to_idx", {}) or {}
        candidate_meta = plan.get("_candidate_meta", {}) or {}
        chapter_title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        lines = text.split("\n")
        removed_lines: List[str] = []
        promoted_lines: List[str] = []

        for lid in sorted(plan.get("remove_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            meta = candidate_meta.get(lid) if isinstance(candidate_meta, dict) else None
            if not self._validate_chapter_line_remove(line, candidate_meta=meta):
                continue
            removed_lines.append(line.strip())
            lines[idx] = ""

        for lid in sorted(plan.get("promote_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            meta = candidate_meta.get(lid) if isinstance(candidate_meta, dict) else None
            if not self._validate_chapter_line_promote(line, candidate_meta=meta):
                continue
            stripped = line.strip()
            stripped_norm = _normalize_simple(stripped.rstrip(".:"))
            # Avoid promoting chapter-title fragments duplicated inside the body
            # (often leaked running headers on chapter-open pages).
            if chapter_title_norm and stripped_norm and (
                stripped_norm == chapter_title_norm or stripped_norm in chapter_title_norm
            ):
                if len(stripped.split()) <= 8:
                    continue
            if not stripped.endswith((".", ":")):
                stripped += "."
            if not stripped.startswith("#"):
                lines[idx] = f"#### {stripped}"
            promoted_lines.append(stripped)

        out = "\n".join(lines)
        if removed_lines or promoted_lines:
            self._record_op(
                "llm_chapter_line_roles",
                {
                    "chapter_title": chapter_title,
                    "removed": removed_lines[:20],
                    "promoted": promoted_lines[:20],
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                },
            )
        return out

    def _llm_plan_chapter_footnote_marker_insertions(
        self,
        chapter_title: str,
        body_text: str,
        footnotes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        LLM planner for converting OCR-surviving inline bare letter note markers
        (e.g. `b Beekapore`) into plain markdown refs (`[^b] Beekapore`).
        The existing footnote linker then resolves page-scoped ambiguity.
        """
        empty = {
            "insert_occurrences": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty
        if not body_text or not footnotes:
            return empty

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text)}
        unreferenced_defs = [
            fn for fn in footnotes
            if str(fn.get("id", "")).strip() and str(fn.get("id", "")).strip() not in referenced_ids
        ]
        if not unreferenced_defs:
            return empty

        resolvable_markers = _collect_resolvable_letter_markers(footnotes)
        if not resolvable_markers:
            return empty

        missing_marker_counts: Dict[str, int] = {}
        def_preview_rows: List[str] = []
        for fn in unreferenced_defs:
            fid = str(fn.get("id", "")).strip()
            marker = _footnote_inline_marker_letter(fid)
            if not marker:
                continue
            marker = marker.lower()
            if marker not in resolvable_markers:
                continue
            # Bare "a" is too ambiguous in prose; superscript-a is normalized deterministically.
            if marker == "a":
                continue
            missing_marker_counts[marker] = missing_marker_counts.get(marker, 0) + 1
            txt = re.sub(r'\s+', ' ', str(fn.get("text", "")).strip())[:120]
            def_preview_rows.append(
                f'- id={json.dumps(fid, ensure_ascii=False)} | marker={json.dumps(marker)} | starts={json.dumps(txt, ensure_ascii=False)}'
            )
            if len(def_preview_rows) >= 80:
                break
        if not missing_marker_counts:
            return empty

        # Candidate bare markers: isolated single letter before a capitalized token.
        # Example: "the b Beekapore", "joined by c Ummaud ..."
        marker_pat = re.compile(r'(^|[^A-Za-z\[\]])([b-z])(?=\s+[A-Z][A-Za-z])')
        candidates: List[Dict[str, Any]] = []
        for m in marker_pat.finditer(body_text):
            marker = (m.group(2) or "").lower()
            if marker not in missing_marker_counts:
                continue
            letter_start = m.start(2)
            letter_end = m.end(2)
            line_start = body_text.rfind("\n", 0, letter_start) + 1
            line_end = body_text.find("\n", letter_end)
            if line_end == -1:
                line_end = len(body_text)
            line_text = body_text[line_start:line_end].strip()
            if not line_text:
                continue
            if line_text.startswith(("#", ">", "[^")):
                continue
            # Reject obvious heading-like lines (e.g. "a AHMED SHAW ..." if marker regex broadens later).
            alpha = [c for c in line_text if c.isalpha()]
            upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
            if len(line_text.split()) <= 10 and upper_ratio >= 0.8:
                continue
            left = re.sub(r"\s+", " ", body_text[max(0, letter_start - 70):letter_start]).strip()
            right = re.sub(r"\s+", " ", body_text[letter_end:min(len(body_text), letter_end + 90)]).strip()
            candidates.append(
                {
                    "occurrence": len(candidates) + 1,
                    "marker": marker,
                    "start": letter_start,
                    "end": letter_end,
                    "context": (left + "[MARK]" + right).strip(),
                    "line_preview": line_text[:160],
                }
            )
            if len(candidates) >= 140:
                break

        if not candidates:
            return empty

        candidate_rows = [
            (
                f'- occurrence={c["occurrence"]} | marker={json.dumps(c["marker"])} | '
                f'context={json.dumps(c["context"], ensure_ascii=False)} | '
                f'line={json.dumps(c["line_preview"], ensure_ascii=False)}'
            )
            for c in candidates
        ]
        prompt = FOOTNOTE_MARKER_INSERT_OPS_PROMPT.format(
            chapter_title=chapter_title,
            missing_letters=json.dumps(sorted(missing_marker_counts.keys())),
            definitions="\n".join(def_preview_rows),
            candidates="\n".join(candidate_rows),
        )
        try:
            data, planner_ms, planner_provider = self._planner_json_generate(
                op_name="footnote_marker_insert",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}
            requested = set()
            for raw in (data.get("insert_marker_occurrences", []) or []):
                try:
                    occ = int(raw)
                except Exception:
                    continue
                if 1 <= occ <= len(candidates):
                    requested.add(occ)

            # Fail-closed cap per marker based on count of unreferenced defs for that marker.
            kept: set = set()
            per_marker_used: Dict[str, int] = {}
            by_occ = {c["occurrence"]: c for c in candidates}
            for occ in sorted(requested):
                c = by_occ[occ]
                marker = str(c["marker"])
                used = per_marker_used.get(marker, 0)
                if used >= int(missing_marker_counts.get(marker, 0)):
                    continue
                kept.add(occ)
                per_marker_used[marker] = used + 1

            return {
                "insert_occurrences": kept,
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_planner_provider": planner_provider,
                "_planner_ms": round(float(planner_ms), 1),
                "_candidates": candidates,
            }
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _apply_llm_chapter_footnote_marker_insertions(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> str:
        """
        Apply validated insertion of plain letter footnote refs (`[^b]`) at
        OCR-surviving inline marker positions. No prose rewriting.
        """
        if self.archival_mode or not body_text or not footnotes:
            return body_text
        plan = self._llm_plan_chapter_footnote_marker_insertions(chapter_title, body_text, footnotes)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.6:
            return body_text
        insert_occurrences = set(plan.get("insert_occurrences", set()) or set())
        candidates = list(plan.get("_candidates", []) or [])
        if not insert_occurrences or not candidates:
            return body_text

        chosen = [c for c in candidates if int(c.get("occurrence", 0)) in insert_occurrences]
        if not chosen:
            return body_text

        # Apply from end to start so offsets remain stable.
        out = body_text
        applied: List[Dict[str, Any]] = []
        for c in sorted(chosen, key=lambda x: int(x["start"]), reverse=True):
            start = int(c["start"])
            end = int(c["end"])
            marker = str(c["marker"]).lower()
            if start < 0 or end > len(out) or start >= end:
                continue
            if out[start:end].lower() != marker:
                continue
            out = out[:start] + f"[^{marker}]" + out[end:]
            applied.append(
                {
                    "occurrence": int(c["occurrence"]),
                    "marker": marker,
                    "context": str(c.get("context", ""))[:160],
                }
            )

        if applied:
            self._record_op(
                "llm_footnote_marker_insert",
                {
                    "chapter_title": chapter_title,
                    "inserted": list(reversed(applied))[:80],
                    "count": len(applied),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                    "planner_provider": str(plan.get("_planner_provider", "") or ""),
                    "planner_ms": float(plan.get("_planner_ms", 0.0) or 0.0),
                },
            )
        return out

    def _llm_plan_chapter_footnote_links(
        self,
        chapter_title: str,
        body_text: str,
        footnotes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        LLM planner for ambiguous chapter footnote linking only.
        Returns occurrence-index based remaps. Fail-closed.
        """
        empty = {
            "reassignments": {},
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty
        if not body_text or not footnotes:
            return empty

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        matches = list(ref_pat.finditer(body_text))
        if not matches:
            return empty

        def_ids = [str(fn.get("id", "")).strip() for fn in footnotes if str(fn.get("id", "")).strip()]
        defs_by_base: Dict[str, List[str]] = {}
        for fid in def_ids:
            defs_by_base.setdefault(_footnote_base_id(fid), [])
            if fid not in defs_by_base[_footnote_base_id(fid)]:
                defs_by_base[_footnote_base_id(fid)].append(fid)

        if not defs_by_base:
            return empty

        ambiguous_bases: set = set()
        ref_ids = [m.group(1) for m in matches]
        ref_counts = Counter(ref_ids)
        for base, ids in defs_by_base.items():
            if len(ids) > 1:
                ambiguous_bases.add(base)
            if any(fid != base for fid in ids):
                if any(ref_counts.get(fid, 0) == 0 for fid in ids if fid != base):
                    ambiguous_bases.add(base)
        if not ambiguous_bases:
            return empty

        protected_occurrences: set = set()
        for base, ids in defs_by_base.items():
            if base not in ambiguous_bases:
                continue
            if base not in ids:
                continue
            for occ_idx, rid in enumerate(ref_ids, start=1):
                if rid == base:
                    protected_occurrences.add(occ_idx)
                    break

        def_preview_rows: List[str] = []
        valid_target_ids: set = set()
        for base in sorted(ambiguous_bases):
            for fid in defs_by_base.get(base, []):
                fn = next((x for x in footnotes if str(x.get("id", "")).strip() == fid), None)
                start = ""
                if fn:
                    start = re.sub(r'\s+', ' ', str(fn.get("text", "")).strip())[:120]
                def_preview_rows.append(
                    f'- id={json.dumps(fid, ensure_ascii=False)} | base={json.dumps(base, ensure_ascii=False)} | '
                    f'starts={json.dumps(start, ensure_ascii=False)}'
                )
                valid_target_ids.add(fid)
                if len(def_preview_rows) >= 120:
                    break
            if len(def_preview_rows) >= 120:
                break

        if not def_preview_rows:
            return empty

        candidate_occurrences: List[int] = []
        ref_rows: List[str] = []
        for occ_idx, m in enumerate(matches, start=1):
            rid = m.group(1)
            base = _footnote_base_id(rid)
            if base not in ambiguous_bases:
                continue
            if occ_idx in protected_occurrences:
                continue
            # Preserve deterministic page-scoped links; LLM should only resolve
            # still-ambiguous plain refs (e.g. `[^1]` -> `[^1-p172]`).
            if _is_page_scoped_footnote_id(rid):
                continue
            left = body_text[max(0, m.start() - 70):m.start()]
            right = body_text[m.end():min(len(body_text), m.end() + 70)]
            left = re.sub(r'\s+', ' ', left)
            right = re.sub(r'\s+', ' ', right)
            ref_rows.append(
                f'- occurrence={occ_idx} | current_ref_id={json.dumps(rid, ensure_ascii=False)} | '
                f'base={json.dumps(base, ensure_ascii=False)} | '
                f'context={json.dumps((left + "[REF]" + right).strip(), ensure_ascii=False)}'
            )
            candidate_occurrences.append(occ_idx)
            if len(ref_rows) >= 140:
                break

        if not ref_rows:
            return empty

        prompt = FOOTNOTE_LINK_OPS_PROMPT.format(
            chapter_title=chapter_title,
            definitions="\n".join(def_preview_rows),
            references="\n".join(ref_rows),
        )
        try:
            data, planner_ms, planner_provider = self._planner_json_generate(
                op_name="footnote_link",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}

            valid_occ_set = set(candidate_occurrences)
            valid_occ_to_match = {i: matches[i - 1] for i in valid_occ_set if 1 <= i <= len(matches)}
            reassignments: Dict[int, str] = {}
            for row in (data.get("reassign_ref_occurrences", []) or []):
                if not isinstance(row, dict):
                    continue
                try:
                    occ = int(row.get("occurrence"))
                except Exception:
                    continue
                target = str(row.get("target_def_id", "")).strip()
                if occ not in valid_occ_set or not target or target not in valid_target_ids:
                    continue
                cur_rid = valid_occ_to_match[occ].group(1)
                if _footnote_base_id(cur_rid) != _footnote_base_id(target):
                    continue
                if _is_page_scoped_footnote_id(cur_rid):
                    continue
                # Avoid no-op assignments.
                if cur_rid == target:
                    continue
                reassignments[occ] = target

            out = {
                "reassignments": reassignments,
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_planner_provider": planner_provider,
                "_planner_ms": round(float(planner_ms), 1),
            }
            shadow = self._shadow_json_plan(prompt, op_name="footnote_link", priority="normal")
            if isinstance(shadow, dict):
                self._record_shadow_comparison(
                    "footnote_link",
                    {
                        "chapter_title": chapter_title,
                        "primary_count": len(reassignments),
                        "shadow_count": len(shadow.get("reassign_ref_occurrences", []) or []),
                    },
                )
            return out
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _apply_llm_chapter_footnote_link_ops(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply LLM-guided occurrence->definition footnote linking for ambiguous
        chapter refs. Deterministic validation only; note text is never rewritten.
        """
        if self.archival_mode:
            return body_text, footnotes
        plan = self._llm_plan_chapter_footnote_links(chapter_title, body_text, footnotes)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.55:
            return body_text, footnotes
        reassignments = plan.get("reassignments", {}) or {}
        if not reassignments:
            return body_text, footnotes

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        matches = list(ref_pat.finditer(body_text))
        if not matches:
            return body_text, footnotes

        # Recompute "protected" first plain refs when a plain base definition
        # exists, so apply is fail-closed even if a planner returns one.
        defs_by_base: Dict[str, List[str]] = {}
        for fn in footnotes:
            fid = str(fn.get("id", "")).strip()
            if not fid:
                continue
            base = _footnote_base_id(fid)
            defs_by_base.setdefault(base, [])
            if fid not in defs_by_base[base]:
                defs_by_base[base].append(fid)
        protected_occurrences: set = set()
        for base, ids in defs_by_base.items():
            if base not in ids:
                continue
            for occ_idx, m in enumerate(matches, start=1):
                if m.group(1) == base:
                    protected_occurrences.add(occ_idx)
                    break

        pieces: List[str] = []
        last_end = 0
        applied: List[Dict[str, Any]] = []
        for occ_idx, m in enumerate(matches, start=1):
            pieces.append(body_text[last_end:m.start()])
            cur_rid = m.group(1)
            target = str(reassignments.get(occ_idx, cur_rid))
            if occ_idx in protected_occurrences:
                target = cur_rid
            if _is_page_scoped_footnote_id(cur_rid):
                target = cur_rid
            pieces.append(f"[^{target}]")
            if target != cur_rid:
                applied.append({"occurrence": occ_idx, "from": cur_rid, "to": target})
            last_end = m.end()
        pieces.append(body_text[last_end:])
        out = "".join(pieces)

        if applied:
            self._record_op(
                "llm_footnote_link",
                {
                    "chapter_title": chapter_title,
                    "relinked": applied[:80],
                    "count": len(applied),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                    "planner_provider": str(plan.get("_planner_provider", "") or ""),
                    "planner_ms": float(plan.get("_planner_ms", 0.0) or 0.0),
                },
            )
        return out, footnotes

    def _looks_like_suspect_unreferenced_footnote_def_text(self, text: str, chapter_title: str) -> bool:
        """
        Conservative heuristic for defs likely created from chapter/running-header
        contamination rather than a real note. Used to bound LLM cleanup candidates.
        """
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not s:
            return False
        if len(s) > 120:
            return False
        if _looks_like_footnote_definition_line(s):
            return False
        words = s.split()
        if not words or len(words) > 12:
            return False
        # Real notes often include punctuation, numerals, abbreviations, or commas.
        if any(ch.isdigit() for ch in s):
            return False
        if "," in s or ";" in s:
            return False
        if re.search(r'(?i)\b(?:vol|page|pp|ibid|vide|see|called|called also|meaning|lit\.)\b', s):
            return False

        alpha = [c for c in s if c.isalpha()]
        if len(alpha) < 4:
            return False
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)

        title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        text_norm = _normalize_simple(s.rstrip(".:"))
        if title_norm and text_norm and len(words) <= 10:
            if text_norm == title_norm or text_norm in title_norm:
                return True

        # Strongly uppercase short strings are often heading fragments.
        if upper_ratio >= 0.9 and len(words) <= 8:
            return True

        # OCR header fragments sometimes preserve title case but still mirror the chapter heading.
        if upper_ratio >= 0.6 and len(words) <= 6 and title_norm and text_norm and text_norm in title_norm:
            return True

        return False

    def _looks_like_strong_bogus_unreferenced_footnote_def(
        self,
        fid: str,
        text: str,
        next_fid: str = "",
        next_text: str = "",
    ) -> bool:
        """
        Very strong deterministic signature for note defs that are actually
        chapter/running-header contamination (fail-closed).
        """
        fid_s = str(fid or "").strip()
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not fid_s or not s:
            return False
        if not re.fullmatch(r"c\d+-[a-z](?:-\d+)?", fid_s):
            return False
        if len(s) > 90:
            return False
        if any(ch.isdigit() for ch in s):
            return False
        words = s.split()
        if not (2 <= len(words) <= 8):
            return False
        # Must look like a transliterated heading fragment, not a normal note sentence.
        alpha = [c for c in s if c.isalpha()]
        if len(alpha) < 6:
            return False
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if upper_ratio < 0.9:
            return False
        if re.search(r"[,;:!?]", s):
            return False
        if re.search(r"(?i)\b(?:the|called|probably|according|about|where|which|from|with|and|or)\b", s):
            return False
        # Extra confidence: these often precede a page-scoped note that looks like a translation/gloss.
        nf = str(next_fid or "").strip()
        nt = re.sub(r"\s+", " ", str(next_text or "").strip())
        if nf and re.fullmatch(r"[a-z]-p\d+(?:-\d+)?", nf) and nt:
            if len(nt) <= 140 and re.match(r"^[A-Z][a-z]", nt):
                return True
        # Still accept without the adjacency clue if the signal is otherwise very strong.
        return True

    def _merge_split_unreferenced_page_scoped_footnote_defs(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> List[Dict[str, str]]:
        """
        Deterministically merge page-scoped continuation defs (`...-pNN-2`, `...-pNN-3`, ...)
        into their base def when the continuation itself is unreferenced.
        """
        if self.archival_mode or not footnotes:
            return footnotes

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text or "")}

        work = [dict(fn) for fn in footnotes]
        id_to_index: Dict[str, int] = {}
        for idx, fn in enumerate(work):
            fid = str(fn.get("id", "")).strip()
            if fid and fid not in id_to_index:
                id_to_index[fid] = idx

        drop_indexes: set = set()
        merged_rows: List[Dict[str, Any]] = []
        cont_pat = re.compile(r"^(.*-p\d+)-(\d+)$")

        for idx, fn in enumerate(work):
            fid = str(fn.get("id", "")).strip()
            if not fid or fid in referenced_ids:
                continue
            m = cont_pat.fullmatch(fid)
            if not m:
                continue
            try:
                suffix_num = int(m.group(2))
            except Exception:
                continue
            if suffix_num < 2:
                continue

            base_id = str(m.group(1)).strip()
            base_idx = id_to_index.get(base_id)
            if base_idx is None or base_idx >= idx:
                continue

            cont_text = str(fn.get("text", "")).strip()
            base_text = str(work[base_idx].get("text", "")).strip()
            if not cont_text or not base_text:
                continue

            if _normalize_footnote_text_for_compare(cont_text) in _normalize_footnote_text_for_compare(base_text):
                drop_indexes.add(idx)
                merged_rows.append({"from_id": fid, "into_id": base_id, "reason": "duplicate_continuation"})
                continue

            joiner = ""
            if base_text.endswith("-"):
                joiner = ""
            elif base_text.endswith((".", ";", ":", ",", "?", "!")):
                joiner = "\n"
            else:
                joiner = "\n"
            work[base_idx]["text"] = (base_text + joiner + cont_text).strip()
            drop_indexes.add(idx)
            merged_rows.append({"from_id": fid, "into_id": base_id, "reason": "split_continuation_merge"})

        if not drop_indexes:
            return footnotes

        out = [fn for i, fn in enumerate(work) if i not in drop_indexes]
        self._record_op(
            "deterministic_split_footnote_def_merge",
            {
                "chapter_title": chapter_title,
                "count": len(merged_rows),
                "sample": merged_rows[:40],
            },
        )
        return out

    def _merge_unreferenced_page_scoped_sibling_defs(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> List[Dict[str, str]]:
        """
        Deterministic fallback for unresolved page-scoped sibling defs.

        Example:
        - [^a-p102] is referenced
        - [^b-p102] is unreferenced with no viable anchor candidate in body
        => merge b-p102 text into a-p102 to preserve note content without
           inventing a new reference.
        """
        if self.archival_mode or not footnotes:
            return footnotes

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text or "")}
        unresolved_ids = {
            str(fn.get("id", "")).strip()
            for fn in footnotes
            if str(fn.get("id", "")).strip() and str(fn.get("id", "")).strip() not in referenced_ids
        }
        if not unresolved_ids:
            return footnotes

        work = [dict(fn) for fn in footnotes]
        page_pat = re.compile(r"^([a-z])-p(\d+)$", flags=re.IGNORECASE)
        by_page: Dict[str, List[Tuple[int, str]]] = {}
        for idx, fn in enumerate(work):
            fid = str(fn.get("id", "")).strip()
            m = page_pat.fullmatch(fid)
            if not m:
                continue
            page_key = m.group(2)
            by_page.setdefault(page_key, []).append((idx, fid))

        drop_idxs: set = set()
        merges: List[Dict[str, Any]] = []

        for idx, fn in enumerate(work):
            src_id = str(fn.get("id", "")).strip()
            src_text = str(fn.get("text", "")).strip()
            if not src_id or not src_text:
                continue
            if src_id not in unresolved_ids:
                continue
            m = page_pat.fullmatch(src_id)
            if not m:
                continue
            page_key = m.group(2)
            siblings = by_page.get(page_key, [])
            if len(siblings) < 2:
                continue

            # Prefer nearest earlier sibling already referenced in body.
            target_idx: Optional[int] = None
            target_id: str = ""
            for s_idx, s_id in siblings:
                if s_idx >= idx:
                    continue
                if s_id in referenced_ids:
                    target_idx = s_idx
                    target_id = s_id
            if target_idx is None:
                # Fallback: nearest earlier sibling, even if not referenced.
                for s_idx, s_id in siblings:
                    if s_idx < idx:
                        target_idx = s_idx
                        target_id = s_id

            if target_idx is None or not target_id:
                continue
            if target_idx in drop_idxs or idx in drop_idxs:
                continue

            tgt_text = str(work[target_idx].get("text", "")).strip()
            if not tgt_text:
                continue

            norm_src = _normalize_footnote_text_for_compare(src_text)
            norm_tgt = _normalize_footnote_text_for_compare(tgt_text)
            if norm_src in norm_tgt:
                drop_idxs.add(idx)
                merges.append(
                    {
                        "from_id": src_id,
                        "into_id": target_id,
                        "reason": "duplicate_sibling_page_scoped",
                    }
                )
                continue

            joiner = "" if tgt_text.endswith("-") else "\n"
            work[target_idx]["text"] = (tgt_text + joiner + src_text).strip()
            drop_idxs.add(idx)
            merges.append(
                {
                    "from_id": src_id,
                    "into_id": target_id,
                    "reason": "unreferenced_sibling_page_scoped_merge",
                }
            )

        if not merges:
            return footnotes

        out = [fn for i, fn in enumerate(work) if i not in drop_idxs]
        self._record_op(
            "deterministic_sibling_page_footnote_merge",
            {
                "chapter_title": chapter_title,
                "count": len(merges),
                "sample": merges[:40],
            },
        )
        return out

    def _footnote_anchor_match_surface(self, text: str) -> str:
        """
        Normalize OCR glyph variants while preserving string length for match offsets.
        """
        if not text:
            return ""
        return text.translate(str.maketrans({
            "ſ": "s",
            "’": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
        }))

    def _derive_footnote_anchor_phrases(self, footnote_text: str, chapter_title: str) -> List[str]:
        """
        Build conservative literal anchor phrases from a footnote definition.
        """
        raw = re.sub(r"\s+", " ", str(footnote_text or "").strip())
        if not raw:
            return []

        lead = re.split(r"[.;:!?]", raw, maxsplit=1)[0].strip()
        if not lead:
            lead = raw

        tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", lead)
        anchors: List[str] = []
        stop = {
            "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at", "by",
            "from", "with", "this", "that", "these", "those", "is", "are", "was", "were",
            "be", "been", "being", "it", "as", "who", "whom", "which", "where", "when",
            "his", "her", "their", "its", "he", "she", "they", "we", "i",
        }

        if len(tokens) >= 2:
            anchors.append(" ".join(tokens[: min(6, len(tokens))]))
        elif tokens:
            anchors.append(tokens[0])

        cap_run: List[str] = []
        for tok in tokens[:10]:
            if tok[:1].isupper():
                cap_run.append(tok)
                if len(cap_run) >= 4:
                    break
            elif cap_run:
                break
        if len(cap_run) >= 1:
            anchors.append(" ".join(cap_run))

        for tok in tokens[:10]:
            t = tok.strip()
            if len(t) < 5:
                continue
            if t.lower() in stop:
                continue
            anchors.append(t)

        if len(anchors) <= 1:
            title_tokens = [
                t for t in re.findall(r"[A-Za-z][A-Za-z'’-]*", str(chapter_title or ""))
                if len(t) >= 5 and t.lower() not in {"chapter", "sultan", "period", "history"}
            ]
            if title_tokens:
                anchors.append(" ".join(title_tokens[:2]))
                anchors.append(title_tokens[0])

        out: List[str] = []
        seen: set = set()
        for a in anchors:
            s = re.sub(r"\s+", " ", str(a or "").strip(" '\".,;:()[]{}"))
            if len(s) < 2:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= 6:
                break
        return out

    def _build_chapter_footnote_residual_candidates(
        self,
        chapter_title: str,
        body_text: str,
        footnotes: List[Dict[str, str]],
        unresolved_ids: set,
    ) -> List[Dict[str, Any]]:
        """
        Candidate insertion points for unresolved defs, found via literal anchor
        phrase matches in chapter body.
        """
        if not body_text or not footnotes or not unresolved_ids:
            return []

        body_surface = self._footnote_anchor_match_surface(body_text)
        candidates: List[Dict[str, Any]] = []
        occ_counter = 1
        cap_total = 220

        for fn in footnotes:
            fid = str(fn.get("id", "")).strip()
            if not fid or fid not in unresolved_ids:
                continue
            text = str(fn.get("text", "")).strip()
            anchors = self._derive_footnote_anchor_phrases(text, chapter_title)
            if not anchors:
                continue

            seen_positions: set = set()
            found_for_def = False
            for anchor in anchors:
                anchor_surface = self._footnote_anchor_match_surface(anchor)
                if len(anchor_surface) < 3:
                    continue
                try:
                    matches = list(re.finditer(re.escape(anchor_surface), body_surface, flags=re.IGNORECASE))
                except re.error:
                    continue
                if not matches:
                    continue

                per_anchor = 0
                for m in matches:
                    insert_at = int(m.end())
                    if insert_at in seen_positions:
                        continue
                    if insert_at <= 0 or insert_at > len(body_text):
                        continue
                    right = body_text[insert_at:min(len(body_text), insert_at + 10)]
                    left = body_text[max(0, insert_at - 3):insert_at]
                    if right.startswith("[^") or left.endswith("[^"):
                        continue
                    seen_positions.add(insert_at)
                    found_for_def = True
                    per_anchor += 1

                    c_left = re.sub(r"\s+", " ", body_text[max(0, insert_at - 80):insert_at]).strip()
                    c_right = re.sub(r"\s+", " ", body_text[insert_at:min(len(body_text), insert_at + 90)]).strip()
                    candidates.append(
                        {
                            "occurrence": occ_counter,
                            "target_def_id": fid,
                            "start": insert_at,
                            "anchor": anchor[:90],
                            "context": (c_left + "[REF]" + c_right).strip(),
                        }
                    )
                    occ_counter += 1
                    if len(candidates) >= cap_total or per_anchor >= 3:
                        break
                if len(candidates) >= cap_total:
                    break
                # Keep only strongest successful anchor family per definition.
                if found_for_def:
                    break
            if len(candidates) >= cap_total:
                break

        return candidates

    def _is_valid_residual_drop_candidate(self, fid: str, text: str, chapter_title: str) -> bool:
        """
        Conservative validator for residual LLM drop suggestions.
        """
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not s:
            return False
        if self._looks_like_suspect_unreferenced_footnote_def_text(s, chapter_title):
            return True
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", s)
        if len(words) <= 8:
            return True
        if _is_page_scoped_footnote_id(fid) and re.match(r"^[a-z]", s):
            return True
        return False

    def _llm_plan_chapter_footnote_residual_ops(
        self,
        chapter_title: str,
        body_text: str,
        footnotes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Planner for residual chapter-local footnote mismatches:
        - insert unresolved refs at candidate inline positions
        - optionally drop validated residual noise defs
        """
        empty = {
            "insert_occurrences": set(),
            "drop_def_ids": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty
        if not body_text or not footnotes:
            return empty

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text)}
        unresolved = [
            fn for fn in footnotes
            if str(fn.get("id", "")).strip() and str(fn.get("id", "")).strip() not in referenced_ids
        ]
        if not unresolved:
            return empty

        unresolved_ids = {str(fn.get("id", "")).strip() for fn in unresolved}
        candidates = self._build_chapter_footnote_residual_candidates(
            chapter_title,
            body_text,
            footnotes,
            unresolved_ids,
        )
        if not candidates:
            return empty

        def_rows: List[str] = []
        drop_candidate_ids: set = set()
        drop_rows: List[str] = []
        for fn in unresolved:
            fid = str(fn.get("id", "")).strip()
            txt = re.sub(r"\s+", " ", str(fn.get("text", "")).strip())[:180]
            def_rows.append(
                f'- id={json.dumps(fid, ensure_ascii=False)} | starts={json.dumps(txt, ensure_ascii=False)}'
            )
            if self._is_valid_residual_drop_candidate(fid, txt, chapter_title):
                drop_candidate_ids.add(fid)
                drop_rows.append(
                    f'- id={json.dumps(fid, ensure_ascii=False)} | starts={json.dumps(txt, ensure_ascii=False)}'
                )
            if len(def_rows) >= 100:
                break

        cand_rows = [
            (
                f'- occurrence={c["occurrence"]} | target_def_id={json.dumps(c["target_def_id"], ensure_ascii=False)} | '
                f'anchor={json.dumps(c["anchor"], ensure_ascii=False)} | '
                f'context={json.dumps(c["context"], ensure_ascii=False)}'
            )
            for c in candidates
        ]
        prompt = FOOTNOTE_RESIDUAL_OPS_PROMPT.format(
            chapter_title=chapter_title,
            definitions="\n".join(def_rows),
            candidates="\n".join(cand_rows),
            drop_candidates="\n".join(drop_rows) if drop_rows else "- (none)",
        )
        try:
            data, planner_ms, planner_provider = self._planner_json_generate(
                op_name="footnote_residual_cleanup",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}

            by_occ = {int(c["occurrence"]): c for c in candidates if int(c.get("occurrence", 0)) > 0}
            requested_insert: set = set()
            used_defs: set = set()
            for row in (data.get("insert_ref_occurrences", []) or []):
                if not isinstance(row, dict):
                    continue
                try:
                    occ = int(row.get("occurrence"))
                except Exception:
                    continue
                cand = by_occ.get(occ)
                if not cand:
                    continue
                target = str(row.get("target_def_id", "")).strip()
                if target and target != str(cand.get("target_def_id", "")):
                    continue
                def_id = str(cand.get("target_def_id", "")).strip()
                if not def_id or def_id in used_defs:
                    continue
                requested_insert.add(occ)
                used_defs.add(def_id)

            requested_drop: set = set()
            for raw in (data.get("drop_def_ids", []) or []):
                fid = str(raw or "").strip()
                if fid and fid in drop_candidate_ids:
                    requested_drop.add(fid)

            return {
                "insert_occurrences": requested_insert,
                "drop_def_ids": requested_drop,
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_planner_provider": planner_provider,
                "_planner_ms": round(float(planner_ms), 1),
                "_candidates": candidates,
            }
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _apply_llm_chapter_footnote_residual_ops(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Apply residual insertion/drop ops with deterministic validation.
        """
        if self.archival_mode or not body_text or not footnotes:
            return body_text, footnotes

        plan = self._llm_plan_chapter_footnote_residual_ops(chapter_title, body_text, footnotes)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.55:
            return body_text, footnotes

        candidates = list(plan.get("_candidates", []) or [])
        insert_occurrences = set(plan.get("insert_occurrences", set()) or set())
        drop_ids_requested = set(plan.get("drop_def_ids", set()) or set())
        if not insert_occurrences and not drop_ids_requested:
            return body_text, footnotes

        by_occ = {
            int(c.get("occurrence", 0)): c
            for c in candidates
            if int(c.get("occurrence", 0)) > 0
        }
        chosen = [by_occ[occ] for occ in sorted(insert_occurrences) if occ in by_occ]

        out_body = body_text
        inserted: List[Dict[str, Any]] = []
        for c in sorted(chosen, key=lambda x: int(x.get("start", 0)), reverse=True):
            fid = str(c.get("target_def_id", "")).strip()
            start = int(c.get("start", -1))
            if not fid or start < 0 or start > len(out_body):
                continue
            right = out_body[start:min(len(out_body), start + 8)]
            if right.startswith("[^"):
                continue
            out_body = out_body[:start] + f"[^{fid}]" + out_body[start:]
            inserted.append(
                {
                    "occurrence": int(c.get("occurrence", 0)),
                    "target_def_id": fid,
                    "anchor": str(c.get("anchor", ""))[:80],
                    "context": str(c.get("context", ""))[:160],
                }
            )

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        refs_after_insert = {m.group(1) for m in ref_pat.finditer(out_body)}
        dropped: List[Dict[str, Any]] = []
        kept: List[Dict[str, str]] = []
        drop_enabled = float(plan.get("confidence", 0.0) or 0.0) >= 0.65

        for fn in footnotes:
            fid = str(fn.get("id", "")).strip()
            txt = str(fn.get("text", "")).strip()
            if (
                drop_enabled
                and fid in drop_ids_requested
                and fid not in refs_after_insert
                and self._is_valid_residual_drop_candidate(fid, txt, chapter_title)
            ):
                dropped.append(
                    {
                        "id": fid,
                        "text": re.sub(r"\s+", " ", txt)[:180],
                    }
                )
                continue
            kept.append(fn)

        if inserted or dropped:
            self._record_op(
                "llm_footnote_residual_cleanup",
                {
                    "chapter_title": chapter_title,
                    "inserted": list(reversed(inserted))[:80],
                    "inserted_count": len(inserted),
                    "dropped": dropped[:40],
                    "dropped_count": len(dropped),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                    "planner_provider": str(plan.get("_planner_provider", "") or ""),
                    "planner_ms": float(plan.get("_planner_ms", 0.0) or 0.0),
                },
            )
        return out_body, kept

    def _llm_plan_chapter_footnote_def_cleanup(
        self,
        chapter_title: str,
        body_text: str,
        footnotes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        LLM planner for dropping obviously bogus unreferenced footnote definitions
        (header/title contamination only). Fail-closed and validator-gated.
        """
        empty = {
            "drop_def_ids": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty
        if not footnotes:
            return empty

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text or "")}

        candidates: List[Dict[str, str]] = []
        candidate_ids: set = set()
        for fn in footnotes:
            fid = str(fn.get("id", "")).strip()
            if not fid or fid in referenced_ids:
                continue
            txt = str(fn.get("text", "")).strip()
            if not self._looks_like_suspect_unreferenced_footnote_def_text(txt, chapter_title):
                continue
            candidates.append({
                "id": fid,
                "text": re.sub(r"\s+", " ", txt)[:180],
            })
            candidate_ids.add(fid)
            if len(candidates) >= 40:
                break

        if not candidates:
            return empty

        prompt_lines = [
            f'- id={json.dumps(c["id"], ensure_ascii=False)} | text={json.dumps(c["text"], ensure_ascii=False)}'
            for c in candidates
        ]
        prompt = (
            "You are classifying suspect unreferenced footnote definitions in an OCR-assembled historical book.\n"
            "Task: select ONLY definitions that are clearly chapter/running-header contamination or title fragments,\n"
            "NOT real explanatory notes. Be conservative.\n\n"
            f"Chapter title: {json.dumps(chapter_title, ensure_ascii=False)}\n\n"
            "Candidates (all are currently unreferenced):\n"
            + "\n".join(prompt_lines)
            + "\n\nReturn JSON with keys:\n"
            "{\n"
            '  "drop_def_ids": ["..."],\n'
            '  "confidence": 0.0,\n'
            '  "reason": "short reason"\n'
            "}\n"
        )
        try:
            data, planner_ms, planner_provider = self._planner_json_generate(
                op_name="footnote_def_cleanup",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}
            drop_ids = set()
            for raw in (data.get("drop_def_ids", []) or []):
                fid = str(raw or "").strip()
                if fid in candidate_ids:
                    drop_ids.add(fid)
            return {
                "drop_def_ids": drop_ids,
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_planner_provider": planner_provider,
                "_planner_ms": round(float(planner_ms), 1),
            }
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _apply_llm_chapter_footnote_def_cleanup_ops(
        self,
        body_text: str,
        footnotes: List[Dict[str, str]],
        chapter_title: str,
    ) -> List[Dict[str, str]]:
        """
        Drop only LLM-classified, validator-approved bogus unreferenced defs.
        Reader-only and fail-closed.
        """
        if self.archival_mode or not footnotes:
            return footnotes

        ref_pat = re.compile(r'\[\^([\w*†‡§\d-]+)\]')
        referenced_ids = {m.group(1) for m in ref_pat.finditer(body_text or "")}
        strong_auto_drop_ids: set = set()
        for idx, fn in enumerate(footnotes):
            fid = str(fn.get("id", "")).strip()
            if not fid or fid in referenced_ids:
                continue
            txt = str(fn.get("text", "")).strip()
            next_fid = ""
            next_text = ""
            if idx + 1 < len(footnotes):
                next_fid = str(footnotes[idx + 1].get("id", "")).strip()
                next_text = str(footnotes[idx + 1].get("text", "")).strip()
            if self._looks_like_strong_bogus_unreferenced_footnote_def(fid, txt, next_fid, next_text):
                strong_auto_drop_ids.add(fid)

        plan = self._llm_plan_chapter_footnote_def_cleanup(chapter_title, body_text, footnotes)
        llm_drop_ids: set = set()
        if float(plan.get("confidence", 0.0) or 0.0) >= 0.65:
            llm_drop_ids = set(plan.get("drop_def_ids", set()) or set())

        drop_ids_requested = strong_auto_drop_ids | llm_drop_ids
        if not drop_ids_requested:
            return footnotes

        kept: List[Dict[str, str]] = []
        dropped: List[Dict[str, Any]] = []
        for fn in footnotes:
            fid = str(fn.get("id", "")).strip()
            txt = str(fn.get("text", "")).strip()
            if fid not in drop_ids_requested:
                kept.append(fn)
                continue
            # Final safety validation.
            if fid in referenced_ids:
                kept.append(fn)
                continue
            if fid in strong_auto_drop_ids:
                dropped.append(
                    {
                        "id": fid,
                        "text": re.sub(r"\s+", " ", txt)[:180],
                        "source": "deterministic_strong_header_noise",
                    }
                )
                continue
            if not self._looks_like_suspect_unreferenced_footnote_def_text(txt, chapter_title):
                kept.append(fn)
                continue
            dropped.append(
                {
                    "id": fid,
                    "text": re.sub(r"\s+", " ", txt)[:180],
                    "source": "llm",
                }
            )

        if dropped:
            self._record_op(
                "llm_footnote_def_cleanup",
                {
                    "chapter_title": chapter_title,
                    "dropped": dropped[:40],
                    "count": len(dropped),
                    "deterministic_count": len([d for d in dropped if d.get("source") == "deterministic_strong_header_noise"]),
                    "llm_count": len([d for d in dropped if d.get("source") == "llm"]),
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                    "planner_provider": str(plan.get("_planner_provider", "") or ""),
                    "planner_ms": float(plan.get("_planner_ms", 0.0) or 0.0),
                },
            )
        return kept

    def _llm_plan_backmatter_structure(self, chapter_title: str, text: str) -> Dict[str, Any]:
        """
        LLM planner for reference-heavy backmatter formatting (reader-only).
        Targets indexes, bibliographies, appendices, references.
        Returns line-id-based ops. Fail-closed.
        """
        empty = {
            "remove_ids": set(),
            "promote_ids": set(),
            "letter_divider_ids": set(),
            "confidence": 0.0,
            "reason": "",
        }
        if self.archival_mode:
            return empty
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return empty
        if not (self.gemini or self.claude):
            return empty
        if not self._is_reference_like_backmatter_chapter(chapter_title, text):
            return empty

        lines = text.split("\n")
        non_empty_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
        if not non_empty_idxs:
            return empty

        stripped_counts = Counter(
            lines[i].strip()
            for i in non_empty_idxs
            if lines[i].strip() and not lines[i].strip().startswith(("#", "|", ">", "[^"))
        )

        def _near_nonempty(start: int, step: int) -> str:
            j = start + step
            hops = 0
            while 0 <= j < len(lines) and hops < 8:
                s = lines[j].strip()
                if s:
                    return s
                j += step
                hops += 1
            return ""

        candidates: List[Dict[str, Any]] = []
        line_id_to_idx: Dict[int, int] = {}
        cid = 1

        for idx in non_empty_idxs:
            s = lines[idx].strip()
            if not s:
                continue
            if s.startswith(("#", "|", ">", "[^")):
                continue

            blank_before = (idx == 0) or (not lines[idx - 1].strip())
            blank_after = (idx + 1 >= len(lines)) or (not lines[idx + 1].strip())
            prev_line = _near_nonempty(idx, -1)
            next_line = _near_nonempty(idx, +1)
            repeats = int(stripped_counts.get(s, 0))

            alpha = [c for c in s if c.isalpha()]
            upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
            single_letter = bool(re.fullmatch(r'[A-Z]', s))
            same_letter_next = bool(
                single_letter and next_line and _normalize_simple(next_line).startswith(s.lower())
            )
            headingish = bool(
                re.search(r'(?i)^(index|appendix|bibliography|references?|works cited|authorities)\b', s)
                or CHAPTER_KEYWORDS.match(s)
                or _is_standalone_ordinal_marker(s)
                or (alpha and upper_ratio >= 0.75 and len(s.split()) <= 12)
            )
            removableish = bool(
                self._is_page_number_line(s)
                or self._is_library_artifact_line(s)
                or _looks_like_running_header_with_page_number(s)
                or _looks_like_page_header_combo(s)
                or re.match(r'^\d+\s+[A-Z]\s+\d+$', s)
                or re.search(r'(?i)^\s*printed by\b', s)
                or re.search(r'(?i)^\s*london\s*:\s*$', s)
            )

            if len(s) > 180:
                continue
            if not (
                removableish
                or headingish
                or (single_letter and (blank_before or blank_after))
                or (repeats >= 2 and len(s.split()) <= 8)
            ):
                continue

            candidates.append({
                "id": cid,
                "idx": idx,
                "text": s,
                "prev": prev_line[:120],
                "next": next_line[:120],
                "repeats": repeats,
                "blank_before": blank_before,
                "blank_after": blank_after,
                "single_letter": single_letter,
                "same_letter_next": same_letter_next,
            })
            line_id_to_idx[cid] = idx
            cid += 1
            if len(candidates) >= 140:
                break

        if not candidates:
            return empty

        backmatter_kind = _chapter_title_kind(chapter_title)
        prompt_lines = []
        for c in candidates:
            prompt_lines.append(
                f"{c['id']}. text={json.dumps(c['text'], ensure_ascii=False)} | "
                f"prev={json.dumps(c['prev'], ensure_ascii=False)} | "
                f"next={json.dumps(c['next'], ensure_ascii=False)} | "
                f"repeats={c['repeats']} | "
                f"isolated={c['blank_before'] and c['blank_after']} | "
                f"single_letter={c['single_letter']} | same_letter_next={c['same_letter_next']}"
            )

        prompt = BACKMATTER_STRUCTURE_OPS_PROMPT.format(
            chapter_title=chapter_title,
            backmatter_kind=backmatter_kind,
            lines="\n".join(prompt_lines),
        )
        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="backmatter_structure",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {**empty, "reason": "invalid json"}

            def _collect_ids(key: str) -> set:
                vals = data.get(key, []) or []
                out_ids = set()
                valid_ids = set(line_id_to_idx.keys())
                for v in vals:
                    try:
                        lid = int(v)
                    except Exception:
                        continue
                    if lid in valid_ids:
                        out_ids.add(lid)
                return out_ids

            out = {
                "remove_ids": _collect_ids("remove_line_ids"),
                "promote_ids": _collect_ids("promote_heading_line_ids"),
                "letter_divider_ids": _collect_ids("promote_letter_divider_line_ids"),
                "confidence": float(data.get("confidence", 0.0) or 0.0),
                "reason": str(data.get("reason", "") or ""),
                "_line_id_to_idx": line_id_to_idx,
                "_candidate_meta": {c["id"]: c for c in candidates},
            }

            shadow = self._shadow_json_plan(prompt, op_name="backmatter_structure", priority="high")
            if isinstance(shadow, dict):
                self._record_shadow_comparison(
                    "backmatter_structure",
                    {
                        "chapter_title": chapter_title,
                        "primary_remove": sorted(int(x) for x in out["remove_ids"])[:30],
                        "shadow_remove": sorted(
                            int(x) for x in (shadow.get("remove_line_ids", []) or [])
                            if str(x).isdigit()
                        )[:30],
                    },
                )
            return out
        except Exception as e:
            return {**empty, "reason": str(e)}

    def _validate_backmatter_remove_line(self, line: str, *, candidate_meta: Optional[Dict[str, Any]] = None) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith(("#", "|", ">", "[^")):
            return False
        if len(s) > 180:
            return False
        if _looks_like_footnote_definition_line(s):
            return False
        if self._is_page_number_line(s) or self._is_library_artifact_line(s):
            return True
        if _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
            return True
        if re.match(r'^\d+\s+[A-Z]\s+\d+$', s):
            return True
        if re.search(r'(?i)^\s*printed by\b', s):
            return True
        if re.search(r'(?i)^\s*london\s*:\s*$', s):
            return True
        if re.search(r'(?i)\bstamford street\b|\bduke street\b', s):
            return True
        # Repeated short all-caps lines in reference backmatter are usually leaked headers.
        repeats = int((candidate_meta or {}).get("repeats", 0) or 0)
        alpha = [c for c in s if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        if repeats >= 2 and upper_ratio >= 0.75 and len(s.split()) <= 10:
            return True
        return False

    def _validate_backmatter_heading_line(self, line: str, *, candidate_meta: Optional[Dict[str, Any]] = None) -> bool:
        s = line.strip()
        if not s or s.startswith(("#", "|", ">", "[^")):
            return False
        if len(s) > 180 or len(s.split()) > 18:
            return False
        if self._is_page_number_line(s) or self._is_library_artifact_line(s):
            return False
        if _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
            return False
        if re.search(r'(?i)^\s*printed by\b', s):
            return False
        if re.search(r'(?i)\b(page|facing page)\b', s):
            return False
        if re.search(r'(?i)^(index|appendix|bibliography|references?|works cited|authorities)\b', s):
            return True
        if CHAPTER_KEYWORDS.match(s) or _is_standalone_ordinal_marker(s):
            return True
        alpha = [c for c in s if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if alpha else 0.0
        return upper_ratio >= 0.65 and len(s.split()) <= 14

    def _validate_backmatter_letter_divider(
        self,
        line: str,
        *,
        candidate_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        s = line.strip()
        if not re.fullmatch(r'[A-Z]', s):
            return False
        if self._is_page_number_line(s):
            return False
        meta = candidate_meta or {}
        if not bool(meta.get("same_letter_next", False)):
            return False
        return bool(meta.get("blank_before", False) or meta.get("blank_after", False))

    def _apply_llm_backmatter_structure_ops(self, text: str, chapter_title: str) -> str:
        """
        Reader-only backmatter formatting pass driven by LLM line-level ops and
        strict deterministic validators. No text rewriting.
        """
        if self.archival_mode:
            return text
        plan = self._llm_plan_backmatter_structure(chapter_title, text)
        if float(plan.get("confidence", 0.0) or 0.0) < 0.55:
            return text

        id_map = plan.get("_line_id_to_idx", {}) or {}
        candidate_meta = plan.get("_candidate_meta", {}) or {}
        lines = text.split("\n")
        removed_lines: List[str] = []
        promoted_lines: List[str] = []
        promoted_letters: List[str] = []

        for lid in sorted(plan.get("remove_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            meta = candidate_meta.get(lid) if isinstance(candidate_meta, dict) else None
            if not self._validate_backmatter_remove_line(line, candidate_meta=meta):
                continue
            removed_lines.append(line.strip())
            lines[idx] = ""

        for lid in sorted(plan.get("promote_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            meta = candidate_meta.get(lid) if isinstance(candidate_meta, dict) else None
            if not self._validate_backmatter_heading_line(line, candidate_meta=meta):
                continue
            stripped = line.strip()
            if not stripped.endswith((".", ":")):
                stripped += "."
            lines[idx] = f"#### {stripped}"
            promoted_lines.append(stripped)

        for lid in sorted(plan.get("letter_divider_ids", set()) or set()):
            idx = id_map.get(lid)
            if idx is None or idx >= len(lines):
                continue
            line = lines[idx]
            meta = candidate_meta.get(lid) if isinstance(candidate_meta, dict) else None
            if not self._validate_backmatter_letter_divider(line, candidate_meta=meta):
                continue
            stripped = line.strip()
            lines[idx] = f"#### {stripped}."
            promoted_letters.append(stripped)

        out = "\n".join(lines)
        if removed_lines or promoted_lines or promoted_letters:
            self._record_op(
                "llm_backmatter_structure",
                {
                    "chapter_title": chapter_title,
                    "removed": removed_lines[:20],
                    "promoted": promoted_lines[:20],
                    "promoted_letter_dividers": promoted_letters[:20],
                    "confidence": float(plan.get("confidence", 0.0) or 0.0),
                    "reason": str(plan.get("reason", "") or ""),
                },
            )
        return out

    def _promote_reader_body_heading_blocks(self, text: str, chapter_title: str) -> str:
        """
        Deterministically promote obvious in-body section headings that remain as
        isolated plain-text lines after chapter assembly (reader edition only).
        Conservative by design: only touches short, visually isolated blocks.
        """
        if self.archival_mode or not text.strip():
            return text

        chapter_kind = _chapter_title_kind(chapter_title)
        if chapter_kind in {"frontmatter", "preface"}:
            return text

        lines = text.split("\n")
        out = list(lines)
        chapter_title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        line_counts = Counter(
            ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith(("#", "|", ">", "[^"))
        )
        promoted_blocks: List[str] = []
        i = 0

        def _next_nonempty(idx: int) -> str:
            j = idx + 1
            while j < len(out):
                s = out[j].strip()
                if s:
                    return s
                j += 1
            return ""

        def _is_candidate_line(s: str) -> bool:
            if not s:
                return False
            if s.startswith(("#", "|", ">", "[^")):
                return False
            if _looks_like_footnote_definition_line(s):
                return False
            if self._is_page_number_line(s) or self._is_library_artifact_line(s):
                return False
            if len(s) > 120 or len(s.split()) > 14:
                return False
            return True

        while i < len(out):
            s = out[i].strip()
            if not s:
                i += 1
                continue
            if i > 0 and out[i - 1].strip():
                i += 1
                continue
            if not _is_candidate_line(s):
                i += 1
                continue

            block_idxs: List[int] = []
            block_lines: List[str] = []
            j = i
            while j < len(out):
                cur = out[j].strip()
                if not cur:
                    break
                if not _is_candidate_line(cur):
                    break
                block_idxs.append(j)
                block_lines.append(cur)
                j += 1
                if len(block_lines) >= 4:
                    break

            if not block_lines:
                i += 1
                continue

            # Require visual isolation (blank after block or end-of-text).
            if j < len(out) and out[j].strip():
                i += 1
                continue

            joined = " ".join(block_lines)
            joined_norm = _normalize_simple(joined.rstrip(".:"))
            if not joined_norm:
                i += 1
                continue
            if chapter_title_norm and (joined_norm == chapter_title_norm or joined_norm in chapter_title_norm):
                i += 1
                continue
            if _looks_like_discourse_cue_not_heading(joined):
                i += 1
                continue
            if re.search(r'[;,@]\s*$', joined):
                i += 1
                continue

            next_nonempty = _next_nonempty(j - 1)
            if not next_nonempty:
                i += 1
                continue
            if next_nonempty.startswith(("#", "|", ">", "[^")):
                i += 1
                continue

            # Avoid promoting repeated running-header leaks.
            if len(block_lines) == 1 and line_counts.get(block_lines[0], 0) >= 2:
                i += 1
                continue

            alpha = [c for c in joined if c.isalpha()]
            if not alpha:
                i += 1
                continue
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            titleish_words = sum(
                1 for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", joined)
                if (w[:1].isupper() or w.lower() in {"of", "and", "the", "in", "to", "for", "on", "with", "by"})
            )
            word_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’()\\-]*", joined)
            titleish_ratio = (titleish_words / len(word_tokens)) if word_tokens else 0.0

            headingish = False
            if CHAPTER_KEYWORDS.match(block_lines[0]) or _is_standalone_ordinal_marker(block_lines[0]):
                headingish = True
            elif upper_ratio >= 0.72 and len(word_tokens) <= 18:
                headingish = True
            elif titleish_ratio >= 0.7 and len(word_tokens) <= 12 and not re.match(r'^[a-z]', joined):
                headingish = True

            # Avoid converting short prose fragments that happen to be title-cased.
            if headingish and re.search(r'[.!?]["”’\')\]]*$', joined) and not joined.endswith(":"):
                headingish = False

            if not headingish:
                i += 1
                continue

            # Preserve source text while flattening wrapped heading lines into one markdown heading.
            out[block_idxs[0]] = f"#### {joined}"
            for bi in block_idxs[1:]:
                out[bi] = ""
            promoted_blocks.append(joined)
            i = j + 1

        if promoted_blocks:
            self._record_op(
                "deterministic_body_heading_promote",
                {
                    "chapter_title": chapter_title,
                    "count": len(promoted_blocks),
                    "sample": promoted_blocks[:12],
                },
            )
        return "\n".join(out)

    def _strip_leading_chapter_running_header(self, body_text: str, chapter_title: str) -> str:
        """
        Remove a short leaked running header at the start of a chapter body
        (e.g. `INDIAN SHIPPING.`, `MAHOMEDAN PERIOD`) when it is visually
        isolated and immediately followed by another heading/title line.
        """
        if not body_text.strip() or self.archival_mode:
            return body_text

        lines = body_text.split("\n")
        non_empty = [idx for idx, ln in enumerate(lines) if ln.strip()]
        if len(non_empty) < 2:
            return body_text
        first_idx = non_empty[0]
        first = lines[first_idx].strip()
        if first.startswith(("#", "|", ">", "[^")) or _looks_like_footnote_definition_line(first):
            return body_text
        if self._is_page_number_line(first):
            return body_text
        if len(first.split()) > 4 or len(first) > 50:
            # Allow a slightly longer library artifact line to be removed.
            if not self._is_library_artifact_line(first):
                return body_text
        if CHAPTER_KEYWORDS.match(first):
            return body_text

        alpha = [c for c in first if c.isalpha()]
        if not alpha:
            if not self._is_library_artifact_line(first):
                return body_text
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if upper_ratio < 0.8 and not self._is_library_artifact_line(first):
            return body_text

        # Must be visually isolated.
        if first_idx + 1 < len(lines) and lines[first_idx + 1].strip():
            return body_text

        second_idx = non_empty[1]
        second = lines[second_idx].strip()
        second_norm = _normalize_simple(second.rstrip(".:"))
        title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        second_headingish = (
            second.startswith("#")
            or bool(CHAPTER_KEYWORDS.match(second))
            or bool(title_norm and second_norm and second_norm in title_norm)
        )
        if self._is_library_artifact_line(first):
            lines[first_idx] = ""
            return "\n".join(lines)
        if not second_headingish:
            return body_text

        lines[first_idx] = ""
        return "\n".join(lines)

    def _strip_leading_split_chapter_label_title(self, body_text: str, chapter_title: str) -> str:
        """
        Remove chapter-opening duplicates split across two isolated lines:
        `CHAPTER II.` + `THE MOGUL PERIOD : ...`
        """
        if not body_text.strip():
            return body_text
        lines = body_text.split("\n")
        non_empty = [idx for idx, ln in enumerate(lines) if ln.strip()]
        if len(non_empty) < 2:
            return body_text
        title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        first_idx = non_empty[0]
        first = lines[first_idx].strip()
        if not re.match(r'(?i)^chapter\s+([ivxlcdm]+|\d+)\.?\s*$', first):
            return body_text
        # Allow one blank line between label and subtitle/title.
        second_idx = non_empty[1]
        second = lines[second_idx].strip()
        second_norm = _normalize_simple(second.rstrip(".:"))
        if not second_norm or not title_norm or second_norm not in title_norm:
            return body_text
        lines[first_idx] = ""
        lines[second_idx] = ""
        return "\n".join(lines)

    def _strip_leading_chapter_opening_wrapper_stack(self, body_text: str, chapter_title: str) -> str:
        """
        Reader-only cleanup for chapter-opening wrapper stacks that can survive
        earlier passes when lines are packed without blank separators, e.g.:
        - `INDIAN SHIPPING.` + `INTRODUCTION.`
        - `UNIV. OF` + `CALIFORNIA` + chapter title

        This pass is conservative and only inspects the first few non-empty lines.
        """
        if self.archival_mode or not body_text.strip():
            return body_text

        lines = body_text.split("\n")
        title_norm = _normalize_simple((chapter_title or "").rstrip(".:"))
        if not title_norm:
            return body_text

        def _non_empty_indexes() -> List[int]:
            return [i for i, ln in enumerate(lines) if ln.strip()]

        def _is_divider_line(s: str) -> bool:
            return bool(re.fullmatch(r'[\-—–_*=·•♦◈\s]{3,}', s))

        def _is_short_upperish(s: str) -> bool:
            if not s or s.startswith(("#", "|", ">", "[^")):
                return False
            if len(s) > 70 or len(s.split()) > 7:
                return False
            alpha = [c for c in s if c.isalpha()]
            if not alpha:
                return False
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            return upper_ratio >= 0.75

        def _is_exact_title_dup(s: str) -> bool:
            norm = _normalize_simple(s.rstrip(".:"))
            return bool(norm and norm == title_norm)

        def _is_headingish(s: str) -> bool:
            if not s:
                return False
            if s.startswith("#"):
                return True
            return bool(
                CHAPTER_KEYWORDS.match(s)
                or _is_standalone_ordinal_marker(s)
                or _is_exact_title_dup(s)
            )

        changed = False
        # Small bounded loop so we only peel obvious wrapper lines from the top.
        for _ in range(8):
            non_empty = _non_empty_indexes()
            if not non_empty:
                break
            top = non_empty[0]
            if top > 20:
                break
            first = lines[top].strip()
            if not first:
                break

            removable_simple = (
                self._is_page_number_line(first)
                or _is_divider_line(first)
                or self._is_library_artifact_line(first)
                or re.match(r'(?i)^univ\.?\s+of$', first)
                or re.match(r'(?i)^california\.?$', first)
            )
            if removable_simple:
                lines[top] = ""
                changed = True
                continue

            if _is_exact_title_dup(first):
                # Duplicate of chapter title under the emitted markdown chapter heading.
                lines[top] = ""
                changed = True
                continue

            if len(non_empty) >= 2:
                second = lines[non_empty[1]].strip()
                # Common leak: running header immediately followed by the real chapter title
                # or subheading without a blank separator.
                if (
                    _is_short_upperish(first)
                    and not CHAPTER_KEYWORDS.match(first)
                    and not _is_exact_title_dup(first)
                    and (_is_headingish(second) or _is_short_upperish(second))
                ):
                    lines[top] = ""
                    changed = True
                    continue

            # Stop once we hit likely real content.
            break

        return "\n".join(lines) if changed else body_text

    def _strip_leading_composite_title_component_lines(
        self,
        body_text: str,
        chapter_title: str,
    ) -> str:
        """
        Remove leading heading-like lines already represented in a composite
        chapter title rendered as the chapter markdown heading.

        Example:
        - title: "INTRODUCTION. I.-ISOLATION AND INTERCOURSE."
        - body starts with:
            INTRODUCTION.
            I.—ISOLATION AND INTERCOURSE.
        """
        if self.archival_mode or not body_text.strip() or not chapter_title.strip():
            return body_text

        title_norm = _normalize_simple(chapter_title.rstrip(".:"))
        if not title_norm:
            return body_text

        if not re.search(r'(?i)\b(book|part|chapter|introduction|preface|appendix|index)\b', chapter_title):
            return body_text

        lines = body_text.split("\n")
        removed_any = False
        non_empty_seen = 0

        for idx, line in enumerate(lines[:24]):
            s = line.strip()
            if not s:
                continue

            non_empty_seen += 1
            if non_empty_seen > 8:
                break

            if s.startswith(("#", "|", ">", "[^")):
                break
            if _looks_like_footnote_definition_line(s):
                break
            if re.match(r'^[a-z]', s):
                break
            if len(s) > 120 or len(s.split()) > 14:
                break

            s_norm = _normalize_simple(s.rstrip(".:"))
            if not s_norm:
                continue

            headingish = (
                _looks_like_standalone_chapter_heading_line(s)
                or _is_standalone_ordinal_marker(s)
                or bool(re.match(r'(?i)^[ivxlcdm]+[\.\-—–:]\s*\S+', s))
                or bool(re.search(r'(?i)\b(book|part|chapter|introduction|preface|appendix|index)\b', s))
            )
            if not headingish:
                break

            if len(s_norm) < 6 or not (s_norm == title_norm or s_norm in title_norm or title_norm in s_norm):
                break

            lines[idx] = ""
            removed_any = True

        if not removed_any:
            return body_text
        return "\n".join(lines).lstrip("\n")

    def _strip_leading_library_artifact_headings(self, body_text: str) -> str:
        """
        Remove leading markdown headings that are actually library artifacts,
        e.g. `#### UNIV. OF CALIFORNIA` promoted by a prior pass.
        """
        if not body_text.strip():
            return body_text
        lines = body_text.split("\n")
        non_empty = [idx for idx, ln in enumerate(lines) if ln.strip()]
        changed = False
        for idx in non_empty[:4]:
            s = lines[idx].strip()
            if not s.startswith("#"):
                break
            label = re.sub(r"^#{1,6}\s*", "", s).strip()
            if self._is_library_artifact_line(label):
                lines[idx] = ""
                changed = True
                continue
            break
        return "\n".join(lines) if changed else body_text

    def _is_reader_library_artifact_block_line(self, line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        label = re.sub(r"^#{1,6}\s*", "", s).strip()
        if self._is_library_artifact_line(label):
            return True
        if re.search(r'(?i)\bdate\s+due\b', label):
            return True
        if re.search(r"(?i)\brec['’]?\s*d\b", label):
            return True
        if re.search(r'(?i)\bld-?url\b', label):
            return True
        if re.search(r'(?i)\b(?:loan|non-?renewable|quarter\s+loan)\b', label):
            return True
        if re.search(r'(?i)\breturn\s+this\s+material\s+to\s+the\s+library\b', label):
            return True
        if re.search(r'(?i)\bfrom\s+which\s+it\s+was\s+borrowed\b', label):
            return True
        if re.search(r'(?i)\b(?:hilgard\s+avenue|los\s+angeles,\s*ca\b)\b', label):
            return True
        if re.search(r'(?i)\bform\s+l\d', label):
            return True
        return False

    def _is_reader_library_stampish_fragment(self, line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return True  # allow blanks inside removable clusters
        label = re.sub(r"^#{1,6}\s*", "", s).strip()
        if self._is_reader_library_artifact_block_line(s):
            return True
        if re.fullmatch(r'(?i)(?:date\s+due|srl|loan|2\s+week\s+loan(?:\s+\d+.*)?)', label):
            return True
        if re.fullmatch(r'(?i)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2}\s+\d{4}', label):
            return True
        if re.fullmatch(r"(?i)[a-z][a-z .,'’:-]{1,20}", label) and len(label.split()) <= 3:
            # OCR fragments like "Connel" inside a stamp block; only used via cluster expansion.
            return True
        if re.fullmatch(r"(?i)[A-Z0-9][A-Z0-9 .,'’:/-]{2,80}", label):
            # Stamp/date rows are often uppercase-heavy.
            alpha = [c for c in label if c.isalpha()]
            if alpha:
                upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
                if upper_ratio >= 0.7:
                    return True
        return False

    def _strip_reader_library_artifact_blocks(self, body_text: str, chapter_title: str) -> str:
        """
        Remove mid-book library slip/stamp blocks that survive page classification and
        get promoted into headings (e.g. DATE DUE / LOAN forms on scanned library copies).
        Reader-only, cluster-based, fail-conservative.
        """
        if self.archival_mode or not body_text.strip():
            return body_text

        lines = body_text.split("\n")
        nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
        if not nonempty:
            return body_text

        strong_idxs = [i for i in nonempty if self._is_reader_library_artifact_block_line(lines[i])]
        if not strong_idxs:
            return body_text

        to_remove: set = set()
        removed_samples: List[str] = []

        for seed in strong_idxs:
            start = seed
            end = seed

            # Expand upward through local stamp fragments/blanks.
            while start - 1 >= 0 and self._is_reader_library_stampish_fragment(lines[start - 1]):
                start -= 1

            # Expand downward through the artifact cluster.
            gap_budget = 1
            j = end + 1
            while j < len(lines):
                if self._is_reader_library_stampish_fragment(lines[j]):
                    end = j
                    gap_budget = 1
                    j += 1
                    continue
                if gap_budget > 0 and not lines[j].strip():
                    end = j
                    gap_budget -= 1
                    j += 1
                    continue
                break

            block_lines = [lines[k].strip() for k in range(start, end + 1) if lines[k].strip()]
            if not block_lines:
                continue

            artifactish_count = sum(
                1 for s in block_lines
                if self._is_reader_library_artifact_block_line(s) or self._is_reader_library_stampish_fragment(s)
            )
            if artifactish_count < 3:
                continue

            for k in range(start, end + 1):
                if lines[k].strip():
                    if len(removed_samples) < 20:
                        removed_samples.append(lines[k].strip())
                    to_remove.add(k)
                else:
                    # Drop blank lines inside removed block too, to avoid gaps.
                    to_remove.add(k)

        if not to_remove:
            return body_text

        out = [("" if i in to_remove else ln) for i, ln in enumerate(lines)]
        self._record_op(
            "reader_library_artifact_block_strip",
            {
                "chapter_title": chapter_title,
                "count": len([i for i in to_remove if lines[i].strip()]),
                "sample": removed_samples[:20],
            },
        )
        return "\n".join(out)

    def _is_reference_like_backmatter_chapter(self, chapter_title: str, text: str) -> bool:
        title = (chapter_title or "").strip()
        kind = _chapter_title_kind(title)
        if kind in {"index", "appendix"}:
            return True
        if re.search(r'(?i)\b(bibliography|references?|works cited|authorities)\b', title):
            return True
        # Content heuristic for untitled/poorly-titled index chapters.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) < 20:
            return False
        indexish = 0
        sample = lines[: min(120, len(lines))]
        for s in sample:
            if s.startswith(("#", "|", ">", "[^")):
                continue
            if re.search(r',\s*\d', s):
                indexish += 1
                continue
            if re.search(r'\b\d{1,3}(?:[-,]\d{1,3})+\b', s):
                indexish += 1
        return indexish >= max(8, len(sample) // 5)

    def _normalize_reader_backmatter_body(self, text: str, chapter_title: str) -> str:
        """
        Reader-only cleanup for backmatter/reference-heavy chapters (index,
        bibliography, appendices with list-like material). Keeps text exact;
        only applies structural promotion/removal of obvious non-content noise.
        """
        if self.archival_mode or not text.strip():
            return text
        if not self._is_reference_like_backmatter_chapter(chapter_title, text):
            return text

        lines = text.split("\n")
        out = list(lines)
        removed: List[str] = []
        promoted: List[str] = []

        def _next_nonempty_idx(start: int) -> Optional[int]:
            j = start + 1
            while j < len(out):
                if out[j].strip():
                    return j
                j += 1
            return None

        def _prev_nonempty_idx(start: int) -> Optional[int]:
            j = start - 1
            while j >= 0:
                if out[j].strip():
                    return j
                j -= 1
            return None

        # Pass 1: strip obvious leaked page/header markers and promote index letter dividers.
        for i, raw in enumerate(list(out)):
            s = raw.strip()
            if not s:
                continue
            if s.startswith(("#", "|", ">", "[^")):
                continue

            # Strong header/page-marker combos in backmatter.
            if _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
                removed.append(s)
                out[i] = ""
                continue
            if re.match(r'^\d+\s+[A-Z]\s+\d+$', s):
                removed.append(s)
                out[i] = ""
                continue

            # Promote isolated single-letter alphabetical dividers in indexes.
            if re.fullmatch(r'[A-Z]', s):
                prev_i = _prev_nonempty_idx(i)
                next_i = _next_nonempty_idx(i)
                if next_i is None:
                    continue
                next_line = out[next_i].strip()
                if next_line.startswith(("#", "|", ">", "[^")):
                    continue
                # Confirm this behaves like an index divider: next entry usually begins
                # with the same letter, and the line is visually isolated.
                same_letter_start = _normalize_simple(next_line).startswith(s.lower())
                isolated = (prev_i is None or not out[i - 1].strip()) and (i + 1 >= len(out) or not out[i + 1].strip())
                if same_letter_start and isolated:
                    out[i] = f"#### {s}."
                    promoted.append(s)
                continue

        # Pass 2: remove trailing printer colophon noise in reader backmatter if present.
        tail_nonempty = [idx for idx, ln in enumerate(out) if ln.strip()]
        if len(tail_nonempty) >= 2:
            last_nonempty = tail_nonempty[-1]
            # Find a trailing colophon start near the end (e.g. "LONDON :" / "PRINTED BY ...")
            colophon_start: Optional[int] = None
            for idx in reversed(tail_nonempty[-8:]):
                s = out[idx].strip().lower()
                if s.startswith("printed by") or re.match(r'^[a-z .-]+:\s*$', s):
                    colophon_start = idx
            if colophon_start is not None:
                tail_text = " ".join(
                    out[idx].strip().lower()
                    for idx in tail_nonempty
                    if idx >= colophon_start
                )
                if "printed by" in tail_text and ("london" in tail_text or "limited" in tail_text):
                    for idx in range(colophon_start, last_nonempty + 1):
                        s = out[idx].strip()
                        if s:
                            removed.append(s)
                        out[idx] = ""

        # Pass 3 (index-only): join wrapped index continuations that were split into
        # a new line/paragraph, e.g. page-number tails after a comma.
        if _chapter_title_kind(chapter_title) == "index":
            def _prev_nonempty(i0: int) -> Optional[int]:
                j = i0 - 1
                while j >= 0:
                    if out[j].strip():
                        return j
                    j -= 1
                return None

            def _looks_like_new_index_entry(s: str) -> bool:
                if not s or s.startswith(("#", "|", ">", "[^")):
                    return False
                return bool(re.match(r"^[A-ZĀĪŪṚṜḶḸṂṄÑṆŚṢ][^,]{0,80},\s*\d", s))

            def _looks_like_index_continuation(s: str) -> bool:
                if not s or s.startswith(("#", "|", ">", "[^")):
                    return False
                if _looks_like_footnote_definition_line(s):
                    return False
                return bool(re.match(r"^(?:\d|[ivxlcdm]+\b|[a-z(])", s, flags=re.IGNORECASE))

            for i, raw in enumerate(list(out)):
                s = raw.strip()
                if not _looks_like_index_continuation(s):
                    continue
                prev_i = _prev_nonempty(i)
                if prev_i is None:
                    continue
                prev_s = out[prev_i].rstrip()
                if not prev_s.strip().endswith(","):
                    continue
                if _looks_like_new_index_entry(s):
                    continue
                spacer = "" if s[:1] in ",.;:)]" else " "
                out[prev_i] = prev_s + spacer + s
                out[i] = ""
                removed.append(s)

        # Pass 4: compact excessive blank lines (common in indexes after cleanup).
        compact: List[str] = []
        blank_run = 0
        for ln in out:
            if ln.strip():
                blank_run = 0
                compact.append(ln)
                continue
            blank_run += 1
            if blank_run <= 2:
                compact.append("")

        result = "\n".join(compact)
        if removed or promoted:
            self._record_op(
                "backmatter_normalize",
                {
                    "chapter_title": chapter_title,
                    "removed": removed[:20],
                    "promoted_letter_dividers": promoted[:20],
                },
            )
        return result

    def _normalize_toc_text(self, text: str) -> str:
        """
        Convert machine-emitted TOC table JSON and dotted illustration lines
        into markdown tables.
        """
        def _join_wrapped(parts: List[str], nxt: str) -> List[str]:
            s = nxt.strip()
            if not s:
                return parts
            if not parts:
                return [s]
            prev = parts[-1]
            if prev.endswith('-'):
                parts[-1] = prev + s
            else:
                parts.append(s)
            return parts

        def _build_raw_chapter_toc_table(lines_in: List[str]) -> Optional[List[str]]:
            chapter_pat = re.compile(
                r'^\*?\s*CHAPTER\s+([IVXLCDM]+|\d+)(?:\s*[.\-—–]+\s*)?(.*)$',
                re.IGNORECASE,
            )
            page_tail_pat = re.compile(
                r'^(.*?)(?:\s*[.·•]+\s*|\s{2,})([0-9]{1,4}|[IVXLCDM]{1,8})\s*$',
                re.IGNORECASE,
            )

            rows: List[List[str]] = []
            pre_lines: List[str] = []
            post_lines: List[str] = []
            in_entries = False
            current_chapter: Optional[str] = None
            desc_parts: List[str] = []

            def _finalize_if_possible() -> bool:
                nonlocal current_chapter, desc_parts
                if not current_chapter or not desc_parts:
                    return False
                joined = " ".join(desc_parts).strip()
                m = page_tail_pat.match(joined)
                if not m:
                    return False
                body = re.sub(r'\s+', ' ', m.group(1)).strip(" .")
                page = m.group(2).upper() if m.group(2).isalpha() else m.group(2)
                if not body:
                    return False
                rows.append([current_chapter, body, page])
                current_chapter = None
                desc_parts = []
                return True

            for raw in lines_in:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    if in_entries and (current_chapter or desc_parts):
                        desc_parts = _join_wrapped(desc_parts, "")
                    if not in_entries:
                        pre_lines.append(line)
                    else:
                        post_lines.append(line)
                    continue

                if _looks_like_running_header_with_page_number(stripped) or _looks_like_page_header_combo(stripped):
                    # Running header leakage on TOC continuation pages.
                    continue
                if self._is_page_number_line(stripped):
                    continue
                if re.match(r'(?i)^intro(?:duction)?\.?\s+page$', stripped):
                    continue
                if stripped.upper() == "PAGE":
                    continue

                chap = chapter_pat.match(stripped.strip("*").strip())
                if chap:
                    in_entries = True
                    if current_chapter and desc_parts:
                        # Previous entry never reached a page token; preserve it as post text.
                        post_lines.append(f"{current_chapter} {' '.join(desc_parts)}".strip())
                        current_chapter = None
                        desc_parts = []

                    current_chapter = chap.group(1).upper()
                    rest = chap.group(2).strip()
                    if rest:
                        desc_parts = _join_wrapped(desc_parts, rest)
                        _finalize_if_possible()
                    continue

                if in_entries and current_chapter:
                    desc_parts = _join_wrapped(desc_parts, stripped)
                    _finalize_if_possible()
                    continue

                if not in_entries:
                    pre_lines.append(line)
                else:
                    post_lines.append(line)

            if current_chapter and desc_parts:
                # Keep incomplete tail as plain text instead of guessing.
                post_lines.append(f"{current_chapter} {' '.join(desc_parts)}".strip())

            if len(rows) < 3:
                return None

            table = _render_markdown_table([["Chapter", "Contents", "Page"]] + rows)
            out: List[str] = []
            # Promote contents heading for reader navigation.
            kept_pre = []
            saw_contents = False
            for ln in pre_lines:
                s = ln.strip()
                if re.match(r'(?i)^contents\.?$', s):
                    if not saw_contents:
                        kept_pre.append("### CONTENTS.")
                        saw_contents = True
                    continue
                kept_pre.append(ln)
            out.extend(kept_pre)
            if out and out[-1].strip():
                out.append("")
            out.append(table)
            if post_lines:
                if out and out[-1].strip():
                    out.append("")
                out.extend(post_lines)
            return out

        def _build_generic_contents_table(lines_in: List[str]) -> Optional[List[str]]:
            """
            Fallback parser for OCR contents pages with mixed BOOK/PART/CHAPTER entries
            and wrapped lines. Produces a simple Entry/Page table.
            """
            page_tail_pat = re.compile(
                r'^(.*?)(?:\s*[.·•]+\s*|\s{2,})([0-9]{1,4}|[IVXLCDM]{1,8})\s*$',
                re.IGNORECASE,
            )
            section_pat = re.compile(
                r'(?i)^(?:book|part|indexes?)\b'
            )

            out_prefix: List[str] = []
            saw_contents = False

            def _clean_text(s: str) -> str:
                s = re.sub(r'\s+', ' ', s).strip()
                s = re.sub(r'(?:\s*[.·•])+\s*$', '', s).strip()
                return s

            candidate_lines: List[StructuredTextLine] = []

            for idx, raw in enumerate(lines_in):
                s = raw.strip()
                if not s:
                    candidate_lines.append(StructuredTextLine(index=idx, raw=raw))
                    continue
                header_like_toc_entry = bool(
                    re.match(r'(?i)^(?:chapter|book|part)\s+([ivxlcdm]+|\d+)\b', s)
                )
                if (not header_like_toc_entry) and (
                    _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s)
                ):
                    continue
                if self._is_page_number_line(s):
                    continue
                if re.match(r'(?i)^intro(?:duction)?\.?\s+page$', s):
                    continue
                if s.upper() == "PAGE":
                    continue
                if re.fullmatch(r'[\-—–_·•◈\s]+', s):
                    continue
                if re.fullmatch(r'[a-z]', s):
                    continue

                if s.upper().startswith("CONTENTS"):
                    saw_contents = True
                    if not out_prefix or _normalize_simple(out_prefix[-1]) != _normalize_simple(s):
                        out_prefix.append("CONTENTS.")
                    continue

                # OCR tables are already handled above; do not reinterpret them here.
                if s.startswith("|"):
                    return None

                candidate_lines.append(StructuredTextLine(index=idx, raw=raw))

            def _ignore_line(s: str) -> bool:
                if not s:
                    return False
                header_like_toc_entry = bool(
                    re.match(r'(?i)^(?:chapter|book|part)\s+([ivxlcdm]+|\d+)\b', s)
                )
                if not header_like_toc_entry and (
                    _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s)
                ):
                    return True
                if self._is_page_number_line(s):
                    return True
                if re.match(r'(?i)^intro(?:duction)?\.?\s+page$', s):
                    return True
                if s.upper() == "PAGE":
                    return True
                if re.fullmatch(r'[\-—–_·•◈\s]+', s):
                    return True
                # OCR signature/footer fragments (e.g. trailing "b" after roman folio).
                if re.fullmatch(r'[a-z]', s):
                    return True
                return False

            def _normalize_page(page: str) -> str:
                page = str(page or "").strip()
                return page.upper() if page.isalpha() else page

            def _label_predicate(joined: str) -> bool:
                return bool(
                    (
                        section_pat.match(joined)
                        or re.match(r'(?i)^introduction\.?$', joined)
                    )
                    and not page_tail_pat.match(joined)
                )

            parsed_entries = _collect_wrapped_entry_page_rows(
                candidate_lines,
                page_tail_pattern=page_tail_pat,
                ignore_line=_ignore_line,
                normalize_page=_normalize_page,
                label_predicate=_label_predicate,
            )

            rows: List[List[str]] = []
            for ent in parsed_entries:
                entry_text = _clean_text(ent.text)
                page = ent.page or ""
                if not entry_text:
                    continue
                rows.append([entry_text, page])

            if not saw_contents or len(rows) < 4:
                return None

            out: List[str] = []
            prefix = out_prefix or ["CONTENTS."]
            promoted = False
            for ln in prefix:
                if re.match(r'(?i)^contents\.?$', ln.strip()) and not promoted:
                    out.append("### CONTENTS.")
                    promoted = True
                else:
                    out.append(ln)
            out.append("")
            out.append(_render_markdown_table([["Entry", "Page"]] + rows))
            return out

        def _build_illustrations_table(lines_in: List[str]) -> Optional[List[str]]:
            heading_idx = None
            heading_text = None
            for i, raw in enumerate(lines_in):
                s = raw.strip()
                if re.match(r'(?i)^(illustrations|list of (illustrations|plates|maps|paintings))\.?$', s):
                    heading_idx = i
                    heading_text = s
                    break
            if heading_idx is None:
                return None

            page_tail_pat = re.compile(
                r'^(.*?)(?:\s*[.·•]+\s*|\s{2,})(Frontis(?:-\s*|\s*)piece|[\[{(]?[0-9]{1,4})\s*$',
                re.IGNORECASE,
            )
            rows: List[List[str]] = []
            pre = lines_in[:heading_idx]
            pending: List[str] = []
            saw_rows = False

            def _flush_pending() -> bool:
                nonlocal pending, saw_rows
                if not pending:
                    return False
                joined = re.sub(r'\s+', ' ', " ".join(pending)).strip()
                if not joined:
                    pending = []
                    return False
                m = page_tail_pat.match(joined)
                if m:
                    caption = re.sub(r'(?:\s*[.·•])+\s*$', '', m.group(1)).strip()
                    page = m.group(2)
                    page = "Frontispiece" if re.match(r'(?i)^frontis', page) else re.sub(r'^[\[{(]+', '', page)
                    if caption:
                        rows.append([caption, page])
                        saw_rows = True
                    pending = []
                    return True
                m2 = re.match(r'^(.*\D)\s+(Frontis(?:-\s*|\s*)piece|[0-9]{1,4})\s*$', joined, re.IGNORECASE)
                if m2:
                    caption = m2.group(1).strip()
                    page = m2.group(2)
                    page = "Frontispiece" if re.match(r'(?i)^frontis', page) else re.sub(r'^[\[{(]+', '', page)
                    if caption:
                        rows.append([caption, page])
                        saw_rows = True
                    pending = []
                    return True
                # Group labels ending with ':' are useful and should be preserved.
                if joined.endswith(":"):
                    rows.append([joined, ""])
                    saw_rows = True
                    pending = []
                    return True
                return False

            for raw in lines_in[heading_idx + 1:]:
                s = raw.strip()
                if not s:
                    _flush_pending()
                    continue
                # Page continuation for prior row (e.g., "235" on next line after "235").
                # This must run before generic page-number stripping.
                if re.fullmatch(r'\d{1,4}', s) and rows and rows[-1][1]:
                    if s != rows[-1][1]:
                        rows[-1][1] = f"{rows[-1][1]}, {s}"
                    continue
                if _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
                    continue
                if self._is_page_number_line(s):
                    continue
                if s.upper() == "FACING PAGE":
                    continue
                if re.fullmatch(r'[ivxlcdm]+\s*(?:[_*].*)?$', s, re.IGNORECASE):
                    continue
                pending.append(s)
                _flush_pending()

            _flush_pending()
            if not saw_rows or len(rows) < 4:
                return None

            out: List[str] = []
            out.extend(pre)
            if out and out[-1].strip():
                out.append("")
            h = heading_text or "ILLUSTRATIONS."
            if not h.endswith((".", ":")):
                h = h + "."
            out.append(f"### {h}")
            out.append("")
            out.append(_render_markdown_table([["Illustration", "Facing Page"]] + rows))
            return out

        def _compact_toc_rows(rows: List[List[Any]]) -> List[List[str]]:
            compacted: List[List[str]] = []
            for row in rows:
                cells = [str(c).strip() for c in row if str(c).strip()]
                if not cells:
                    continue
                cleaned = [c for c in cells if not re.fullmatch(r'[.·•\s]+', c)]
                if not cleaned:
                    cleaned = cells
                if len(cleaned) >= 3:
                    left = re.sub(r'\s+', ' ', " ".join(cleaned[:-1])).strip()
                    left = re.sub(r'(?:\s*\.\.)+\s*$', '', left).strip()
                    compacted.append([left, cleaned[-1]])
                elif len(cleaned) == 2:
                    left = re.sub(r'(?:\s*\.\.)+\s*$', '', cleaned[0]).strip()
                    compacted.append([left, cleaned[1]])
                else:
                    compacted.append([cleaned[0]])
            return compacted

        def _ensure_entry_page_header(rows: List[List[str]]) -> List[List[str]]:
            if not rows:
                return rows
            first_text = " ".join(rows[0]).lower()
            if re.search(r'\b(page|chapter|appendix|entry)\b', first_text):
                return rows
            return [["Entry", "Page"]] + rows

        lines = text.split('\n')
        rewritten: List[str] = []

        for line in lines:
            stripped = line.strip()
            if TOC_JSON_TABLE_LINE.match(stripped):
                try:
                    data = json.loads(stripped)
                except Exception:
                    rewritten.append(line)
                    continue
                rows = data.get("table")
                if isinstance(rows, list) and rows and all(isinstance(r, list) for r in rows):
                    compact_rows = _compact_toc_rows(rows)
                    compact_rows = _ensure_entry_page_header(compact_rows)
                    rewritten.append(_render_markdown_table(compact_rows))
                    rewritten.append("")
                else:
                    rewritten.append(line)
                continue
            rewritten.append(line)

        # Compact OCR-style markdown tables used in contents pages:
        # keep semantic first/last cells and drop dot-leader filler columns.
        compacted_lines: List[str] = []
        i = 0
        while i < len(rewritten):
            if not rewritten[i].strip().startswith("|"):
                compacted_lines.append(rewritten[i])
                i += 1
                continue

            block: List[str] = []
            while i < len(rewritten) and rewritten[i].strip().startswith("|"):
                block.append(rewritten[i])
                i += 1

            parsed_rows: List[List[str]] = []
            has_dot_filler = False
            for row_line in block:
                cells = [c.strip() for c in row_line.strip().strip("|").split("|")]
                if all(re.fullmatch(r':?-{3,}:?', c or '') for c in cells):
                    continue
                parsed_rows.append(cells)
                if any(re.fullmatch(r'[.·•\s]+', c or '') for c in cells):
                    has_dot_filler = True

            if parsed_rows and has_dot_filler:
                compact_rows = _compact_toc_rows(parsed_rows)
                compact_rows = _ensure_entry_page_header(compact_rows)
                compacted_lines.append(_render_markdown_table(compact_rows))
            else:
                compacted_lines.extend(block)

        rewritten = compacted_lines

        out_lines: List[str] = []
        illustration_rows: List[List[str]] = []
        insert_at: Optional[int] = None

        for line in rewritten:
            stripped = line.strip()
            if stripped.upper() == "FACING PAGE":
                continue
            row_match = TOC_ILLUSTRATION_ROW.match(stripped)
            if row_match:
                page_raw = row_match.group(3).strip()
                page_norm = re.sub(r'[\s\-]+', '', page_raw.lower())
                if page_norm == "frontispiece":
                    page_value = "Frontispiece"
                elif re.match(r'^\d+$', page_raw):
                    page_value = page_raw
                else:
                    out_lines.append(line)
                    continue
                if insert_at is None:
                    insert_at = len(out_lines)
                caption = re.sub(r'(?:\s*\.\.)+\s*$', '', row_match.group(2).strip())
                caption = re.sub(r'\s+', ' ', caption).strip()
                illustration_rows.append([
                    row_match.group(1).strip(),
                    caption,
                    page_value,
                ])
                continue

            out_lines.append(line)

        if insert_at is not None and len(illustration_rows) >= 3:
            md_table = _render_markdown_table(
                [["No.", "Illustration", "Facing Page"]] + illustration_rows
            )
            table_lines = ["", md_table, ""]
            out_lines = out_lines[:insert_at] + table_lines + out_lines[insert_at:]

        parsed_illustrations = _build_illustrations_table(out_lines)
        if parsed_illustrations is not None:
            out_lines = parsed_illustrations
        else:
            # Fallback parser for raw OCR-style chapter contents pages (reader edition only).
            parsed_generic = _build_generic_contents_table(out_lines)
            if parsed_generic is not None:
                out_lines = parsed_generic
            else:
                parsed_toc = _build_raw_chapter_toc_table(out_lines)
                if parsed_toc is not None:
                    out_lines = parsed_toc

        return _repair_markdown_table_header_restarts("\n".join(out_lines))

    def _merge_adjacent_frontmatter_tables(self, text: str) -> str:
        """
        Merge adjacent markdown tables with identical headers in frontmatter.
        This consolidates multi-page TOCs and lists of illustrations/plates/maps.
        """
        if not text:
            return text
        text = _repair_markdown_table_header_restarts(text)
        blocks = _segment_structured_text_blocks(text)
        if not blocks:
            return text

        out_blocks: List[StructuredTextBlock] = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            parsed = _parse_markdown_table_block(block)
            if parsed is None:
                out_blocks.append(block)
                i += 1
                continue

            header, sep, rows = parsed
            merged_rows = list(rows)
            j = i + 1
            trailing_blank_blocks: List[StructuredTextBlock] = []

            while j < len(blocks):
                if blocks[j].kind == "blank":
                    trailing_blank_blocks.append(blocks[j])
                    j += 1
                    continue
                nxt = _parse_markdown_table_block(blocks[j])
                if nxt is None:
                    break
                n_header, n_sep, n_rows = nxt
                if (n_header, n_sep) != (header, sep):
                    break
                merged_rows.extend(n_rows)
                trailing_blank_blocks = []
                j += 1

            merged_table_lines = [header, sep] + merged_rows
            merged_block = StructuredTextBlock(
                kind="markdown_table",
                start=block.start,
                end=blocks[j - 1].end if j > i else block.end,
                lines=[StructuredTextLine(index=k, raw=ln) for k, ln in enumerate(merged_table_lines)],
            )
            out_blocks.append(merged_block)
            out_blocks.extend(trailing_blank_blocks)
            i = j

        return _repair_markdown_table_header_restarts(_render_structured_blocks(out_blocks))

    def _normalize_illustration_page_text(self, text: str) -> str:
        """
        Keep illustration pages concise by preserving only a single
        illustration marker/caption line.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        lines = [line for line in lines if not self._is_page_number_line(line)]
        lines = [line for line in lines if not self._is_library_artifact_line(line)]
        lines = [line for line in lines if not re.match(r'^\d{2,4}%$', line)]
        if not lines:
            return ""

        for line in lines:
            if line.startswith("[Illustration"):
                return _canonical_illustration_marker_from_caption(line)

        caption_lines: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # "To face p." / facing-page placement notes are useful in TOC/list
            # contexts but not as the primary in-text illustration caption.
            if re.search(r'(?i)\bto\s+face\s+p\.?', s):
                continue
            if re.search(r'(?i)\bfacing\s+page\b', s):
                continue
            # Strip light markdown wrappers from OCR output.
            s = s.strip("*_ ")
            s = s.strip("[] ")
            if not s:
                continue
            caption_lines.append(s)
            if len(caption_lines) >= 4:
                break

        if caption_lines:
            return _canonical_illustration_marker_from_caption(" ".join(caption_lines))

        # Fallback: preserve first remaining line verbatim inside marker.
        return _canonical_illustration_marker_from_caption(lines[0])

    def _record_op(self, op_type: str, payload: Dict[str, Any]):
        self.assembly_ops.append({"type": op_type, **payload})

    def _shadow_json_plan(
        self,
        prompt: str,
        *,
        op_name: str = "unknown",
        priority: str = "normal",
    ) -> Optional[Dict[str, Any]]:
        """
        Run shadow/challenger model for comparison only. Never applied directly.
        """
        if not self.shadow_planner:
            return None
        now = time.time()
        if now < self._shadow_circuit_open_until:
            self._shadow_skips_run += 1
            return None
        if self._shadow_calls_run >= self._shadow_max_calls_per_run:
            self._shadow_skips_run += 1
            return None

        priority_norm = (priority or "normal").strip().lower()
        if priority_norm not in {"low", "normal", "high"}:
            priority_norm = "normal"

        if priority_norm == "low":
            digest = hashlib.sha1(f"{op_name}\n{prompt}".encode("utf-8", "ignore")).digest()
            bucket = digest[0] / 255.0
            if bucket > self._shadow_low_priority_sample_rate:
                self._shadow_skips_run += 1
                return None

        since_last = now - self._shadow_last_call_ts
        if since_last < self._shadow_min_interval_sec:
            if priority_norm == "high":
                time.sleep(self._shadow_min_interval_sec - since_last)
            else:
                self._shadow_skips_run += 1
                return None

        def _is_rate_limit_error(msg: str) -> bool:
            m = (msg or "").lower()
            return (
                "http 429" in m
                or "rate limit" in m
                or "too many requests" in m
                or "quota" in m
            )

        def _open_circuit(reason: str, cooldown_sec: float):
            self._shadow_circuit_open_until = time.time() + max(30.0, float(cooldown_sec))
            self._shadow_circuit_notice_emitted = True
            self._record_op(
                "shadow_circuit_open",
                {
                    "op_name": op_name,
                    "reason": reason[:300],
                    "cooldown_sec": round(float(cooldown_sec), 1),
                },
            )
        try:
            self._shadow_last_call_ts = time.time()
            self._shadow_calls_run += 1
            response = self.shadow_planner.generate(
                prompt=prompt,
                json_mode=True,
            )
            if not getattr(response, "success", False):
                err = str(getattr(response, "error", "unknown"))
                self._shadow_consecutive_errors += 1
                self._record_op(
                    "shadow_llm_error",
                    {"op_name": op_name, "priority": priority_norm, "error": err[:500]},
                )
                if _is_rate_limit_error(err):
                    _open_circuit(err, self._shadow_cooldown_sec)
                elif self._shadow_consecutive_errors >= self._shadow_error_threshold:
                    _open_circuit(err, min(180.0, self._shadow_cooldown_sec))
                return None
            self._shadow_consecutive_errors = 0
            self._shadow_circuit_notice_emitted = False
            data = getattr(response, "json_data", None) or self._parse_json_response(getattr(response, "text", ""))
            return data if isinstance(data, dict) else None
        except Exception as e:
            err = str(e)
            self._shadow_consecutive_errors += 1
            self._record_op(
                "shadow_llm_error",
                {"op_name": op_name, "priority": priority_norm, "error": err[:500]},
            )
            if _is_rate_limit_error(err):
                _open_circuit(err, self._shadow_cooldown_sec)
            elif self._shadow_consecutive_errors >= self._shadow_error_threshold:
                _open_circuit(err, min(180.0, self._shadow_cooldown_sec))
            return None

    def _record_shadow_comparison(self, op_name: str, payload: Dict[str, Any]):
        self._record_op(f"shadow_compare_{op_name}", payload)

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # LLMs sometimes emit a valid JSON object followed by commentary or a second object.
        # Extract the first balanced JSON object conservatively.
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        data = json.loads(candidate)
                    except Exception:
                        return None
                    return data if isinstance(data, dict) else None
        return None

    def _llm_select_margin_candidates(
        self,
        page_number: int,
        text: str,
    ) -> Dict[str, Any]:
        """
        Ask LLM to choose which edge candidates are margin artifacts.
        Returns:
            {"remove_ids": set[int], "confidence": float, "reason": str}
        """
        lines = text.split('\n')
        non_empty_idxs = [i for i, line in enumerate(lines) if line.strip()]
        if not non_empty_idxs:
            return {"remove_ids": set(), "confidence": 0.0, "reason": "empty"}

        candidate_idxs: List[int] = []
        for idx in non_empty_idxs[:3]:
            candidate_idxs.append(idx)
        for idx in non_empty_idxs[-3:]:
            if idx not in candidate_idxs:
                candidate_idxs.append(idx)

        # Map candidate id -> line index
        cands: List[Tuple[int, int, str, str]] = []
        cid = 1
        for idx in candidate_idxs:
            position = "top" if idx in non_empty_idxs[:3] else "bottom"
            cands.append((cid, idx, lines[idx].strip(), position))
            cid += 1
        if not cands:
            return {"remove_ids": set(), "confidence": 0.0, "reason": "no candidates"}

        candidates_text = "\n".join(
            f"{cid}. ({position}) {line}"
            for cid, _, line, position in cands
        )
        prompt = MARGIN_OPS_PROMPT.format(candidates=candidates_text)

        if not (self.gemini or self.claude):
            return {"remove_ids": set(), "confidence": 0.0, "reason": "planner unavailable"}

        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="margin_candidates",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {"remove_ids": set(), "confidence": 0.0, "reason": "invalid json"}

            remove_candidate_ids = data.get("remove_candidate_ids", []) or []
            confidence = float(data.get("confidence", 0.0) or 0.0)
            reason = str(data.get("reason", "") or "")

            valid_ids = {cid for cid, _, _, _ in cands}
            requested_ids = set()
            for x in remove_candidate_ids:
                try:
                    xid = int(x)
                except Exception:
                    continue
                if xid in valid_ids:
                    requested_ids.add(xid)

            shadow = self._shadow_json_plan(prompt, op_name="margin_candidates", priority="low")
            if isinstance(shadow, dict):
                shadow_ids = set()
                for x in (shadow.get("remove_candidate_ids", []) or []):
                    try:
                        xid = int(x)
                    except Exception:
                        continue
                    if xid in valid_ids:
                        shadow_ids.add(xid)
                self._record_shadow_comparison(
                    "margin_candidates",
                    {
                        "page": page_number,
                        "primary_ids": sorted(requested_ids),
                        "shadow_ids": sorted(shadow_ids),
                        "agree": sorted(requested_ids) == sorted(shadow_ids),
                    },
                )

            return {"remove_ids": requested_ids, "confidence": confidence, "reason": reason}
        except Exception as e:
            return {"remove_ids": set(), "confidence": 0.0, "reason": str(e)}

    def _llm_select_top_stack_candidates(
        self,
        page_number: int,
        text: str,
        page_type: str,
    ) -> Dict[str, Any]:
        """
        LLM planner for top-of-page stack cleanup (reader edition only).
        Targets cases like:
        - UNIV. OF / CALIFORNIA / MAHOMEDAN PERIOD
        - repeated running labels that lack page numbers
        """
        if not (self.gemini or self.claude):
            return {"remove_ids": set(), "confidence": 0.0, "reason": "planner unavailable"}
        if page_type not in ("text", "frontmatter", "backmatter", "toc"):
            return {"remove_ids": set(), "confidence": 0.0, "reason": "page_type not eligible"}

        lines = text.split('\n')
        non_empty_idxs = [i for i, line in enumerate(lines) if line.strip()]
        if len(non_empty_idxs) < 2:
            return {"remove_ids": set(), "confidence": 0.0, "reason": "too few lines"}

        top_idxs = non_empty_idxs[:5]
        cands: List[Tuple[int, int, str]] = []
        for cid, idx in enumerate(top_idxs, start=1):
            cands.append((cid, idx, lines[idx].strip()))

        candidates_text = "\n".join(f"{cid}. {line}" for cid, _, line in cands)
        prompt = TOP_STACK_OPS_PROMPT.format(
            page_number=page_number,
            page_type=page_type,
            candidates=candidates_text,
        )

        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="top_stack",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return {"remove_ids": set(), "confidence": 0.0, "reason": "invalid json"}
            ids = data.get("remove_candidate_ids", []) or []
            confidence = float(data.get("confidence", 0.0) or 0.0)
            reason = str(data.get("reason", "") or "")
            valid_ids = {cid for cid, _, _ in cands}
            remove_ids = set()
            for x in ids:
                try:
                    xid = int(x)
                except Exception:
                    continue
                if xid in valid_ids:
                    remove_ids.add(xid)

            shadow = self._shadow_json_plan(prompt, op_name="top_stack", priority="low")
            if isinstance(shadow, dict):
                shadow_ids = set()
                for x in (shadow.get("remove_candidate_ids", []) or []):
                    try:
                        xid = int(x)
                    except Exception:
                        continue
                    if xid in valid_ids:
                        shadow_ids.add(xid)
                self._record_shadow_comparison(
                    "top_stack",
                    {
                        "page": page_number,
                        "page_type": page_type,
                        "primary_ids": sorted(remove_ids),
                        "shadow_ids": sorted(shadow_ids),
                        "agree": sorted(remove_ids) == sorted(shadow_ids),
                    },
                )
            return {"remove_ids": remove_ids, "confidence": confidence, "reason": reason}
        except Exception as e:
            return {"remove_ids": set(), "confidence": 0.0, "reason": str(e)}

    def _validate_top_stack_candidate(self, line: str, *, page_type: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if page_type in ("frontmatter", "toc"):
            norm = _normalize_simple(stripped.rstrip(".:"))
            if norm in FRONTMATTER_SECTION_HEADING_KEYS:
                return False
        if _looks_like_footnote_definition_line(stripped):
            return False
        if len(stripped) > 120:
            return False
        if self._is_page_number_line(stripped):
            return True
        if self._is_library_artifact_line(stripped):
            return True
        if re.match(r'(?i)^univ\.?\s+of$', stripped):
            return True
        if re.match(r'(?i)^california\.?$', stripped):
            return True
        # Protect chapter-start pages from losing legitimate opener headings.
        if page_type == "chapter_start":
            return False
        word_count = len(stripped.split())
        alpha = [c for c in stripped if c.isalpha()]
        if alpha:
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        else:
            upper_ratio = 0.0
        if word_count <= 5 and upper_ratio >= 0.9:
            return True
        return False

    def _apply_llm_top_stack_cleanup(self, text: str, *, page_number: int, page_type: str) -> str:
        if self.archival_mode:
            return text
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return text
        if not self.llm_assembly_ops or not (self.gemini or self.claude):
            return text
        # TOC pages depend on preserving section labels ("CONTENTS", "ILLUSTRATIONS")
        # for deterministic table parsing later in cleanup.
        if page_type == "toc":
            return text

        plan = self._llm_select_top_stack_candidates(page_number, text, page_type)
        remove_ids = plan.get("remove_ids", set()) or set()
        confidence = float(plan.get("confidence", 0.0) or 0.0)
        if confidence < 0.7 or not remove_ids:
            return text

        lines = text.split('\n')
        non_empty_idxs = [i for i, line in enumerate(lines) if line.strip()]
        top_idxs = non_empty_idxs[:5]
        cid_map = {cid: idx for cid, idx in enumerate(top_idxs, start=1)}
        removed_lines: List[str] = []
        for rid in sorted(remove_ids):
            idx = cid_map.get(rid)
            if idx is None:
                continue
            candidate = lines[idx]
            if not self._validate_top_stack_candidate(candidate, page_type=page_type):
                continue
            removed_lines.append(candidate.strip())
            lines[idx] = ""
        if removed_lines:
            self._record_op(
                "llm_top_stack_remove",
                {
                    "page": page_number,
                    "page_type": page_type,
                    "lines": removed_lines,
                    "confidence": confidence,
                    "reason": str(plan.get("reason", "") or ""),
                },
            )
        return "\n".join(lines)

    def _llm_plan_reader_heading_layout(self, chapters: List["Chapter"]) -> Dict[int, Dict[str, Any]]:
        """
        LLM planner for heading hierarchy/emission policy. Returns mapping by chapter index.
        Fail-closed to deterministic defaults.
        """
        defaults: Dict[int, Dict[str, Any]] = {
            idx: {
                "heading_level": _default_heading_level_for_title(ch.title),
                "emit_if_empty": _should_emit_structural_empty_chapter(ch.title),
                "kind": _chapter_title_kind(ch.title),
                "confidence": 0.0,
            }
            for idx, ch in enumerate(chapters)
        }
        if self.archival_mode:
            return defaults
        if self.reader_assembly_mode not in ("llm_guided", "llm_chunks"):
            return defaults
        if not (self.gemini or self.claude):
            return defaults

        chapter_lines = []
        for idx, ch in enumerate(chapters):
            chapter_lines.append(
                f"{idx}. title={ch.title!r}; empty={ch.is_empty}; pages={len(ch.pages)}"
            )
        prompt = HEADING_PLAN_PROMPT.format(chapters="\n".join(chapter_lines))

        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="heading_plan",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return defaults
            rows = data.get("chapters", [])
            if not isinstance(rows, list):
                return defaults

            out = dict(defaults)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    idx = int(row.get("chapter_index"))
                except Exception:
                    continue
                if idx not in out:
                    continue
                try:
                    level = int(row.get("heading_level", out[idx]["heading_level"]))
                except Exception:
                    level = out[idx]["heading_level"]
                if level < 2 or level > 4:
                    level = out[idx]["heading_level"]
                emit_if_empty = bool(row.get("emit_if_empty", out[idx]["emit_if_empty"]))
                kind = str(row.get("kind", out[idx]["kind"]) or out[idx]["kind"]).lower()
                if kind not in {"frontmatter", "book", "part", "chapter", "index", "appendix", "preface", "introduction", "other"}:
                    kind = out[idx]["kind"]
                # Safety: never suppress non-empty chapters
                if not chapters[idx].is_empty:
                    emit_if_empty = True
                # Safety: preserve structural wrappers even if LLM says no
                if _should_emit_structural_empty_chapter(chapters[idx].title):
                    emit_if_empty = True
                out[idx] = {
                    "heading_level": level,
                    "emit_if_empty": emit_if_empty,
                    "kind": kind,
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                }
            self._record_op(
                "llm_heading_plan",
                {
                    "chapters": len(chapters),
                    "mode": self.reader_assembly_mode,
                    "sample": {str(i): out[i] for i in list(out.keys())[:8]},
                },
            )

            shadow = self._shadow_json_plan(prompt, op_name="heading_plan", priority="high")
            if isinstance(shadow, dict) and isinstance(shadow.get("chapters"), list):
                shadow_rows = {}
                for row in shadow["chapters"]:
                    if not isinstance(row, dict):
                        continue
                    try:
                        idx = int(row.get("chapter_index"))
                    except Exception:
                        continue
                    if idx < 0 or idx >= len(chapters):
                        continue
                    shadow_rows[idx] = {
                        "heading_level": row.get("heading_level"),
                        "emit_if_empty": row.get("emit_if_empty"),
                        "kind": row.get("kind"),
                    }
                diffs = 0
                for idx in range(len(chapters)):
                    p = out.get(idx, {})
                    s = shadow_rows.get(idx)
                    if not s:
                        continue
                    if (
                        p.get("heading_level") != s.get("heading_level")
                        or bool(p.get("emit_if_empty")) != bool(s.get("emit_if_empty"))
                        or str(p.get("kind")) != str(s.get("kind")).lower()
                    ):
                        diffs += 1
                self._record_shadow_comparison(
                    "heading_plan",
                    {
                        "chapters": len(chapters),
                        "shadow_rows": len(shadow_rows),
                        "diff_count": diffs,
                    },
                )
            return out
        except Exception:
            return defaults

    def _validate_margin_candidate(self, line: str) -> bool:
        """
        Strong safety validation before removing any line by LLM recommendation.
        """
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith('[^'):
            return False
        if len(stripped) > 80:
            return False
        word_count = len(stripped.split())
        if word_count > 12:
            return False
        if self._is_page_number_line(stripped):
            return True
        # Never allow LLM margin cleanup to remove chapter/section labels.
        if CHAPTER_KEYWORDS.match(stripped):
            return False
        if stripped == stripped.upper() and word_count <= 10:
            return True
        # A short title-like line without terminal punctuation can be margin/header.
        if word_count <= 8 and not re.search(r'[.!?]["”’\')\]]*$', stripped):
            return True
        return False

    def _cleanup_page_text(
        self,
        text: str,
        page_number: int,
        page_type: str,
        margin_rules: Dict[str, set],
    ) -> str:
        """
        Clean a page with deterministic + LLM-validated edge cleanup.
        """
        if self.archival_mode:
            # Archival master should preserve visible text exactly as extracted
            # (except blank-page commentary already filtered upstream).
            return text

        # Frequency-based margin stripping is useful on body pages but too risky on
        # frontmatter, where repeated title/imprint lines are common and meaningful.
        if page_type in ("text", "chapter_start", "backmatter"):
            cleaned = self._strip_detected_margins(text, margin_rules)
        else:
            cleaned = text
        cleaned = self._strip_top_running_header(cleaned, page_type)
        cleaned = self._strip_top_page_header_stack(cleaned, page_type)
        cleaned = self._apply_llm_top_stack_cleanup(
            cleaned,
            page_number=page_number,
            page_type=page_type,
        )
        cleaned, removed_inline_signatures = self._strip_inline_signature_prefixes(cleaned, page_type)
        if removed_inline_signatures:
            self._record_op(
                "inline_signature_strip",
                {"page": page_number, "page_type": page_type, "prefixes": removed_inline_signatures[:10]},
            )
        cleaned, removed_artifacts = self._strip_library_artifacts(cleaned, page_type)
        if removed_artifacts:
            self._record_op(
                "artifact_strip",
                {"page": page_number, "page_type": page_type, "lines": removed_artifacts[:10]},
            )

        cleaned, removed_scan_noise = self._strip_reader_frontmatter_scan_noise(cleaned, page_type)
        if removed_scan_noise:
            self._record_op(
                "frontmatter_scan_noise_strip",
                {"page": page_number, "page_type": page_type, "lines": removed_scan_noise[:10]},
            )
            # Roman numerals/page markers often sit above a running header; after removing them,
            # re-run top header stripping once to catch the newly exposed label.
            cleaned = self._strip_top_running_header(cleaned, page_type)
            cleaned = self._strip_top_page_header_stack(cleaned, page_type)
            cleaned = self._apply_llm_top_stack_cleanup(
                cleaned,
                page_number=page_number,
                page_type=page_type,
            )

        cleaned, removed_checkout = self._strip_reader_library_checkout_page(cleaned, page_type)
        if removed_checkout:
            self._record_op(
                "library_checkout_strip",
                {"page": page_number, "page_type": page_type, "lines": removed_checkout[:10]},
            )
            if not cleaned.strip():
                return ""

        if page_type == "illustration":
            normalized = self._normalize_illustration_page_text(cleaned)
            if normalized and normalized != cleaned.strip():
                self._record_op(
                    "illustration_normalize",
                    {"page": page_number, "line": normalized},
                )
            return normalized
        if page_type == "toc":
            normalized_toc = self._normalize_toc_text(cleaned)
            if normalized_toc != cleaned:
                self._record_op(
                    "toc_normalize",
                    {"page": page_number},
                )
            return normalized_toc

        if not self.llm_assembly_ops or not (self.gemini or self.claude):
            return cleaned
        if page_type != "text":
            return cleaned

        llm_ops = self._llm_select_margin_candidates(page_number, cleaned)
        remove_ids: set = llm_ops.get("remove_ids", set())
        confidence = float(llm_ops.get("confidence", 0.0) or 0.0)
        reason = llm_ops.get("reason", "")
        if confidence < 0.7 or not remove_ids:
            return cleaned

        lines = cleaned.split('\n')
        non_empty_idxs = [i for i, line in enumerate(lines) if line.strip()]
        candidate_idxs: List[int] = []
        for idx in non_empty_idxs[:3]:
            candidate_idxs.append(idx)
        for idx in non_empty_idxs[-3:]:
            if idx not in candidate_idxs:
                candidate_idxs.append(idx)

        cid_map: Dict[int, int] = {}
        cid = 1
        for idx in candidate_idxs:
            cid_map[cid] = idx
            cid += 1

        removed_lines: List[str] = []
        for rid in sorted(remove_ids):
            idx = cid_map.get(rid)
            if idx is None:
                continue
            line = lines[idx]
            if not self._validate_margin_candidate(line):
                continue
            removed_lines.append(line.strip())
            lines[idx] = ""

        if removed_lines:
            self._record_op(
                "llm_margin_remove",
                {
                    "page": page_number,
                    "lines": removed_lines,
                    "confidence": confidence,
                    "reason": reason,
                },
            )
        return '\n'.join(lines)

    def _llm_should_join_boundary(
        self,
        prev_page: int,
        next_page: int,
        prev_tail: str,
        next_head: str,
    ) -> Tuple[bool, str, float, str]:
        """
        Ask LLM if boundary should be joined.
        Returns (should_join, join_mode, confidence, reason).
        """
        if not self.llm_assembly_ops or not (self.gemini or self.claude):
            return False, "auto", 0.0, "llm disabled"
        prompt = BOUNDARY_OPS_PROMPT.format(
            prev_page=prev_page,
            next_page=next_page,
            prev_tail=prev_tail,
            next_head=next_head,
        )
        try:
            data, _planner_ms, _planner_provider = self._planner_json_generate(
                op_name="boundary_join",
                prompt=prompt,
                thinking=getattr(self.gemini, "default_thinking", None) if self.gemini else None,
            )
            if not isinstance(data, dict):
                return False, "auto", 0.0, "invalid json"
            should_join = bool(data.get("should_join", False))
            join_mode = str(data.get("join_mode", "auto") or "auto").strip().lower()
            if join_mode in {"join_with_space"}:
                join_mode = "space"
            elif join_mode in {"join_without_space"}:
                join_mode = "nospace"
            elif join_mode in {"dehyphenate"}:
                join_mode = "drop_hyphen"
            if join_mode not in {"auto", "keep_hyphen", "drop_hyphen", "space", "nospace"}:
                join_mode = "auto"
            confidence = float(data.get("confidence", 0.0) or 0.0)
            reason = str(data.get("reason", "") or "")
            if not should_join:
                join_mode = "auto"
            return should_join, join_mode, confidence, reason
        except Exception as e:
            return False, "auto", 0.0, str(e)

    def _join_cross_page_words(
        self,
        extractions: List[PageExtraction],
        page_types: Dict[int, str],
    ) -> Dict[Tuple[int, int], str]:
        """
        Use LLM to detect likely page-boundary joins.

        Returns:
            Dict[(prev_page, next_page)] -> join_mode for forced boundary merges.
        """
        # Build ordered list of non-blank extractions
        content_pages = [
            ext for ext in extractions
            if page_types.get(ext.page_number) not in ("blank", "illustration")
            and ext.text.strip()
        ]

        join_overrides: Dict[Tuple[int, int], str] = {}
        joins_made = 0
        for i in range(len(content_pages) - 1):
            prev = content_pages[i]
            nxt = content_pages[i + 1]
            if page_types.get(nxt.page_number) == "chapter_start":
                continue

            # Quick check: does the previous page end with a hyphen or mid-word?
            prev_text = prev.text.rstrip()
            if not prev_text:
                continue
            last_line = prev_text.split('\n')[-1].rstrip()
            likely_continuation = (
                last_line.endswith('-') or
                (last_line and last_line[-1].isalpha() and last_line[-1] not in '.!?;:"\')')
            )
            if not likely_continuation:
                continue

            # Get tail/head window for boundary analysis
            prev_tail = '\n'.join(prev_text.split('\n')[-3:])
            next_head = '\n'.join(nxt.text.split('\n')[:3])

            try:
                should_join, join_mode, confidence, reason = self._llm_should_join_boundary(
                    prev_page=prev.page_number,
                    next_page=nxt.page_number,
                    prev_tail=prev_tail,
                    next_head=next_head,
                )
                if should_join and confidence >= 0.65:
                    join_overrides[(prev.page_number, nxt.page_number)] = join_mode or "auto"
                    joins_made += 1
                    self._record_op(
                        "llm_boundary_join",
                        {
                            "prev_page": prev.page_number,
                            "next_page": nxt.page_number,
                            "join_mode": join_mode or "auto",
                            "confidence": confidence,
                            "reason": reason,
                        },
                    )
                    self._log(
                        f"Boundary join approved for pages {prev.page_number}-{nxt.page_number} "
                        f"(confidence={confidence:.2f})"
                    )
            except Exception as e:
                self._log(f"Word join failed for pages {prev.page_number}-{nxt.page_number}: {e}")

        if joins_made:
            self._log(f"Cross-page word joins: {joins_made}")
        return join_overrides

    def _group_into_chapters(
        self,
        extractions: List[PageExtraction],
        page_types: Dict[int, str],
        join_overrides: Optional[Any] = None,
        margin_rules: Optional[Dict[str, set]] = None,
    ) -> List[Chapter]:
        """Group pages into chapters based on detected boundaries."""
        if isinstance(join_overrides, set):
            join_override_modes: Dict[Tuple[int, int], str] = {tuple(k): "auto" for k in join_overrides}
        elif isinstance(join_overrides, dict):
            join_override_modes = {}
            for k, v in join_overrides.items():
                try:
                    if not (isinstance(k, tuple) and len(k) == 2):
                        continue
                    pair = (int(k[0]), int(k[1]))
                except Exception:
                    continue
                join_override_modes[pair] = str(v or "auto")
        else:
            join_override_modes = {}
        margin_rules = margin_rules or {"headers": set(), "footers": set()}

        chapters: List[Chapter] = []
        current_pages: List[int] = []
        current_body_parts: List[str] = []
        current_footnotes: List[Dict[str, str]] = []
        current_title = "Front Matter"
        chapter_num = 0
        previous_text_page: Optional[int] = None
        previous_text_part_idx: Optional[int] = None
        previous_text_type: Optional[str] = None

        total_pages_in_run = len(extractions)
        for ext in extractions:
            if self.verbose and (ext.page_number == 1 or ext.page_number % 25 == 0 or ext.page_number == total_pages_in_run):
                self._log(f"Grouping page {ext.page_number}/{total_pages_in_run}...")
            ptype = page_types.get(ext.page_number, "text")

            ext_extracted_type = str(getattr(ext, "page_type", "") or "").strip().lower()
            if ptype == "text" and ext_extracted_type == "text":
                rescue_heading = (ext.chapter_heading or "").strip()
                rescue_heading_norm = _normalize_simple(rescue_heading)
                current_title_norm = _normalize_simple(current_title)
                overlaps_current = bool(
                    current_body_parts
                    and rescue_heading_norm
                    and current_title_norm
                    and (
                        rescue_heading_norm == current_title_norm
                        or rescue_heading_norm in current_title_norm
                        or current_title_norm in rescue_heading_norm
                    )
                )
                if (not overlaps_current) and self._should_promote_text_page_to_chapter_start(ext):
                    ptype = "chapter_start"
                    self._record_op(
                        "chapter_start_rescue",
                        {"page": ext.page_number, "heading": rescue_heading[:160]},
                    )

            if ptype == "chapter_start":
                # Guard: identical repeated title on a continuation page is usually a
                # running header, not a true chapter boundary.
                candidate_title = (ext.chapter_heading or "").strip()
                if not candidate_title:
                    for line in ext.text.split('\n'):
                        stripped = line.strip().lstrip('#').strip()
                        if stripped:
                            candidate_title = stripped[:80]
                            break
                if candidate_title and current_body_parts:
                    cand_norm = _normalize_simple(candidate_title)
                    cur_norm = _normalize_simple(current_title)
                    overlaps_current = bool(
                        cand_norm and cur_norm and (
                            cand_norm == cur_norm or
                            cand_norm in cur_norm or
                            cur_norm in cand_norm
                        )
                    )
                    ext_extracted_type = str(getattr(ext, "page_type", "") or "").strip().lower()
                    looks_rescued_text_opener = (
                        ext_extracted_type == "text"
                        and self._should_promote_text_page_to_chapter_start(ext)
                    )
                    strong_boundary_overlap = bool(
                        re.search(r'(?i)\b(chapter|book|part|appendix|conclusion)\b', candidate_title)
                    )
                    if overlaps_current and not (looks_rescued_text_opener and strong_boundary_overlap):
                        ptype = "text"

            if ptype == "blank":
                current_pages.append(ext.page_number)
                # Preserve text boundary context across inserted blank pages.
                # Many scanned books interleave blank sheets between continued text.
                continue

            if ptype == "illustration":
                current_pages.append(ext.page_number)
                cleaned = clean_page(ext.text, archival_mode=self.archival_mode)
                cleaned = self._cleanup_page_text(
                    cleaned,
                    page_number=ext.page_number,
                    page_type=ptype,
                    margin_rules=margin_rules,
                )
                cleaned = self._strip_contextual_page_top_stack(
                    cleaned,
                    chapter_title=current_title,
                    page_type=ptype,
                    is_chapter_start=False,
                )
                body_text, footnotes = extract_footnotes(
                    cleaned,
                    remove_from_body=not self.archival_mode,
                )
                body_text, footnotes = _remap_colliding_page_footnote_ids(
                    body_text,
                    footnotes,
                    current_footnotes,
                    ext.page_number,
                )
                if body_text.strip():
                    current_body_parts.append(body_text.strip())
                current_footnotes.extend(footnotes)
                # Preserve previous_text_page so text can continue across illustration inserts.
                continue

            if ptype == "chapter_start":
                # Save previous chapter
                if current_body_parts or current_pages:
                    chapters.append(Chapter(
                        number=chapter_num,
                        title=current_title,
                        pages=current_pages[:],
                        body='\n\n'.join(current_body_parts),
                        footnotes=current_footnotes[:],
                    ))

                # Start new chapter
                chapter_num += 1
                current_pages = [ext.page_number]
                current_footnotes = []
                previous_text_page = None
                previous_text_part_idx = None
                previous_text_type = None

                # Determine chapter title
                if ext.chapter_heading:
                    current_title = ext.chapter_heading
                else:
                    # Use first non-blank line
                    for line in ext.text.split('\n'):
                        stripped = line.strip().lstrip('#').strip()
                        if stripped:
                            current_title = stripped[:80]
                            break
                    else:
                        current_title = f"Chapter {chapter_num}"

                # Extract footnotes from this page and add remaining as body
                cleaned = clean_page(ext.text, archival_mode=self.archival_mode)
                cleaned = self._cleanup_page_text(
                    cleaned,
                    page_number=ext.page_number,
                    page_type=ptype,
                    margin_rules=margin_rules,
                )
                cleaned = self._strip_contextual_page_top_stack(
                    cleaned,
                    chapter_title=current_title,
                    page_type=ptype,
                    is_chapter_start=True,
                )
                body_text, footnotes = extract_footnotes(
                    cleaned,
                    remove_from_body=not self.archival_mode,
                )
                body_text, footnotes = _remap_colliding_page_footnote_ids(
                    body_text,
                    footnotes,
                    current_footnotes,
                    ext.page_number,
                )
                if not self.archival_mode:
                    body_text = self._strip_chapter_heading_lines(body_text, current_title)
                current_body_parts = [body_text] if body_text.strip() else []
                if body_text.strip():
                    previous_text_page = ext.page_number
                    previous_text_part_idx = len(current_body_parts) - 1
                    previous_text_type = ptype
                current_footnotes.extend(footnotes)

            else:
                # Regular page — append to current chapter
                current_pages.append(ext.page_number)
                cleaned = clean_page(ext.text, archival_mode=self.archival_mode)
                cleaned = self._cleanup_page_text(
                    cleaned,
                    page_number=ext.page_number,
                    page_type=ptype,
                    margin_rules=margin_rules,
                )
                cleaned = self._strip_contextual_page_top_stack(
                    cleaned,
                    chapter_title=current_title,
                    page_type=ptype,
                    is_chapter_start=False,
                )
                body_text, footnotes = extract_footnotes(
                    cleaned,
                    remove_from_body=not self.archival_mode,
                )
                body_text, footnotes = _remap_colliding_page_footnote_ids(
                    body_text,
                    footnotes,
                    current_footnotes,
                    ext.page_number,
                )
                if not self.archival_mode:
                    body_text = self._strip_repeated_continuation_chapter_header(body_text, current_title)
                if body_text.strip():
                    prose_merge_types = {"text", "frontmatter", "backmatter"}
                    allow_boundary_merge = (
                        ptype in prose_merge_types and
                        previous_text_type in (prose_merge_types | {"chapter_start"})
                    )
                    if (current_body_parts and previous_text_page is not None
                            and previous_text_part_idx is not None
                            and 0 <= previous_text_part_idx < len(current_body_parts)
                            and allow_boundary_merge):
                        force_key = (previous_text_page, ext.page_number)
                        forced_mode = join_override_modes.get(force_key)
                        force_merge = force_key in join_override_modes
                        merged_text, did_merge = _merge_page_boundary(
                            current_body_parts[previous_text_part_idx],
                            body_text,
                            force=force_merge,
                            force_mode=forced_mode,
                        )
                        if did_merge:
                            current_body_parts[previous_text_part_idx] = merged_text
                        else:
                            current_body_parts.append(body_text)
                            previous_text_part_idx = len(current_body_parts) - 1
                    else:
                        current_body_parts.append(body_text)
                        previous_text_part_idx = len(current_body_parts) - 1
                    previous_text_page = ext.page_number
                    previous_text_type = ptype
                current_footnotes.extend(footnotes)

        # Save last chapter
        if current_body_parts or current_pages:
            chapters.append(Chapter(
                number=chapter_num,
                title=current_title,
                pages=current_pages[:],
                body='\n\n'.join(current_body_parts),
                footnotes=current_footnotes[:],
            ))

        return chapters


# ─── QA Report ────────────────────────────────────────────────────────────────

def generate_qa_report(
    book: AssembledBook,
    extractions: List[PageExtraction],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a deterministic QA report after assembly.

    Checks:
    - Word count per chapter
    - Footnote ref/def integrity (orphaned refs, missing defs)
    - Indic script presence
    - Blank page summary
    - Pages flagged for review

    Args:
        book: The assembled book
        extractions: Original page extractions
        output_path: Optional path to write JSON report

    Returns:
        QA report as a dictionary
    """
    page_type_counts = Counter((ext.page_type or "unknown") for ext in extractions)
    blank_pages = [ext.page_number for ext in extractions if ext.page_type == "blank"]
    fallback_pages = sorted(
        ext.page_number
        for ext in extractions
        if "fallback" in str(getattr(ext, "extraction_model", "") or "").lower()
    )
    failed_pages = [
        {
            "page": ext.page_number,
            "page_type": ext.page_type,
            "error": ext.error or "Extraction failed",
            "model": ext.extraction_model,
            "warnings": list(ext.warnings or []),
        }
        for ext in extractions
        if ext.page_type == "error" or not ext.success
    ]
    review_pages = [
        {
            "page": ext.page_number,
            "page_type": ext.page_type,
            "warnings": list(ext.warnings or []),
        }
        for ext in extractions
        if ext.warnings
    ]
    low_confidence_pages = [
        {
            "page": ext.page_number,
            "page_type": ext.page_type,
            "confidence": round(float(ext.confidence or 0.0), 3),
        }
        for ext in extractions
        if ext.success and ext.page_type != "blank" and float(ext.confidence or 0.0) < 0.65
    ]
    empty_nonblank_pages = sorted(
        ext.page_number
        for ext in extractions
        if not (ext.text or "").strip() and ext.page_type != "blank"
    )

    report = {
        "model": MODEL,
        "title": book.title,
        "author": book.author,
        "total_pages": book.total_pages,
        "total_words": book.total_words,
        "total_chapters": len(book.chapters),
        "avg_fidelity": round(book.avg_fidelity, 1),
        "chapters": [],
        "footnote_integrity": {},
        "blank_pages": {
            "count": len(blank_pages),
            "pages": blank_pages,
        },
        "extraction_audit": {
            "page_type_counts": dict(sorted(page_type_counts.items())),
            "summary": {
                "successful_pages": sum(1 for ext in extractions if ext.success),
                "failed_pages": len(failed_pages),
                "blank_pages": len(blank_pages),
                "fallback_pages": len(fallback_pages),
                "review_pages": len(review_pages),
                "low_confidence_pages": len(low_confidence_pages),
                "empty_nonblank_pages": len(empty_nonblank_pages),
            },
            "blank_pages": blank_pages,
            "failed_pages": failed_pages,
            "fallback_pages": fallback_pages,
            "review_pages": review_pages,
            "low_confidence_pages": low_confidence_pages,
            "empty_nonblank_pages": empty_nonblank_pages,
        },
        "indic_script": {
            "detected": False,
            "chapters": [],
        },
        "pages_for_review": book.pages_needing_review,
        "warnings": book.warnings,
        "issues": [],
        "assembly_ops": {
            "count": len(getattr(book, "assembly_ops", []) or []),
            "sample": (getattr(book, "assembly_ops", []) or [])[:20],
        },
    }

    if failed_pages:
        report["issues"].append(
            "Extraction failures on pages: "
            + ", ".join(str(item["page"]) for item in failed_pages[:100])
        )
    if empty_nonblank_pages:
        report["issues"].append(
            "Pages with empty output that are not true blanks: "
            + ", ".join(str(page) for page in empty_nonblank_pages[:100])
        )

    # Per-chapter analysis
    all_refs = set()
    all_def_counts = Counter()  # Use Counter to detect duplicates (Fix #5)

    for chapter in book.chapters:
        chapter_info = {
            "number": chapter.number,
            "title": chapter.title,
            "pages": chapter.pages,
            "word_count": chapter.word_count,
            "footnote_count": len(chapter.footnotes),
            "has_indic_script": chapter.has_indic_script,
        }
        report["chapters"].append(chapter_info)

        if chapter.has_indic_script:
            report["indic_script"]["detected"] = True
            report["indic_script"]["chapters"].append(chapter.number)

        # Find footnote references in body text
        refs_in_body = set(re.findall(r'\[\^([\w*†‡§\d-]+)\]', chapter.body))
        refs_in_body.update(re.findall(r'\$\^\{?([\w*†‡§\d-]+)\}?\$', chapter.body))
        defs_in_chapter = Counter(fn["id"] for fn in chapter.footnotes)

        all_refs.update(refs_in_body)
        all_def_counts.update(defs_in_chapter)

        # Check for orphaned refs (referenced but not defined)
        orphaned = refs_in_body - set(defs_in_chapter.keys())
        if orphaned:
            report["issues"].append(
                f"Chapter {chapter.number} '{chapter.title}': "
                f"Orphaned footnote refs: {sorted(orphaned)}"
            )

        # Check for unreferenced defs
        unreferenced = set(defs_in_chapter.keys()) - refs_in_body
        if unreferenced:
            report["issues"].append(
                f"Chapter {chapter.number} '{chapter.title}': "
                f"Unreferenced footnote defs: {sorted(unreferenced)}"
            )

        # Check for duplicate defs within chapter (Fix #5)
        duplicates = {fid: count for fid, count in defs_in_chapter.items() if count > 1}
        if duplicates:
            report["issues"].append(
                f"Chapter {chapter.number} '{chapter.title}': "
                f"Duplicate footnote defs: {dict(duplicates)}"
            )

    all_defs = set(all_def_counts.keys())
    global_duplicates = {fid: count for fid, count in all_def_counts.items() if count > 1}

    report["footnote_integrity"] = {
        "total_refs": len(all_refs),
        "total_defs": len(all_defs),
        "total_def_occurrences": sum(all_def_counts.values()),
        "duplicate_defs": global_duplicates,
        "orphaned_refs": sorted(all_refs - all_defs),
        "unreferenced_defs": sorted(all_defs - all_refs),
        "ok": all_refs == all_defs and not global_duplicates,
    }

    # Write report if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    return report


def _page_list_to_ranges(pages: List[int]) -> str:
    if not pages:
        return "None"
    ordered = sorted(set(int(p) for p in pages))
    ranges: List[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def write_verification_report_markdown(
    qa_report: Dict[str, Any],
    output_path: str,
) -> Path:
    audit = qa_report.get("extraction_audit", {})
    summary = audit.get("summary", {})
    failed_pages = audit.get("failed_pages", [])
    review_pages = audit.get("review_pages", [])
    low_conf_pages = audit.get("low_confidence_pages", [])
    page_type_counts = audit.get("page_type_counts", {})

    lines = [
        "# Verification Report",
        "",
        f"- Title: {qa_report.get('title', 'Unknown')}",
        f"- Total pages: {qa_report.get('total_pages', 0)}",
        f"- Total words: {qa_report.get('total_words', 0):,}",
        f"- Chapters: {qa_report.get('total_chapters', 0)}",
        f"- True blank pages: {summary.get('blank_pages', 0)}",
        f"- Extraction failures: {summary.get('failed_pages', 0)}",
        f"- Fallback pages: {summary.get('fallback_pages', 0)}",
        f"- Review pages: {summary.get('review_pages', 0)}",
        f"- Low-confidence pages: {summary.get('low_confidence_pages', 0)}",
        "",
        "## Blank Pages",
        "",
        _page_list_to_ranges(audit.get("blank_pages", [])),
        "",
        "## Extraction Failures",
        "",
    ]

    if failed_pages:
        for item in failed_pages:
            lines.append(
                f"- Page {item.get('page')}: {item.get('error') or 'Extraction failed'}"
            )
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Review Pages",
        "",
    ])
    if review_pages:
        for item in review_pages:
            warn_text = "; ".join(item.get("warnings") or []) or "Review requested"
            lines.append(f"- Page {item.get('page')}: {warn_text}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Low-Confidence Pages",
        "",
    ])
    if low_conf_pages:
        for item in low_conf_pages:
            lines.append(
                f"- Page {item.get('page')}: confidence={item.get('confidence')}"
            )
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Page Types",
        "",
    ])
    for key, value in sorted(page_type_counts.items()):
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## QA Issues",
        "",
    ])
    issues = qa_report.get("issues", []) or []
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- None")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def generate_hotspot_report(
    book: AssembledBook,
    output_path: Optional[str] = None,
    *,
    max_items: int = 200,
) -> Dict[str, Any]:
    """
    Generate a deterministic hotspot report for targeted fuzzy review.

    This is advisory (not part of the strict QA gate). It flags suspicious spans
    such as likely running-header leaks, boundary-merge contamination, opener
    duplicates, and malformed structured rows.
    """
    hotspots: List[Dict[str, Any]] = []
    seen_keys: set = set()

    def _chapter_page_span(ch: Chapter) -> str:
        if not ch.pages:
            return ""
        return f"{min(ch.pages)}-{max(ch.pages)}" if len(ch.pages) > 1 else str(ch.pages[0])

    def _add_hotspot(
        *,
        category: str,
        severity: str,
        chapter: Chapter,
        line_no: int,
        line: str,
        reason: str,
        context_lines: Optional[List[str]] = None,
    ) -> None:
        if len(hotspots) >= max_items:
            return
        key = (category, chapter.number, line_no, line.strip())
        if key in seen_keys:
            return
        seen_keys.add(key)
        entry = {
            "category": category,
            "severity": severity,
            "chapter_number": chapter.number,
            "chapter_title": chapter.title,
            "chapter_pages": _chapter_page_span(chapter),
            "line_number": line_no,
            "line": line.rstrip(),
            "reason": reason,
        }
        if context_lines:
            entry["context"] = [ln.rstrip() for ln in context_lines[:5]]
        hotspots.append(entry)

    # Pass 1: per-chapter local scans.
    repeated_header_candidates: Dict[str, List[Tuple[Chapter, int, str]]] = defaultdict(list)
    for chapter in book.chapters:
        chapter_kind = _chapter_title_kind(chapter.title)
        lines = chapter.body.split("\n")
        non_empty_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
        title_norm = _normalize_simple((chapter.title or "").rstrip(".:"))

        # Composite opener duplicates / title-component leftovers at chapter start.
        for idx in non_empty_idxs[:8]:
            s = lines[idx].strip()
            if not s:
                continue
            if s.startswith(("#", "|", ">", "[^")):
                break
            if _looks_like_footnote_definition_line(s):
                break
            if re.match(r'^[a-z]', s):
                break
            s_norm = _normalize_simple(s.rstrip(".:"))
            if not s_norm or len(s_norm) < 6:
                continue
            headingish = (
                _looks_like_standalone_chapter_heading_line(s)
                or _is_standalone_ordinal_marker(s)
                or bool(re.match(r'(?i)^[ivxlcdm]+[\.\-—–:]\s*\S+', s))
                or bool(re.search(r'(?i)\b(book|part|chapter|introduction|preface|appendix|index)\b', s))
            )
            if not headingish:
                break
            if title_norm and (s_norm == title_norm or s_norm in title_norm or title_norm in s_norm):
                lo = max(0, idx - 1)
                hi = min(len(lines), idx + 2)
                _add_hotspot(
                    category="chapter_opener_duplicate",
                    severity="medium",
                    chapter=chapter,
                    line_no=idx + 1,
                    line=lines[idx],
                    reason="Heading-like opener line appears to duplicate a component of the chapter title.",
                    context_lines=lines[lo:hi],
                )

        # Suspicious boundary-merge contamination patterns in prose.
        for idx, raw in enumerate(lines):
            s = raw.strip()
            if not s or s.startswith(("#", "|", ">", "[^")):
                continue
            if re.search(r'\b[a-z]{4,}-[A-Z]{4,}\b', s):
                _add_hotspot(
                    category="suspicious_boundary_merge",
                    severity="high",
                    chapter=chapter,
                    line_no=idx + 1,
                    line=raw,
                    reason="Lowercase word joined directly to an ALL-CAPS token via hyphen; possible header merge contamination.",
                    context_lines=lines[max(0, idx - 1): min(len(lines), idx + 2)],
                )
            elif re.search(r'\b[a-z]{5,}[A-Z]{5,}\b', s):
                _add_hotspot(
                    category="suspicious_boundary_merge",
                    severity="medium",
                    chapter=chapter,
                    line_no=idx + 1,
                    line=raw,
                    reason="Lowercase-to-ALL-CAPS token fusion detected; possible page-boundary merge artifact.",
                    context_lines=lines[max(0, idx - 1): min(len(lines), idx + 2)],
                )

            # Malformed structured page/token residue that often signals parse issues.
            if re.search(r'(?<!\\)[\[{]\d{2,4}\b', s):
                _add_hotspot(
                    category="malformed_page_token",
                    severity="medium",
                    chapter=chapter,
                    line_no=idx + 1,
                    line=raw,
                    reason="Bracketed page-number token residue found in reader output.",
                    context_lines=lines[max(0, idx - 1): min(len(lines), idx + 2)],
                )

        # Malformed markdown table restarts (common in split TOC/list pages):
        # repeated header row appears, but separator row is missing before rows continue.
        seen_table_headers_with_separator: set = set()
        for idx, raw in enumerate(lines):
            s = raw.strip()
            if not _looks_like_markdown_table_row_line(s):
                continue
            if _is_markdown_table_separator_row(s):
                continue

            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                continue
            nxt = lines[j].strip()
            if _is_markdown_table_separator_row(nxt):
                seen_table_headers_with_separator.add(s)
                continue
            if not _looks_like_markdown_table_row_line(nxt):
                continue
            if _markdown_table_cell_count(s) != _markdown_table_cell_count(nxt):
                continue
            if s not in seen_table_headers_with_separator:
                continue

            _add_hotspot(
                category="malformed_markdown_table_restart",
                severity="medium",
                chapter=chapter,
                line_no=idx + 1,
                line=raw,
                reason="Markdown table header row is followed by table rows without an immediate separator row.",
                context_lines=lines[max(0, idx - 1): min(len(lines), j + 2)],
            )

        # Running-header-like short uppercase lines inside body flow.
        if chapter_kind in {"frontmatter", "preface", "index"}:
            continue

        for pos, idx in enumerate(non_empty_idxs):
            s = lines[idx].strip()
            if not s or s.startswith(("#", "|", ">", "[^")):
                continue
            if re.fullmatch(r'[\-—–_*=·•♦◈\s]{3,}', s):
                continue
            if s.startswith(("(", "[")):
                continue
            if len(s) > 80 or len(s.split()) > 8:
                continue
            if _is_standalone_ordinal_marker(s):
                continue
            if re.match(r'(?i)^[ivxlcdm]+$', s):
                continue
            alpha = [c for c in s if c.isalpha()]
            if len(alpha) < 4:
                continue
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            if upper_ratio < 0.8:
                continue

            prev_nonempty = non_empty_idxs[pos - 1] if pos > 0 else None
            next_nonempty = non_empty_idxs[pos + 1] if pos + 1 < len(non_empty_idxs) else None
            prev_line = lines[prev_nonempty].strip() if prev_nonempty is not None else ""
            next_line = lines[next_nonempty].strip() if next_nonempty is not None else ""
            next_looks_prose = bool(next_line and (re.match(r'^[a-z"“\'(\[]', next_line) or len(next_line.split()) > 8))
            prev_looks_prose = bool(prev_line and (re.match(r'^[a-z"“\'(\[]', prev_line) or len(prev_line.split()) > 8))
            if not (prev_looks_prose or next_looks_prose):
                continue

            repeated_header_candidates[_normalize_simple(s.rstrip(".:"))].append((chapter, idx + 1, lines[idx]))

            _add_hotspot(
                category="header_like_body_line",
                severity="low",
                chapter=chapter,
                line_no=idx + 1,
                line=lines[idx],
                reason="Short uppercase line appears inside prose flow; possible running-header leak.",
                context_lines=lines[max(0, idx - 1): min(len(lines), idx + 2)],
            )

    # Pass 2: repeated header-like lines across the book are higher-signal.
    for norm, occs in repeated_header_candidates.items():
        if not norm or len(occs) < 3:
            continue
        # Skip obvious structural headings that legitimately recur as chapter titles.
        if norm in {"introduction", "preface"}:
            continue
        for chapter, line_no, line in occs[:8]:
            _add_hotspot(
                category="repeated_header_like_line",
                severity="medium",
                chapter=chapter,
                line_no=line_no,
                line=line,
                reason=f"Header-like line repeats {len(occs)} times across the assembled book.",
            )

    by_category = Counter(h["category"] for h in hotspots)
    by_severity = Counter(h["severity"] for h in hotspots)
    summary = {
        "total_hotspots": len(hotspots),
        "by_category": dict(sorted(by_category.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "max_items": max_items,
        "truncated": len(hotspots) >= max_items,
    }
    report = {
        "model": MODEL,
        "title": book.title,
        "author": book.author,
        "summary": summary,
        "hotspots": hotspots,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def qa_gate(qa_report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    QA gate: determine if the book passes archival quality standards.

    Returns:
        Tuple of (passed: bool, reasons: list of failure reasons)
    """
    failures = []

    # Footnote integrity must pass
    fn = qa_report.get("footnote_integrity", {})
    if not fn.get("ok", False):
        if fn.get("orphaned_refs"):
            failures.append(f"Orphaned footnote refs: {fn['orphaned_refs']}")
        if fn.get("unreferenced_defs"):
            failures.append(f"Unreferenced footnote defs: {fn['unreferenced_defs']}")
        if fn.get("duplicate_defs"):
            failures.append(f"Duplicate footnote defs: {fn['duplicate_defs']}")

    # Must have at least some content
    if qa_report.get("total_words", 0) == 0:
        failures.append("No words extracted — book is empty")

    # Issues list from QA
    for issue in qa_report.get("issues", []):
        failures.append(issue)

    return (len(failures) == 0, failures)
