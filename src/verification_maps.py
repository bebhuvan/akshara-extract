"""
Project Akshara: Structural Maps + Diff QA
==========================================

Builds source/output JSON maps and a structural diff report that can drive a
repair loop (deterministic fixes, LLM ops fixes, or manual review).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.extraction import PageExtraction
from src.assembly import (
    CHAPTER_KEYWORDS,
    ROMAN_NUMERAL,
    extract_footnotes,
    _looks_like_page_header_combo,
    _looks_like_running_header_with_page_number,
)


INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]

SCAN_NOISE_PATTERNS = [
    re.compile(r"(?i)^\s*uc[- ]?nrlf\s*$"),
    re.compile(r"(?i)^\s*[ab]\s*\d[\d\s]{3,}\s*$"),
    re.compile(r"(?i)^\s*(?:repl|refel|reel)\s+\d+\s*$"),
    re.compile(r"(?i)^\s*reproduced\s+by\s+duopage\s+process\b"),
    re.compile(r"(?i)^\s*in\s+the\s+united\s+states\s+of\s+america\s*$"),
    re.compile(r"(?i)^\s*micro\s+photo\s+inc\.?\s*$"),
    re.compile(r"(?i)^\s*cleveland\s+\d+,\s*ohio\.?\s*$"),
    re.compile(r"(?i)^\s*[a-z]-\d{5,}\s*$"),
    re.compile(r"(?i)^\s*ds\s*\.?\s*\d{1,4}\s*[a-z]?\d*\s*$"),
    re.compile(r"(?i)^\s*r\d{1,4}\s*$"),
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _indic_char_count(text: str) -> int:
    count = 0
    for ch in text:
        cp = ord(ch)
        for start, end in INDIC_RANGES:
            if start <= cp <= end:
                count += 1
                break
    return count


def _script_profile(text: str) -> Dict[str, int]:
    indic = _indic_char_count(text)
    ascii_letters = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    digits = sum(1 for ch in text if ch.isdigit())
    return {
        "indic_chars": indic,
        "ascii_letters": ascii_letters,
        "non_ascii_chars": non_ascii,
        "digits": digits,
    }


def _normalize_loose(text: str) -> str:
    s = text.lower()
    s = s.replace("\u00ad", "")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _looks_like_page_number_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^\[?\(?\d{1,4}\)?\]?[.*]?$", s):
        return True
    if ROMAN_NUMERAL.match(s.rstrip(".").upper()):
        return True
    return False


def _looks_like_scan_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.search(s) for p in SCAN_NOISE_PATTERNS)


def _classify_line(line: str) -> str:
    s = line.strip()
    if not s:
        return "blank"
    if s.startswith("## "):
        return "chapter_heading"
    if s.startswith("# "):
        return "title_heading"
    if s.startswith("[^") and re.match(r"^\[\^[^\]]+\](?::|\s)", s):
        return "footnote_def"
    if s.startswith(">"):
        return "blockquote"
    if s.startswith("|"):
        return "table"
    if s.startswith("#"):
        return "heading"
    if _looks_like_page_number_line(s):
        return "page_number"
    if _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
        return "running_header_with_page"
    if CHAPTER_KEYWORDS.match(s) and len(s.split()) <= 12:
        return "chapterish_label"
    if _looks_like_scan_noise_line(s):
        return "scan_noise"
    return "prose"


def _blockize_lines(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    start = 0
    current: List[str] = []

    def flush(end_idx: int):
        nonlocal current, start
        if not current:
            return
        text = "\n".join(current)
        non_blank = [ln for ln in current if ln.strip()]
        block_type = "blank"
        if non_blank:
            first_type = _classify_line(non_blank[0])
            if all(ln.strip().startswith("|") for ln in non_blank):
                block_type = "table"
            elif all(ln.strip().startswith(">") for ln in non_blank):
                block_type = "blockquote"
            elif first_type == "footnote_def":
                block_type = "footnote_block"
            elif first_type in ("chapter_heading", "title_heading", "heading", "chapterish_label"):
                block_type = "heading_block"
            elif first_type in ("page_number", "running_header_with_page", "scan_noise"):
                block_type = "artifact_candidate"
            else:
                block_type = "prose_block"

        blocks.append(
            {
                "type": block_type,
                "line_start": start + 1,
                "line_end": end_idx,
                "line_count": len(current),
                "char_count": len(text),
                "sha1": _sha1(text),
                "script_profile": _script_profile(text),
                "text": text,
            }
        )
        current = []

    for idx, line in enumerate(lines, start=1):
        if line.strip() and not current:
            start = idx - 1
        if not line.strip():
            if current:
                flush(idx - 1)
            continue
        current.append(line)
    if current:
        flush(len(lines))

    return blocks


def build_source_map(extractions: List[PageExtraction], title: str) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    totals = {
        "pages": 0,
        "blank_pages": 0,
        "footnote_defs": 0,
        "indic_chars": 0,
        "non_ascii_chars": 0,
    }

    for ext in extractions:
        text = ext.text or ""
        lines = text.split("\n") if text else []
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        top_lines = non_empty[:3]
        bottom_lines = non_empty[-3:] if non_empty else []
        _, footnotes = extract_footnotes(text, remove_from_body=False)
        blocks = _blockize_lines(lines)
        page_profile = _script_profile(text)

        page_entry = {
            "page_number": ext.page_number,
            "success": ext.success,
            "page_type": ext.page_type,
            "detected_chapter_heading": ext.chapter_heading,
            "confidence": ext.confidence,
            "warnings": ext.warnings,
            "word_count": ext.word_count,
            "line_count": len(lines),
            "script_profile": page_profile,
            "top_lines": top_lines,
            "bottom_lines": bottom_lines,
            "top_artifact_candidates": [
                ln for ln in top_lines
                if _looks_like_page_number_line(ln)
                or _looks_like_running_header_with_page_number(ln)
                or _looks_like_page_header_combo(ln)
                or _looks_like_scan_noise_line(ln)
            ],
            "footnotes": footnotes,
            "blocks": blocks,
        }
        pages.append(page_entry)

        totals["pages"] += 1
        if ext.is_blank:
            totals["blank_pages"] += 1
        totals["footnote_defs"] += len(footnotes)
        totals["indic_chars"] += page_profile["indic_chars"]
        totals["non_ascii_chars"] += page_profile["non_ascii_chars"]

    return {
        "schema_version": "akshara.structural-map.v1",
        "map_type": "source",
        "title": title,
        "totals": totals,
        "pages": pages,
    }


def build_output_map(markdown: str, *, variant: str) -> Dict[str, Any]:
    lines = markdown.splitlines()
    chapters: List[Dict[str, Any]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    chapter_start_line = 1
    doc_title = ""

    for idx, line in enumerate(lines, start=1):
        if idx == 1 and line.startswith("# "):
            doc_title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_title is not None:
                blocks = _blockize_lines(current_lines)
                chapters.append(
                    {
                        "title": current_title,
                        "line_start": chapter_start_line,
                        "line_end": idx - 1,
                        "word_count": len("\n".join(current_lines).split()),
                        "blocks": blocks,
                    }
                )
            current_title = line[3:].strip()
            current_lines = []
            chapter_start_line = idx
        else:
            current_lines.append(line)

    if current_title is not None:
        blocks = _blockize_lines(current_lines)
        chapters.append(
            {
                "title": current_title,
                "line_start": chapter_start_line,
                "line_end": len(lines),
                "word_count": len("\n".join(current_lines).split()),
                "blocks": blocks,
            }
        )

    text_profile = _script_profile(markdown)
    markdown_fn_defs = re.findall(r"^\[\^([^\]]+)\](?::|\s)", markdown, flags=re.MULTILINE)
    dollar_fn_defs: List[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not re.match(r"^\$\^\{?[\w*†‡§\d-]+\}?\$\s+", stripped):
            continue
        dollar_fn_defs.extend(re.findall(r"\$\^\{?([\w*†‡§\d-]+)\}?\$", stripped))
    footnote_defs = markdown_fn_defs + dollar_fn_defs
    quote_blocks = sum(
        1 for ch in chapters for b in ch["blocks"] if b["type"] == "blockquote"
    )
    table_blocks = sum(
        1 for ch in chapters for b in ch["blocks"] if b["type"] == "table"
    )

    return {
        "schema_version": "akshara.structural-map.v1",
        "map_type": "output",
        "variant": variant,
        "title": doc_title,
        "totals": {
            "chapters": len(chapters),
            "word_count": len(markdown.split()),
            "footnote_defs": len(footnote_defs),
            "indic_chars": text_profile["indic_chars"],
            "non_ascii_chars": text_profile["non_ascii_chars"],
            "blockquote_blocks": quote_blocks,
            "table_blocks": table_blocks,
        },
        "chapters": chapters,
    }


def _iter_output_lines_for_scan(map_data: Dict[str, Any]) -> List[Tuple[str, int, str]]:
    """
    Returns tuples of (chapter_title, line_number_within_chapter_block, line_text)
    for individual lines in output blocks.
    """
    rows: List[Tuple[str, int, str]] = []
    for ch in map_data.get("chapters", []):
        chapter_title = ch.get("title", "")
        for block in ch.get("blocks", []):
            block_start = int(block.get("line_start", 1))
            for offset, line in enumerate(str(block.get("text", "")).split("\n"), start=0):
                rows.append((chapter_title, block_start + offset, line))
    return rows


def generate_structural_diff_report(
    *,
    source_map: Dict[str, Any],
    reader_map: Dict[str, Any],
    archival_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    def _is_frontmatter_chapter(title: str) -> bool:
        return _normalize_loose(title) == "front matter"

    def _is_index_chapter(title: str) -> bool:
        norm = _normalize_loose(title)
        return norm.startswith("index ")

    def _looks_like_toc_entry_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if re.match(r'(?i)^(?:book|part|chapter)\b', s):
            return True
        if re.match(r'(?i)^(?:introduction|conclusion)\.?$', s):
            return True
        if re.match(r'(?i)^[ivxlcdm]+\.\s+', s):
            return True
        if re.match(r'(?i)^(?:i{1,3}|iv|v|vi{0,3}|ix|x)\.\s+.*\b(?:\d+|[ivxlcdm]{1,8})\s*$', s):
            return True
        if re.match(r'(?i)^(?:page|contents)\.?$', s):
            return True
        return False

    def add_issue(
        issue_type: str,
        severity: str,
        location: Dict[str, Any],
        message: str,
        suggested_action: str,
        repair_class: str,
        confidence: float = 0.9,
        evidence: Optional[Dict[str, Any]] = None,
    ):
        issues.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "location": location,
                "message": message,
                "suggested_action": suggested_action,
                "repair_class": repair_class,
                "confidence": confidence,
                "evidence": evidence or {},
            }
        )

    source_pages = source_map.get("pages", [])
    reader_chapters = reader_map.get("chapters", [])

    # 1) Repeated / suspicious chapter titles in reader.
    seen_titles: Dict[str, int] = {}
    for idx, ch in enumerate(reader_chapters):
        title = str(ch.get("title", "")).strip()
        title_norm = _normalize_loose(title)
        if not title_norm:
            continue
        seen_titles[title_norm] = seen_titles.get(title_norm, 0) + 1
        if seen_titles[title_norm] > 1 and len(reader_chapters) > 1:
            add_issue(
                "duplicate_chapter_title",
                "high",
                {"chapter_index": idx, "chapter_title": title},
                f"Repeated chapter title detected in reader output: '{title}'",
                "Recheck chapter-start detection on nearby pages and suppress running-header splits.",
                "deterministic_fix",
                evidence={"title_norm": title_norm, "occurrence": seen_titles[title_norm]},
            )
        if (
            ch.get("word_count", 0)
            and ch["word_count"] < 120
            and idx > 0
            and not re.match(r'(?i)^(?:book|part)\b', title)
        ):
            add_issue(
                "tiny_chapter_segment",
                "medium",
                {"chapter_index": idx, "chapter_title": title},
                f"Small chapter segment ({ch['word_count']} words) may indicate a false split.",
                "Review boundary and chapter-start heuristics around the preceding page boundary.",
                "deterministic_fix",
                confidence=0.7,
            )

    # 2) Reader leakage: page numbers / running headers / scan noise.
    for chapter_title, line_no, line in _iter_output_lines_for_scan(reader_map):
        s = line.strip()
        if not s:
            continue
        # Skip explicit markdown headings and tables.
        if s.startswith(("#", "|", ">")):
            continue
        if _is_frontmatter_chapter(chapter_title) and _looks_like_toc_entry_line(s):
            continue
        # Raw index OCR frequently emits isolated page references on wrapped lines; these
        # are formatting targets but not header/page-number leaks.
        if _is_index_chapter(chapter_title) and _looks_like_page_number_line(s):
            continue
        # Standalone publication years in frontmatter are legitimate imprint data.
        if _is_frontmatter_chapter(chapter_title) and re.fullmatch(r"(1[5-9]\d{2}|20\d{2})", s):
            continue
        if _looks_like_page_number_line(s):
            add_issue(
                "reader_page_number_leak",
                "medium",
                {"chapter_title": chapter_title, "line": line_no},
                f"Standalone page number leaked into reader output: '{s}'",
                "Extend reader page-header/footer stripping for this page pattern.",
                "deterministic_fix",
                confidence=0.85,
                evidence={"line": s},
            )
        elif _looks_like_running_header_with_page_number(s) or _looks_like_page_header_combo(s):
            add_issue(
                "reader_running_header_leak",
                "high",
                {"chapter_title": chapter_title, "line": line_no},
                f"Running header/page-number combo leaked into reader output: '{s}'",
                "Strip top-page header stack or combined header/page-number line in reader assembly.",
                "deterministic_fix",
                confidence=0.95,
                evidence={"line": s},
            )
        elif _looks_like_scan_noise_line(s):
            add_issue(
                "reader_scan_noise_leak",
                "medium",
                {"chapter_title": chapter_title, "line": line_no},
                f"Scan/library artifact leaked into reader output: '{s}'",
                "Add reader-only frontmatter/TOC scan-noise suppression rule or LLM ops classification for this pattern.",
                "deterministic_fix",
                confidence=0.9,
                evidence={"line": s},
            )

    # 3) Cross-map footnote parity (source extracted defs vs outputs).
    source_footnotes = int(source_map.get("totals", {}).get("footnote_defs", 0))
    reader_footnotes = int(reader_map.get("totals", {}).get("footnote_defs", 0))
    if reader_footnotes != source_footnotes:
        add_issue(
            "footnote_count_mismatch_reader_vs_source",
            "high",
            {"scope": "book"},
            f"Reader footnote definition count ({reader_footnotes}) != source extracted footnote count ({source_footnotes})",
            "Inspect footnote extraction and chapter-end note assembly on flagged pages.",
            "deterministic_fix",
            confidence=0.9,
            evidence={"source_footnotes": source_footnotes, "reader_footnotes": reader_footnotes},
        )
    if archival_map is not None:
        archival_footnotes = int(archival_map.get("totals", {}).get("footnote_defs", 0))
        if archival_footnotes != source_footnotes:
            add_issue(
                "footnote_count_mismatch_archival_vs_source",
                "high",
                {"scope": "book"},
                f"Archival footnote definition count ({archival_footnotes}) != source extracted footnote count ({source_footnotes})",
                "Archival assembly should not drop footnotes; inspect footnote collection/retention.",
                "deterministic_fix",
                confidence=0.9,
                evidence={"source_footnotes": source_footnotes, "archival_footnotes": archival_footnotes},
            )

    # 4) Indic/script loss checks.
    source_indic = int(source_map.get("totals", {}).get("indic_chars", 0))
    reader_indic = int(reader_map.get("totals", {}).get("indic_chars", 0))
    if source_indic > 0 and reader_indic < source_indic:
        add_issue(
            "indic_script_loss_reader",
            "critical",
            {"scope": "book"},
            f"Indic character count dropped from source ({source_indic}) to reader ({reader_indic}).",
            "Inspect extraction, normalization, and export transforms for Unicode/script loss.",
            "manual_review",
            confidence=0.8,
            evidence={"source_indic": source_indic, "reader_indic": reader_indic},
        )
    if archival_map is not None:
        archival_indic = int(archival_map.get("totals", {}).get("indic_chars", 0))
        if source_indic > 0 and archival_indic < source_indic:
            add_issue(
                "indic_script_loss_archival",
                "critical",
                {"scope": "book"},
                f"Indic character count dropped from source ({source_indic}) to archival ({archival_indic}).",
                "Archival output must preserve script content; inspect extraction and assembly path immediately.",
                "manual_review",
                confidence=0.9,
                evidence={"source_indic": source_indic, "archival_indic": archival_indic},
            )

    # 5) Quote/verse style drift indicators (reader only, heuristic).
    source_quoteish_lines = 0
    source_poetry_pages = 0
    for p in source_pages:
        text_blocks = p.get("blocks", [])
        source_quoteish_lines += sum(
            1
            for b in text_blocks
            for ln in str(b.get("text", "")).split("\n")
            if ln.strip().startswith(('"', "'", "“", "‘"))
        )
        if p.get("page_type") == "text" and p.get("blocks") and any(
            blk.get("type") == "prose_block" and blk.get("line_count", 0) >= 3 and blk.get("char_count", 0) < 260
            for blk in text_blocks
        ) and any(p.get("warnings", [])) is False:
            # soft signal only; actual poem detection comes from extraction metadata in future map versions
            pass
        if any(k for k in [p.get("detected_chapter_heading")] if k):
            pass
        # Prefer extraction metadata if available (not copied into totals)
        # We infer from top-level page JSON in future versions; omitted here to keep map compact.
    reader_blockquotes = int(reader_map.get("totals", {}).get("blockquote_blocks", 0))
    if source_quoteish_lines >= 5 and reader_blockquotes == 0:
        add_issue(
            "quote_formatting_missed",
            "low",
            {"scope": "book"},
            "Source contains many quote-leading lines but reader output has no blockquote blocks.",
            "Review quote-detection heuristics in publication normalization (reader edition).",
            "llm_ops_fix",
            confidence=0.55,
            evidence={"source_quoteish_lines": source_quoteish_lines, "reader_blockquotes": reader_blockquotes},
        )

    # 6) Top-line artifact pressure from source pages (helps prioritize review).
    for p in source_pages:
        candidates = p.get("top_artifact_candidates") or []
        if len(candidates) >= 2 and p.get("page_type") in ("text", "frontmatter", "backmatter"):
            add_issue(
                "high_header_leak_risk_page",
                "low",
                {"page_number": p.get("page_number")},
                f"Source page has multiple top-line artifact/header candidates: {candidates[:2]}",
                "Prioritize this page in reader-header leakage checks if output still shows artifacts.",
                "deterministic_fix",
                confidence=0.6,
                evidence={"top_artifact_candidates": candidates[:3]},
            )

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    issues.sort(key=lambda x: (severity_rank.get(x["severity"], 9), x["issue_type"]))

    summary = {
        "total_issues": len(issues),
        "by_severity": {
            sev: sum(1 for i in issues if i["severity"] == sev)
            for sev in ("critical", "high", "medium", "low")
        },
        "repair_classes": {
            rc: sum(1 for i in issues if i["repair_class"] == rc)
            for rc in ("deterministic_fix", "llm_ops_fix", "manual_review")
        },
    }

    return {
        "schema_version": "akshara.structural-diff.v1",
        "summary": summary,
        "source_totals": source_map.get("totals", {}),
        "reader_totals": reader_map.get("totals", {}),
        "archival_totals": (archival_map or {}).get("totals", {}),
        "issues": issues,
    }


def write_json(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
