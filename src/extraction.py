"""
Project Akshara: Pass 1 - Text Extraction
==========================================

Extracts text from preprocessed page images using Gemini.
Integrates blank page filtering to prevent LLM commentary from
leaking into the final output.

Usage:
    extractor = PageExtractor(gemini_client)
    result = extractor.extract_page(image_path, page_number)
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

from src.blank_page_filter import is_llm_commentary, filter_blank_page_text
from src.gemini_client import GeminiClient, GeminiResponse, ThinkingLevel, MediaResolution

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Extraction prompt for Gemini
EXTRACTION_PROMPT = """You are a precise OCR system for Project Akshara, an archival book digitization project.

TASK: Transcribe ALL visible text from this page image exactly as printed.

CRITICAL RULES:
1. Transcribe EVERY character exactly as it appears — preserve archaic spellings, punctuation, capitalisation.
2. Preserve paragraph structure with blank lines between paragraphs.
3. Preserve line breaks within poetry, verse, or formatted text.
4. Preserve footnote references and definitions exactly. Use [^N] and [^N]: markup ONLY when the marker/definition is clearly visible in the scan.
5. For non-Latin scripts (Devanagari, Tamil, etc.), transcribe in the original script.
6. If a word is hyphenated at a line break, keep the hyphen: "archaeo-\\nlogical"
7. For illustrations/figures, transcribe ONLY visible caption/label text. Do NOT describe image contents from inference. If there is no visible text, output [BLANK PAGE].
8. For tables, preserve visible row/column text and ordering. Use markdown table format only if the structure is clearly visible; do NOT infer missing rows/cells.
9. Do NOT add any commentary, notes, explanations, summaries, or descriptions of your own.
10. If the page is blank or contains no text, output ONLY: [BLANK PAGE]

OUTPUT: The transcribed text, nothing else."""

# Structure detection prompt
STRUCTURE_PROMPT = """Analyze this page image and identify its structural elements.

Return a JSON object with these fields:
{
    "page_type": "text|title|toc|index|illustration|blank|frontmatter|backmatter",
    "has_headers": true/false,
    "has_footers": true/false,
    "has_footnotes": true/false,
    "has_illustrations": true/false,
    "has_tables": true/false,
    "has_poetry": true/false,
    "has_indic_script": true/false,
    "chapter_heading": "chapter title if this is a chapter start, null otherwise",
    "estimated_word_count": number,
    "confidence": 0.0-1.0
}"""


@dataclass
class PageExtraction:
    """Result of extracting text from a single page."""
    page_number: int
    text: str
    raw_text: str  # Before filtering
    page_type: str  # text, title, toc, illustration, blank, etc.
    has_footnotes: bool = False
    has_illustrations: bool = False
    has_indic_script: bool = False
    has_tables: bool = False
    has_poetry: bool = False
    chapter_heading: Optional[str] = None
    word_count: int = 0
    confidence: float = 0.0
    fidelity: float = 0.0  # Computed from extraction signals, not verified against source
    extraction_model: str = ""
    structure_data: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON caching."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageExtraction':
        """Deserialize from dictionary."""
        return cls(**data)

    @property
    def is_blank(self) -> bool:
        """Check if the source page was identified as truly blank."""
        return self.page_type == "blank"

    @property
    def is_error(self) -> bool:
        """Check if extraction failed for this page."""
        return self.page_type == "error" or not self.success


class PageExtractor:
    """
    Pass 1: Extract text from page images using Gemini.

    Features:
    - Gemini-powered OCR with thinking
    - Structure detection (page type, footnotes, illustrations)
    - Blank page / LLM commentary filtering
    - Fidelity scoring (heuristic, not source-verified)

    Usage:
        client = GeminiClient()
        extractor = PageExtractor(client)
        result = extractor.extract_page("/path/to/page.png", 1)
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        claude_client: Optional[Any] = None,
        extraction_thinking: ThinkingLevel = ThinkingLevel.LOW,
        structure_thinking: ThinkingLevel = ThinkingLevel.MEDIUM,
        resolution: MediaResolution = MediaResolution.HIGH,
        blank_page_filter: bool = True,
        verbose: bool = True
    ):
        self.gemini = gemini_client
        self.claude = claude_client
        self.extraction_thinking = extraction_thinking
        self.structure_thinking = structure_thinking
        self.resolution = resolution
        self.blank_page_filter = blank_page_filter
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[Extract] {message}")

    def _extract_with_claude(self, image_path: str) -> Optional[str]:
        """
        Claude Haiku fallback when Gemini extraction is blocked or empty.
        """
        if not self.claude:
            return None
        safe_image_path = self._prepare_claude_safe_image(image_path)
        try:
            response = self.claude.generate_with_image(
                prompt=EXTRACTION_PROMPT,
                image_path=safe_image_path,
                model="haiku",
                max_tokens=4096,
            )
            if not response.success:
                return None
            text = (response.text or "").strip()
            return text
        except Exception:
            return None
        finally:
            if safe_image_path != image_path:
                try:
                    Path(safe_image_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _prepare_claude_safe_image(self, image_path: str, max_bytes: int = 4_700_000) -> str:
        """
        Claude image uploads hard-fail above ~5MB. Convert large PNGs into a smaller JPEG.
        """
        path = Path(image_path)
        if not PIL_AVAILABLE or not path.exists() or path.stat().st_size <= max_bytes:
            return str(path)

        try:
            img = Image.open(path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            max_dim = 2200
            if max(img.size) > max_dim:
                scale = max_dim / float(max(img.size))
                new_size = (
                    max(1, int(img.size[0] * scale)),
                    max(1, int(img.size[1] * scale)),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            fd, temp_path = tempfile.mkstemp(prefix="akshara_claude_", suffix=".jpg")
            os.close(fd)
            quality = 85
            while quality >= 45:
                img.save(temp_path, format="JPEG", quality=quality, optimize=True)
                if Path(temp_path).stat().st_size <= max_bytes:
                    return temp_path
                quality -= 10
            return temp_path
        except Exception:
            return str(path)

    def _build_segment_prompt(self, segment_index: int, segment_count: int) -> str:
        return (
            EXTRACTION_PROMPT
            + "\n\n"
            + f"SEGMENT NOTE: This image is vertical segment {segment_index} of {segment_count} from a single page. "
            + "Transcribe only the visible text in this segment. "
            + "Do not infer or restore text cut off outside the crop. "
            + "If the first or last visible line is partial, transcribe only the visible characters."
        )

    def _iter_vertical_segments(
        self,
        image_path: str,
        *,
        segment_count: int = 3,
        overlap_px: int = 80,
    ) -> List[Tuple[int, str]]:
        path = Path(image_path)
        if not PIL_AVAILABLE or not path.exists():
            return []

        try:
            img = Image.open(path)
            width, height = img.size
            if height < 900:
                return []

            segments: List[Tuple[int, str]] = []
            band = max(1, height // segment_count)
            for idx in range(segment_count):
                top = max(0, idx * band - (overlap_px if idx > 0 else 0))
                bottom = height if idx == segment_count - 1 else min(height, (idx + 1) * band + overlap_px)
                crop = img.crop((0, top, width, bottom))
                fd, temp_path = tempfile.mkstemp(prefix=f"akshara_seg_{idx+1}_", suffix=".png")
                os.close(fd)
                crop.save(temp_path, format="PNG")
                segments.append((idx + 1, temp_path))
            return segments
        except Exception:
            return []

    def _normalize_line_for_merge(self, line: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", line.lower())

    def _merge_segment_texts(self, texts: List[str]) -> str:
        merged_lines: List[str] = []
        for text in texts:
            lines = text.splitlines()
            if not merged_lines:
                merged_lines.extend(lines)
                continue

            overlap = 0
            max_probe = min(8, len(merged_lines), len(lines))
            for probe in range(max_probe, 0, -1):
                left = [self._normalize_line_for_merge(x) for x in merged_lines[-probe:]]
                right = [self._normalize_line_for_merge(x) for x in lines[:probe]]
                if left == right and any(left):
                    overlap = probe
                    break
            merged_lines.extend(lines[overlap:])

        return "\n".join(merged_lines).strip()

    def _extract_with_segmented_llm(self, image_path: str) -> Optional[Tuple[str, str, str]]:
        segments = self._iter_vertical_segments(image_path)
        if not segments:
            return None

        texts: List[str] = []
        models_used: List[str] = []
        segment_count = len(segments)

        try:
            for segment_index, segment_path in segments:
                prompt = self._build_segment_prompt(segment_index, segment_count)
                response = self.gemini.generate_with_fallback(
                    prompt=prompt,
                    image_path=segment_path,
                    thinking=self.extraction_thinking,
                    resolution=self.resolution,
                )
                segment_text = (response.text or "").strip() if response.success else ""
                segment_model = response.model or "gemini-segment"

                if not segment_text and self.claude:
                    claude_text = self._extract_with_claude(segment_path)
                    if claude_text and claude_text.strip():
                        segment_text = claude_text.strip()
                        segment_model = "claude-haiku-4.5-segment-fallback"

                if not segment_text:
                    return None

                texts.append(segment_text)
                models_used.append(segment_model)

            merged = self._merge_segment_texts(texts)
            if not merged:
                return None
            model_label = "+".join(sorted(set(models_used)))
            note = "Used segmented LLM fallback due provider block/empty full-page extraction"
            return merged, model_label, note
        finally:
            for _, segment_path in segments:
                Path(segment_path).unlink(missing_ok=True)

    def _coerce_structure_data(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Normalize structure payload to a dict.
        Gemini can occasionally emit JSON arrays despite JSON-mode prompts.
        """
        if isinstance(data, dict):
            return data

        if isinstance(data, list):
            # Common case: a single-item list containing the expected object.
            if len(data) == 1 and isinstance(data[0], dict):
                return data[0]

            # Fallback: choose first dict-like item that contains known keys.
            for item in data:
                if isinstance(item, dict) and any(
                    k in item for k in (
                        "page_type", "has_footnotes", "has_illustrations",
                        "has_indic_script", "has_tables", "has_poetry",
                        "chapter_heading", "confidence",
                    )
                ):
                    return item

            # Fallback: convert [{"key": "...", "value": ...}, ...] to dict.
            kv: Dict[str, Any] = {}
            for item in data:
                if isinstance(item, dict) and "key" in item and "value" in item:
                    kv[str(item["key"])] = item["value"]
            if kv:
                return kv

        return None

    def extract_page(
        self,
        image_path: str,
        page_number: int,
        detect_structure: bool = True,
    ) -> PageExtraction:
        """
        Extract text from a single page image.

        Args:
            image_path: Path to the page image
            page_number: Page number (1-indexed)
            detect_structure: Whether to run structure detection

        Returns:
            PageExtraction with text and metadata
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return PageExtraction(
                page_number=page_number,
                text="",
                raw_text="",
                page_type="error",
                error=f"Image not found: {image_path}",
                success=False
            )

        self._log(f"Page {page_number}: Extracting text...")

        # Step 1: Extract raw text
        response = self.gemini.generate_with_fallback(
            prompt=EXTRACTION_PROMPT,
            image_path=str(image_path),
            thinking=self.extraction_thinking,
            resolution=self.resolution,
        )

        used_claude_fallback = False
        fallback_note: Optional[str] = None

        if response.success:
            # Some safety/recitation fallback paths can return success=True with no text.
            raw_text = (response.text or "").strip()
            extraction_model = response.model
        else:
            raw_text = ""
            extraction_model = response.model

        if not raw_text:
            claude_text = self._extract_with_claude(str(image_path))
            if claude_text is not None and claude_text.strip():
                raw_text = claude_text.strip()
                extraction_model = "claude-haiku-4.5-fallback"
                used_claude_fallback = True
                fallback_note = "Used Claude Haiku fallback due Gemini block/empty output"
                self._log(f"Page {page_number}: Gemini empty/blocked, used Claude fallback")
            elif not response.success:
                segmented = self._extract_with_segmented_llm(str(image_path))
                if segmented:
                    raw_text, extraction_model, fallback_note = segmented
                    self._log(f"Page {page_number}: used segmented LLM fallback")
                else:
                    return PageExtraction(
                        page_number=page_number,
                        text="",
                        raw_text="",
                        page_type="error",
                        extraction_model=extraction_model,
                        error=response.error or "Extraction failed",
                        success=False
                    )
            else:
                segmented = self._extract_with_segmented_llm(str(image_path))
                if segmented:
                    raw_text, extraction_model, fallback_note = segmented
                    self._log(f"Page {page_number}: used segmented LLM fallback")
                else:
                    return PageExtraction(
                        page_number=page_number,
                        text="",
                        raw_text="",
                        page_type="error",
                        extraction_model=extraction_model,
                        error=response.error or "Empty extraction response",
                        success=False
                    )

        # Step 2: Filter LLM commentary (Bug Fix #1)
        if self.blank_page_filter and is_llm_commentary(raw_text):
            self._log(f"Page {page_number}: LLM commentary detected → BLANK")
            return PageExtraction(
                page_number=page_number,
                text="",
                raw_text=raw_text,
                page_type="blank",
                extraction_model=extraction_model,
                confidence=0.9,
                success=True
            )

        # Check for explicit blank page marker
        if raw_text.strip() == "[BLANK PAGE]":
            return PageExtraction(
                page_number=page_number,
                text="",
                raw_text=raw_text,
                page_type="blank",
                extraction_model=extraction_model,
                confidence=1.0,
                success=True
            )

        # Filter any remaining commentary lines from the text
        cleaned_text = filter_blank_page_text(raw_text) if self.blank_page_filter else raw_text
        if not cleaned_text:
            return PageExtraction(
                page_number=page_number,
                text="",
                raw_text=raw_text,
                page_type="blank",
                extraction_model=extraction_model,
                confidence=0.8,
                warnings=["Text was entirely LLM commentary after line filtering"],
                success=True
            )

        # Step 3: Structure detection (optional)
        structure_data = None
        page_type = "text"
        has_footnotes = False
        has_illustrations = False
        has_indic_script = False
        has_tables = False
        has_poetry = False
        chapter_heading = None
        confidence = 0.8

        if detect_structure:
            structure_data = self._detect_structure(str(image_path))
            if structure_data:
                page_type = structure_data.get("page_type", "text")
                has_footnotes = structure_data.get("has_footnotes", False)
                has_illustrations = structure_data.get("has_illustrations", False)
                has_indic_script = structure_data.get("has_indic_script", False)
                has_tables = structure_data.get("has_tables", False)
                has_poetry = structure_data.get("has_poetry", False)
                chapter_heading = structure_data.get("chapter_heading")
                confidence = structure_data.get("confidence", 0.8)

        # Also detect footnotes from text patterns
        if not has_footnotes:
            has_footnotes = bool(re.search(r'\[\^[\d\w]+\]', cleaned_text))

        # Detect Indic scripts from text
        if not has_indic_script:
            has_indic_script = self._has_indic_script(cleaned_text)

        # Word count
        word_count = len(cleaned_text.split())

        # Warnings and fidelity computation
        warnings = []
        # Start fidelity at confidence score (0-1 from structure detection)
        fidelity_score = confidence * 100.0  # Scale to 0-100

        if response.safety_blocked:
            warnings.append("Safety filter triggered, used fallback persona")
            fidelity_score -= 15.0  # Safety fallback degrades trust
        if response.recitation_blocked:
            warnings.append("Recitation filter triggered, used fallback persona")
            fidelity_score -= 10.0
        if used_claude_fallback and fallback_note:
            warnings.append(fallback_note)
            fidelity_score -= 5.0
        elif fallback_note:
            warnings.append(fallback_note)
            fidelity_score -= 7.0
            confidence = min(confidence, 0.7)
        if self.blank_page_filter and len(cleaned_text) < len(raw_text) * 0.9:
            warnings.append("Some LLM commentary was filtered from output")
            fidelity_score -= 10.0  # Filtering needed = less trustworthy
        if not detect_structure or not structure_data:
            fidelity_score -= 5.0  # No structure confirmation

        fidelity_score = max(0.0, min(100.0, fidelity_score))

        return PageExtraction(
            page_number=page_number,
            text=cleaned_text,
            raw_text=raw_text,
            page_type=page_type,
            has_footnotes=has_footnotes,
            has_illustrations=has_illustrations,
            has_indic_script=has_indic_script,
            has_tables=has_tables,
            has_poetry=has_poetry,
            chapter_heading=chapter_heading,
            word_count=word_count,
            confidence=confidence,
            fidelity=fidelity_score,
            extraction_model=extraction_model,
            structure_data=structure_data,
            warnings=warnings,
            success=True
        )

    def _detect_structure(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Run structure detection on a page image."""
        try:
            response = self.gemini.generate_with_image(
                prompt=STRUCTURE_PROMPT,
                image_path=image_path,
                thinking=self.structure_thinking,
                resolution=MediaResolution.MEDIUM,
                json_mode=True,
            )

            if response.success and response.json_data:
                coerced = self._coerce_structure_data(response.json_data)
                if coerced is None:
                    self._log(
                        "Structure detection returned non-object JSON; "
                        "skipping structure metadata for this page."
                    )
                return coerced
            elif response.success and response.text:
                # Try to parse text as JSON
                try:
                    # Strip markdown code fences if present
                    text = response.text.strip()
                    if text.startswith("```"):
                        text = re.sub(r'^```(?:json)?\n?', '', text)
                        text = re.sub(r'\n?```$', '', text)
                    parsed = json.loads(text)
                    coerced = self._coerce_structure_data(parsed)
                    if coerced is None:
                        self._log(
                            "Structure detection text parsed as non-object JSON; "
                            "skipping structure metadata for this page."
                        )
                    return coerced
                except json.JSONDecodeError:
                    return None
            return None
        except Exception as e:
            self._log(f"Structure detection failed: {e}")
            return None

    def _has_indic_script(self, text: str) -> bool:
        """Check if text contains Indic script characters."""
        # Unicode ranges for common Indic scripts
        indic_ranges = [
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
        for char in text:
            code = ord(char)
            for start, end in indic_ranges:
                if start <= code <= end:
                    return True
        return False

    def extract_pages(
        self,
        image_paths: List[str],
        start_page: int = 1,
        detect_structure: bool = True,
    ) -> List[PageExtraction]:
        """
        Extract text from multiple pages.

        Args:
            image_paths: List of image file paths (in order)
            start_page: Starting page number
            detect_structure: Whether to run structure detection

        Returns:
            List of PageExtraction results
        """
        results = []
        for i, image_path in enumerate(image_paths):
            page_num = start_page + i
            result = self.extract_page(image_path, page_num, detect_structure)
            results.append(result)

            status = "OK" if result.success else "FAIL"
            if result.page_type == "blank":
                status = "BLANK"
            self._log(f"Page {page_num}: {status} ({result.word_count} words, {result.page_type})")

        successful = sum(1 for r in results if r.success)
        blank = sum(1 for r in results if r.page_type == "blank")
        self._log(f"Extraction complete: {successful}/{len(results)} pages, {blank} blank")

        return results


def save_extraction(extraction: PageExtraction, output_dir: str, book_id: str) -> Path:
    """Save a page extraction as markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"page_{extraction.page_number:04d}.md"
    output_path = output_dir / filename

    lines = []
    lines.append(f"<!-- Page {extraction.page_number} | Type: {extraction.page_type} | "
                 f"Words: {extraction.word_count} | Fidelity: {extraction.fidelity:.0f}% -->")
    lines.append("")

    if extraction.text:
        lines.append(extraction.text)
    elif extraction.page_type == "blank":
        lines.append(f"<!-- [BLANK PAGE {extraction.page_number}] -->")
    elif extraction.is_error:
        lines.append(f"<!-- [EXTRACTION FAILED PAGE {extraction.page_number}] -->")
        if extraction.error:
            lines.append(f"<!-- Error: {extraction.error} -->")
    else:
        lines.append(f"<!-- [NO TEXT EXTRACTED PAGE {extraction.page_number}] -->")

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path
