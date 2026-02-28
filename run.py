#!/usr/bin/env python3
"""
Project Akshara: Pipeline Runner (FLASH Variant)
=================================================

End-to-end pipeline for archival book digitization using Gemini 3 Flash.

Pipeline stages:
  Pass 0 — Preflight: PDF → optimized page images
  Pass 1 — Extraction: Page images → text (Gemini 3 Flash)
  Pass 2 — Assembly: Pages → chapters → book
  Pass 3 — QA Report: Deterministic quality checks (gate in --strict mode)
  Pass 4 — Export: Markdown, EPUB, DOCX

Features:
  - Per-page JSON checkpointing (--resume / --no-cache)
  - YAML configuration (--config) applied to all runtime settings
  - Cost tracking and summary
  - Blank page / LLM commentary filtering
  - QA gate (--strict) blocks export on archival violations

Usage:
    python run.py input/book.pdf
    python run.py input/book.pdf --resume
    python run.py input/book.pdf --pages 1-50
    python run.py input/book.pdf --config config/custom.yaml
    python run.py input/book.pdf --strict
"""

import os
import sys
import json
import time
import re
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# ─── Environment ──────────────────────────────────────────────────────────────

def load_env():
    """Load .env file from project root."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

load_env()

# ─── YAML Config ──────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent / "config" / "default.yaml"

    if path.exists():
        try:
            import yaml
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            print(f"[Config] Loaded: {path}")
            return config
        except ImportError:
            print("[Config] PyYAML not installed, using defaults")
            return {}
        except Exception as e:
            print(f"[Config] Error loading {path}: {e}")
            return {}
    else:
        print(f"[Config] No config file at {path}, using defaults")
        return {}

# ─── Imports ──────────────────────────────────────────────────────────────────

from src.gemini_client import GeminiClient, ThinkingLevel, MediaResolution
from src.claude_client import ClaudeClient
from src.kimi_client import KimiClient
from src.preflight import PreflightProcessor
from src.extraction import PageExtractor, PageExtraction, save_extraction
from src.assembly import (
    BookAssembler,
    generate_qa_report,
    generate_hotspot_report,
    qa_gate,
    write_verification_report_markdown,
)
from src.user_logging import UserLogger
from src.verification_maps import (
    build_source_map,
    build_output_map,
    generate_structural_diff_report,
    write_json,
)

# ─── Defaults (overridden by config) ─────────────────────────────────────────

DEFAULTS = {
    "variant": "FLASH",
    "gemini_model": "flash",
    "gemini_timeout_sec": 120,
    "gemini_max_retries": 3,
    "gemini_retry_delay_sec": 2.0,
    "extraction_thinking": "minimal",
    "structure_thinking": "medium",
    "boundary_thinking": "low",
    "resolution": "high",
    "cost_warn_threshold": 5.0,
    "cost_abort_threshold": 15.0,
    "preflight_dpi": 300,
    "claude_model": "haiku",
    "detect_structure": True,
    "blank_page_filter": True,
    "strict_chapter_detection": True,
    "cache_failures": False,
    "llm_assembly_ops": True,
    "reader_assembly_mode": "deterministic",
    "preserve_nonchapter_headings": True,
    "reader_aggressive_publish_normalization": True,
    "assembly_llm_planner_routing": {},
    "assembly_llm_planner_models": {},
    "write_archival_master": True,
    "kimi_shadow_enabled": False,
    "kimi_shadow_model": "kimi-k2.5",
    "kimi_base_url": "https://api.moonshot.cn/v1",
}

THINKING_MAP = {
    "minimal": ThinkingLevel.MINIMAL,
    "low": ThinkingLevel.LOW,
    "medium": ThinkingLevel.MEDIUM,
    "high": ThinkingLevel.HIGH,
}

RESOLUTION_MAP = {
    "low": MediaResolution.LOW,
    "medium": MediaResolution.MEDIUM,
    "high": MediaResolution.HIGH,
    "ultra_high": MediaResolution.ULTRA_HIGH,
}

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  Project Akshara — Archival Book Digitization Pipeline      ║
║  Variant: FLASH (Gemini 3 Flash + Claude Haiku 4.5)         ║
╚══════════════════════════════════════════════════════════════╝
"""


def apply_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge YAML config into runtime settings, returning effective settings."""
    settings = dict(DEFAULTS)

    gemini_cfg = config.get("gemini", {})
    if gemini_cfg.get("model"):
        raw_model = str(gemini_cfg["model"]).strip()
        model_alias = {
            "gemini-3-flash-preview": "flash",
            "gemini-3.1-pro-preview": "pro",
            "gemini-3-pro-preview": "pro-3",
        }.get(raw_model, raw_model)
        settings["gemini_model"] = model_alias
    if gemini_cfg.get("extraction_thinking"):
        settings["extraction_thinking"] = gemini_cfg["extraction_thinking"]
    if gemini_cfg.get("structure_thinking"):
        settings["structure_thinking"] = gemini_cfg["structure_thinking"]
    if gemini_cfg.get("boundary_thinking"):
        settings["boundary_thinking"] = gemini_cfg["boundary_thinking"]
    if gemini_cfg.get("resolution"):
        settings["resolution"] = gemini_cfg["resolution"]
    if gemini_cfg.get("timeout_sec") is not None:
        settings["gemini_timeout_sec"] = int(gemini_cfg["timeout_sec"])
    if gemini_cfg.get("max_retries") is not None:
        settings["gemini_max_retries"] = int(gemini_cfg["max_retries"])
    if gemini_cfg.get("retry_delay_sec") is not None:
        settings["gemini_retry_delay_sec"] = float(gemini_cfg["retry_delay_sec"])

    cost_cfg = config.get("cost", {})
    if cost_cfg.get("warn_threshold") is not None:
        settings["cost_warn_threshold"] = float(cost_cfg["warn_threshold"])
    if cost_cfg.get("abort_threshold") is not None:
        settings["cost_abort_threshold"] = float(cost_cfg["abort_threshold"])

    preflight_cfg = config.get("preflight", {})
    if preflight_cfg.get("target_dpi") is not None:
        settings["preflight_dpi"] = int(preflight_cfg["target_dpi"])

    extraction_cfg = config.get("extraction", {})
    if extraction_cfg.get("detect_structure") is not None:
        settings["detect_structure"] = bool(extraction_cfg["detect_structure"])
    if extraction_cfg.get("blank_page_filter") is not None:
        settings["blank_page_filter"] = bool(extraction_cfg["blank_page_filter"])

    assembly_cfg = config.get("assembly", {})
    if assembly_cfg.get("strict_chapter_detection") is not None:
        settings["strict_chapter_detection"] = bool(assembly_cfg["strict_chapter_detection"])
    if assembly_cfg.get("llm_assembly_ops") is not None:
        settings["llm_assembly_ops"] = bool(assembly_cfg["llm_assembly_ops"])
    if assembly_cfg.get("reader_assembly_mode"):
        settings["reader_assembly_mode"] = str(assembly_cfg["reader_assembly_mode"]).strip().lower()
    if assembly_cfg.get("preserve_nonchapter_headings") is not None:
        settings["preserve_nonchapter_headings"] = bool(assembly_cfg["preserve_nonchapter_headings"])
    if assembly_cfg.get("reader_aggressive_publish_normalization") is not None:
        settings["reader_aggressive_publish_normalization"] = bool(
            assembly_cfg["reader_aggressive_publish_normalization"]
        )
    routing_cfg = assembly_cfg.get("llm_planner_routing")
    if isinstance(routing_cfg, dict):
        settings["assembly_llm_planner_routing"] = {
            str(k).strip(): str(v).strip().lower()
            for k, v in routing_cfg.items()
            if str(k).strip() and str(v).strip()
        }
    models_cfg = assembly_cfg.get("llm_planner_models")
    if isinstance(models_cfg, dict):
        settings["assembly_llm_planner_models"] = {
            str(k).strip(): str(v).strip()
            for k, v in models_cfg.items()
            if str(k).strip() and str(v).strip()
        }

    cache_cfg = config.get("cache", {})
    if cache_cfg.get("cache_failures") is not None:
        settings["cache_failures"] = bool(cache_cfg["cache_failures"])

    claude_cfg = config.get("claude", {})
    if claude_cfg.get("model"):
        settings["claude_model"] = claude_cfg["model"]

    export_cfg = config.get("export", {})
    if export_cfg.get("write_archival_master") is not None:
        settings["write_archival_master"] = bool(export_cfg["write_archival_master"])

    kimi_cfg = config.get("kimi", {})
    if kimi_cfg.get("shadow_enabled") is not None:
        settings["kimi_shadow_enabled"] = bool(kimi_cfg["shadow_enabled"])
    if kimi_cfg.get("model"):
        settings["kimi_shadow_model"] = str(kimi_cfg["model"]).strip()
    if kimi_cfg.get("base_url"):
        settings["kimi_base_url"] = str(kimi_cfg["base_url"]).strip()

    return settings


def _path_suffix_matches(path: Path, suffix: List[str]) -> bool:
    parts = [p.lower() for p in path.parts]
    return len(parts) >= len(suffix) and parts[-len(suffix):] == [s.lower() for s in suffix]


def _looks_like_explicit_run_dir(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    if "runs" not in parts:
        return False
    idx = parts.index("runs")
    # runs/<scope>/<book>/<run_id>
    return len(parts) >= idx + 4 and parts[idx + 1] in {"full", "tests"}


def _slug_token(value: Any, fallback: str = "na") -> str:
    s = str(value or "").strip().lower()
    if not s:
        s = fallback
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or fallback


def _ascii_book_id(text: str, fallback: str = "book") -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = fallback

    # Prefer the leading title before common subtitle separators when available.
    lead = re.split(r"\s*[;:]\s*", text, maxsplit=1)[0].strip()
    if len(re.sub(r"[^A-Za-z0-9]+", "", lead)) >= 4:
        text = lead

    text = text.replace("&", " and ")
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        text = fallback
    if len(text) > 80:
        text = text[:80].rstrip("_")
    return text or fallback


def _extract_pdf_title(pdf_path: Path) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None

    try:
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata or {}
    except Exception:
        return None

    title = str(metadata.get("title") or "").strip()
    if not title:
        return None
    lowered = title.lower()
    if lowered in {"untitled", "unknown", "none"}:
        return None
    return re.sub(r"\s+", " ", title).strip()


def _derive_book_id(pdf_path: Path) -> str:
    title = _extract_pdf_title(pdf_path)
    if title:
        return _ascii_book_id(title, fallback=_ascii_book_id(pdf_path.stem, fallback="book"))
    return _ascii_book_id(pdf_path.stem, fallback="book")


def _make_run_id(
    *,
    page_range: Optional[tuple],
    variant: str = "flash",
    reader_assembly_mode: str = "deterministic",
    llm_assembly_ops: bool = True,
    strict: bool = False,
) -> str:
    """
    Generate a readable, sortable run folder name.
    Examples:
      20260224-142501__full__flash__llm-guided__ops__strict
      20260224-142501__p0023-0025__flash__deterministic__no-ops
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scope = "full"
    if page_range:
        start, end = page_range
        scope = f"p{int(start):04d}-{int(end):04d}"

    mode = _slug_token(reader_assembly_mode, "deterministic")
    variant_tok = _slug_token(variant, "flash")
    ops_tok = "ops" if llm_assembly_ops else "no-ops"
    strict_tok = "strict" if strict else "nostrict"

    return "__".join([stamp, scope, variant_tok, mode, ops_tok, strict_tok])


def _ensure_unique_run_dir(path: Path) -> Path:
    """
    Avoid collisions when multiple runs start within the same second.
    """
    if not path.exists():
        return path
    parent = path.parent
    stem = path.name
    for i in range(2, 1000):
        candidate = parent / f"{stem}__r{i:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique run directory under {parent}")


def _resolve_output_run_dir(
    base_output_dir: Path,
    book_id: str,
    page_range: Optional[tuple],
    *,
    variant: str = "flash",
    reader_assembly_mode: str = "deterministic",
    llm_assembly_ops: bool = True,
    strict: bool = False,
) -> Path:
    """
    Auto-organize outputs under output/runs/<full|tests>/<book_id>/<run_id>
    when the caller passes a root-ish output directory.
    """
    if _looks_like_explicit_run_dir(base_output_dir):
        return base_output_dir

    rootish_suffixes = [
        ["output"],
        ["output", "runs"],
        ["output", "runs", "full"],
        ["output", "runs", "tests"],
        ["runs"],
        ["runs", "full"],
        ["runs", "tests"],
    ]
    should_auto_nest = any(_path_suffix_matches(base_output_dir, s) for s in rootish_suffixes)
    if not should_auto_nest:
        return base_output_dir

    scope = "tests" if page_range else "full"
    run_id = _make_run_id(
        page_range=page_range,
        variant=variant,
        reader_assembly_mode=reader_assembly_mode,
        llm_assembly_ops=llm_assembly_ops,
        strict=strict,
    )

    if _path_suffix_matches(base_output_dir, ["output"]):
        root = base_output_dir / "runs"
    elif _path_suffix_matches(base_output_dir, ["output", "runs"]) or _path_suffix_matches(base_output_dir, ["runs"]):
        root = base_output_dir
    elif _path_suffix_matches(base_output_dir, ["output", "runs", "full"]) or _path_suffix_matches(base_output_dir, ["runs", "full"]):
        return _ensure_unique_run_dir(base_output_dir / book_id / run_id)
    elif _path_suffix_matches(base_output_dir, ["output", "runs", "tests"]) or _path_suffix_matches(base_output_dir, ["runs", "tests"]):
        return _ensure_unique_run_dir(base_output_dir / book_id / run_id)
    else:
        root = base_output_dir / "runs"

    return _ensure_unique_run_dir(root / scope / book_id / run_id)


# ─── Checkpointing ───────────────────────────────────────────────────────────

class CheckpointManager:
    """Per-page JSON cache for extraction results."""

    def __init__(
        self,
        output_dir: str,
        book_id: str,
        enabled: bool = True,
        cache_failures: bool = False,
    ):
        self.enabled = enabled
        self.cache_failures = cache_failures
        self.cache_dir = Path(output_dir) / ".cache" / book_id
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, page_number: int) -> Optional[PageExtraction]:
        if not self.enabled:
            return None
        cache_file = self.cache_dir / f"page_{page_number:04d}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding='utf-8'))
                return PageExtraction.from_dict(data)
            except Exception:
                return None
        return None

    def put(self, extraction: PageExtraction):
        if not self.enabled:
            return
        if not extraction.success and not self.cache_failures:
            return
        cache_file = self.cache_dir / f"page_{extraction.page_number:04d}.json"
        cache_file.write_text(
            json.dumps(extraction.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def clear(self):
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# ─── Cost Tracking ───────────────────────────────────────────────────────────

def print_cost_summary(
    gemini_client: GeminiClient,
    claude_client: Optional[ClaudeClient],
    model_name: str,
    kimi_client: Optional[KimiClient] = None,
):
    """Print token usage and cost summary."""
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)

    gemini_usage = gemini_client.get_usage_summary()
    gt = gemini_usage["total_tokens"]
    print(f"\nGemini ({model_name}):")
    print(f"  Input tokens:  {gt['input']:>12,}")
    print(f"  Output tokens: {gt['output']:>12,}")
    print(f"  Cached tokens: {gt['cached']:>12,}")
    print(f"  Total tokens:  {gt['total']:>12,}")
    print(f"  Cost:          ${gemini_usage['total_cost_usd']:>11.4f}")

    total_cost = gemini_usage["total_cost_usd"]

    if claude_client:
        claude_usage = claude_client.get_usage_summary()
        ct = claude_usage["total_tokens"]
        print(f"\nClaude (Haiku 4.5):")
        print(f"  Input tokens:  {ct['input']:>12,}")
        print(f"  Output tokens: {ct['output']:>12,}")
        print(f"  Cache write:   {ct['cache_creation']:>12,}")
        print(f"  Cache read:    {ct['cache_read']:>12,}")
        print(f"  Total tokens:  {ct['total']:>12,}")
        print(f"  Cost:          ${claude_usage['total_cost_usd']:>11.4f}")
        total_cost += claude_usage["total_cost_usd"]

    if kimi_client:
        kimi_usage = kimi_client.get_usage_summary()
        kt = kimi_usage["total_tokens"]
        print(f"\nKimi Shadow ({kimi_usage.get('model', 'unknown')}):")
        print(f"  Input tokens:  {kt['input']:>12,}")
        print(f"  Output tokens: {kt['output']:>12,}")
        print(f"  Cached tokens: {kt['cached']:>12,}")
        print(f"  Total tokens:  {kt['total']:>12,}")
        print(f"  Cost*:         ${kimi_usage['total_cost_usd']:>11.4f}")
        print("  *Cost shown may be 0.0 until Moonshot pricing is configured.")
        total_cost += kimi_usage["total_cost_usd"]

    print(f"\n{'─' * 40}")
    print(f"  TOTAL COST:    ${total_cost:>11.4f}")
    print("=" * 60)


def check_cost(gemini_client: GeminiClient, claude_client: Optional[ClaudeClient],
               warn: float, abort: float) -> bool:
    """Check if cost exceeds thresholds. Returns True if should continue."""
    total = gemini_client.total_cost
    if claude_client:
        total += claude_client.total_cost

    if total >= abort:
        print(f"\n[ABORT] Cost ${total:.2f} exceeds abort threshold ${abort:.2f}")
        return False
    if total >= warn:
        print(f"\n[WARN] Cost ${total:.2f} exceeds warning threshold ${warn:.2f}")
    return True


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    output_dir: str = "./output",
    page_range: Optional[tuple] = None,
    resume: bool = False,
    no_cache: bool = False,
    config: Optional[Dict[str, Any]] = None,
    formats: Optional[List[str]] = None,
    strict: bool = False,
    reader_assembly_mode_override: Optional[str] = None,
):
    """
    Run the full Akshara pipeline.

    Args:
        pdf_path: Path to input PDF
        output_dir: Output directory
        page_range: Optional (start, end) tuple
        resume: Resume from checkpoint
        no_cache: Disable caching
        config: YAML config dict
        formats: Export formats (default: ["markdown"])
        strict: If True, QA failures block export and exit non-zero
        reader_assembly_mode_override: Override config for reader assembly mode
    """
    config = config or {}
    formats = formats or ["markdown"]
    start_time = time.time()

    # Apply config to get effective runtime settings
    settings = apply_config(config)
    if reader_assembly_mode_override:
        settings["reader_assembly_mode"] = str(reader_assembly_mode_override).strip().lower()

    extraction_thinking = THINKING_MAP.get(settings["extraction_thinking"], ThinkingLevel.MINIMAL)
    structure_thinking = THINKING_MAP.get(settings["structure_thinking"], ThinkingLevel.MEDIUM)
    resolution = RESOLUTION_MAP.get(settings["resolution"], MediaResolution.HIGH)
    cost_warn = settings["cost_warn_threshold"]
    cost_abort = settings["cost_abort_threshold"]
    preflight_dpi = settings["preflight_dpi"]
    detect_structure = settings["detect_structure"]
    blank_page_filter = settings["blank_page_filter"]
    strict_chapter_detection = settings["strict_chapter_detection"]
    cache_failures = settings["cache_failures"]
    llm_assembly_ops = settings["llm_assembly_ops"]
    reader_assembly_mode = settings["reader_assembly_mode"]
    preserve_nonchapter_headings = settings["preserve_nonchapter_headings"]
    reader_aggressive_publish_normalization = settings["reader_aggressive_publish_normalization"]
    write_archival_master = settings["write_archival_master"]
    kimi_shadow_enabled = settings["kimi_shadow_enabled"]
    kimi_shadow_model = settings["kimi_shadow_model"]
    kimi_base_url = settings["kimi_base_url"]

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    book_id = _derive_book_id(pdf_path)
    output_dir = _resolve_output_run_dir(
        Path(output_dir),
        book_id,
        page_range,
        variant=settings["variant"],
        reader_assembly_mode=reader_assembly_mode,
        llm_assembly_ops=llm_assembly_ops,
        strict=strict,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gemini_model_name = settings["gemini_model"]
    logger = UserLogger(str(output_dir), book_id, settings["variant"], verbose=True)
    logger.info(
        "Pipeline configured.",
        model=gemini_model_name,
        gemini_timeout_sec=settings["gemini_timeout_sec"],
        gemini_max_retries=settings["gemini_max_retries"],
        extraction_thinking=settings["extraction_thinking"],
        structure_thinking=settings["structure_thinking"],
        detect_structure=detect_structure,
        blank_page_filter=blank_page_filter,
        strict_chapter_detection=strict_chapter_detection,
        strict_mode=strict,
        preserve_nonchapter_headings=preserve_nonchapter_headings,
        reader_aggressive_publish_normalization=reader_aggressive_publish_normalization,
        assembly_llm_planner_routing=settings["assembly_llm_planner_routing"],
        assembly_llm_planner_models=settings["assembly_llm_planner_models"],
        write_archival_master=write_archival_master,
        kimi_shadow_enabled=kimi_shadow_enabled,
        kimi_shadow_model=kimi_shadow_model,
    )

    print(BANNER)
    print(f"Input:    {pdf_path}")
    print(f"Book ID:  {book_id}")
    print(f"Run Dir:  {output_dir}")
    print(f"Model:    {gemini_model_name}")
    print(
        f"Gemini:   timeout={settings['gemini_timeout_sec']}s, "
        f"retries={settings['gemini_max_retries']}, "
        f"retry_delay={settings['gemini_retry_delay_sec']}s"
    )
    print(f"Thinking: extraction={settings['extraction_thinking']}, structure={settings['structure_thinking']}")
    print(f"Resume:   {resume}")
    print(f"Cache:    {'disabled' if no_cache else 'enabled'}")
    print(f"Strict:   {strict}")
    print(f"Cost:     warn=${cost_warn:.0f}, abort=${cost_abort:.0f}")
    print(f"Extract:  detect_structure={detect_structure}, blank_page_filter={blank_page_filter}")
    print(
        "Assemble: "
        f"strict_chapter_detection={strict_chapter_detection}, "
        f"llm_assembly_ops={llm_assembly_ops}, "
        f"reader_assembly_mode={reader_assembly_mode}, "
        f"preserve_nonchapter_headings={preserve_nonchapter_headings}, "
        f"reader_aggressive_publish_normalization={reader_aggressive_publish_normalization}"
    )
    if settings["assembly_llm_planner_routing"]:
        print(f"Planner routing: {settings['assembly_llm_planner_routing']}")
    if settings["assembly_llm_planner_models"]:
        print(f"Planner models:  {settings['assembly_llm_planner_models']}")
    print(f"Output:   write_archival_master={write_archival_master}")
    print()
    logger.info("Run started. Preparing to initialize model clients.")

    # ─── Initialize Clients ───────────────────────────────────────────────

    gemini = GeminiClient(
        default_model=gemini_model_name,
        default_thinking=extraction_thinking,
        timeout=settings["gemini_timeout_sec"],
        max_retries=settings["gemini_max_retries"],
        retry_delay=settings["gemini_retry_delay_sec"],
        verbose=True,
    )

    claude = None
    try:
        claude = ClaudeClient(default_model=settings["claude_model"], verbose=True)
    except Exception as e:
        print(f"[Warning] Claude client unavailable: {e}")
        print("          Assembly will skip cross-page word joining.")

    kimi_shadow = None
    if kimi_shadow_enabled:
        try:
            kimi_shadow = KimiClient(
                default_model=kimi_shadow_model,
                base_url=kimi_base_url,
                verbose=True,
            )
            print(f"[Kimi] Shadow challenger enabled: model={kimi_shadow_model}")
            logger.info(
                "Kimi shadow challenger enabled.",
                model=kimi_shadow_model,
                base_url=kimi_base_url,
            )
        except Exception as e:
            print(f"[Warning] Kimi shadow client unavailable: {e}")
            logger.warning("Kimi shadow client unavailable", error=str(e))

    # ─── Pass 0: Preflight ────────────────────────────────────────────────

    print("\n" + "━" * 60)
    print("PASS 0: PREFLIGHT")
    print("━" * 60)
    logger.stage_start(
        "Preflight",
        "Converting PDF pages into high-quality images for OCR.",
        dpi=preflight_dpi,
    )

    image_dir = output_dir / "images" / book_id
    preflight = PreflightProcessor(target_dpi=preflight_dpi, verbose=True)
    preflight_result = preflight.process_pdf(
        str(pdf_path),
        str(image_dir),
        page_range,
        output_name=book_id,
    )

    print(f"Preflight: {len(preflight_result.pages)} pages processed")
    logger.stage_done(
        "Preflight",
        f"Prepared {len(preflight_result.pages)} page images.",
        pages=len(preflight_result.pages),
        warnings=len(preflight_result.warnings),
        errors=len(preflight_result.errors),
    )
    for err in preflight_result.errors:
        print(f"  [ERROR] {err}")
        logger.error(err, stage="preflight")

    # ─── Pass 1: Extraction ───────────────────────────────────────────────

    print("\n" + "━" * 60)
    print(f"PASS 1: EXTRACTION ({gemini_model_name})")
    print("━" * 60)
    logger.stage_start(
        "Extraction",
        "Reading each page image and transcribing visible text.",
        model=gemini_model_name,
    )

    pages_dir = output_dir / "pages" / book_id
    checkpoint = CheckpointManager(
        str(output_dir),
        book_id,
        enabled=(not no_cache),
        cache_failures=cache_failures,
    )

    if not resume and not no_cache:
        checkpoint.clear()

    extractor = PageExtractor(
        gemini_client=gemini,
        claude_client=claude,
        extraction_thinking=extraction_thinking,
        structure_thinking=structure_thinking,
        resolution=resolution,
        blank_page_filter=blank_page_filter,
        verbose=True,
    )

    extractions: List[PageExtraction] = []
    cached_count = 0
    extracted_count = 0

    for page_img in preflight_result.pages:
        page_num = page_img.page_number

        if not check_cost(gemini, claude, cost_warn, cost_abort):
            print("Pipeline aborted due to cost limit.")
            break

        if resume:
            cached = checkpoint.get(page_num)
            if cached:
                extractions.append(cached)
                cached_count += 1
                logger.page(
                    page_num,
                    "Loaded from cache",
                    words=cached.word_count,
                    page_type=cached.page_type,
                    source="cache",
                )
                continue

        extraction = extractor.extract_page(
            str(page_img.image_path), page_num, detect_structure=detect_structure,
        )
        extractions.append(extraction)
        extracted_count += 1
        if extraction.success:
            status = "Blank page" if extraction.is_blank else "Extracted"
        else:
            status = "Extraction failed"
        logger.page(
            page_num,
            status,
            words=extraction.word_count,
            page_type=extraction.page_type,
            success=extraction.success,
        )

        checkpoint.put(extraction)
        save_extraction(extraction, str(pages_dir), book_id)

    print(f"\nExtraction: {extracted_count} extracted, {cached_count} from cache, "
          f"{len(extractions)} total")
    logger.stage_done(
        "Extraction",
        f"Finished {len(extractions)} pages ({extracted_count} fresh, {cached_count} cached).",
        extracted=extracted_count,
        cached=cached_count,
        total=len(extractions),
    )

    # ─── Pass 2: Assembly ─────────────────────────────────────────────────

    print("\n" + "━" * 60)
    print("PASS 2: ASSEMBLY")
    print("━" * 60)
    logger.stage_start(
        "Assembly",
        "Merging page-level text into chapters and final book markdown.",
    )

    assembler = BookAssembler(
        gemini_client=gemini,
        claude_client=claude,
        shadow_planner_client=kimi_shadow,
        strict_chapter_detection=strict_chapter_detection,
        llm_assembly_ops=llm_assembly_ops,
        reader_assembly_mode=reader_assembly_mode,
        archival_mode=False,
        preserve_nonchapter_headings=preserve_nonchapter_headings,
        aggressive_publish_normalization=reader_aggressive_publish_normalization,
        llm_planner_routing=settings["assembly_llm_planner_routing"],
        llm_planner_models=settings["assembly_llm_planner_models"],
        verbose=True,
    )

    title = book_id.replace('_', ' ')
    for ext in extractions[:5]:
        if ext.page_type == "title" and ext.chapter_heading:
            title = ext.chapter_heading
            break

    book = assembler.assemble(extractions, title=title)

    assembled_path = output_dir / f"{book_id}_assembled.md"
    assembled_path.write_text(book.master_markdown, encoding='utf-8')
    print(f"Assembled: {assembled_path}")

    archival_path = None
    archival_book = None
    if write_archival_master:
        archival_assembler = BookAssembler(
            gemini_client=None,
            claude_client=None,
            strict_chapter_detection=strict_chapter_detection,
            llm_assembly_ops=False,
            reader_assembly_mode="deterministic",
            archival_mode=True,
            preserve_nonchapter_headings=True,
            aggressive_publish_normalization=False,
            verbose=True,
        )
        archival_book = archival_assembler.assemble(extractions, title=title)
        archival_path = output_dir / f"{book_id}_archival_master.md"
        archival_path.write_text(archival_book.master_markdown, encoding='utf-8')
        print(f"Archival:  {archival_path}")
    logger.stage_done(
        "Assembly",
        f"Built assembled book with {len(book.chapters)} chapters.",
        assembled_path=str(assembled_path),
        archival_path=str(archival_path) if archival_path else None,
        chapters=len(book.chapters),
        total_words=book.total_words,
    )

    # ─── Pass 3: QA Report + Gate ─────────────────────────────────────────

    print("\n" + "━" * 60)
    print("PASS 3: QA REPORT")
    print("━" * 60)
    logger.stage_start(
        "QA",
        "Running deterministic checks for footnotes and structural consistency.",
    )

    qa_path = output_dir / f"{book_id}_qa_report.json"
    qa_report = generate_qa_report(book, extractions, str(qa_path))
    verification_md_path = output_dir / f"{book_id}_verification_report.md"
    write_verification_report_markdown(qa_report, str(verification_md_path))
    hotspot_path = output_dir / f"{book_id}_hotspot_report.json"
    hotspot_report = generate_hotspot_report(book, str(hotspot_path))
    qa_report["hotspots"] = hotspot_report.get("summary", {})
    write_json(qa_report, qa_path)

    print(f"QA Report: {qa_path}")
    print(f"  Chapters:  {qa_report['total_chapters']}")
    print(f"  Words:     {qa_report['total_words']:,}")
    print(f"  Blank:     {qa_report['blank_pages']['count']} pages")
    print(f"  Failed:    {qa_report['extraction_audit']['summary']['failed_pages']} pages")
    print(f"  Review:    {qa_report['extraction_audit']['summary']['review_pages']} pages")
    print(f"  Fidelity:  {qa_report['avg_fidelity']:.1f}% (heuristic, not source-verified)")
    fn_status = 'OK' if qa_report['footnote_integrity']['ok'] else 'ISSUES'
    print(f"  Footnotes: {fn_status}")
    print(
        "  Hotspots:"
        f"  {hotspot_report['summary']['total_hotspots']} "
        f"(high={hotspot_report['summary']['by_severity'].get('high', 0)}, "
        f"medium={hotspot_report['summary']['by_severity'].get('medium', 0)}, "
        f"low={hotspot_report['summary']['by_severity'].get('low', 0)})"
    )
    print(f"Hotspots:  {hotspot_path}")
    print(f"Verify:    {verification_md_path}")
    if qa_report['issues']:
        for issue in qa_report['issues']:
            print(f"  [ISSUE] {issue}")
            logger.warning(issue, stage="qa")
    logger.qa_summary(qa_report)
    logger.info(
        "Hotspot report generated.",
        hotspot_report=str(hotspot_path),
        hotspot_total=hotspot_report["summary"]["total_hotspots"],
        hotspot_by_category=hotspot_report["summary"]["by_category"],
        hotspot_by_severity=hotspot_report["summary"]["by_severity"],
    )

    # Structural maps + diff report (for repair-loop triage)
    source_map = build_source_map(extractions, title=title)
    reader_map = build_output_map(book.master_markdown, variant="reader")
    archival_map = build_output_map(archival_book.master_markdown, variant="archival") if archival_book else None
    source_map_path = output_dir / f"{book_id}_source_map.json"
    reader_map_path = output_dir / f"{book_id}_reader_map.json"
    archival_map_path = output_dir / f"{book_id}_archival_map.json" if archival_map else None
    structural_diff_path = output_dir / f"{book_id}_structural_diff_report.json"

    write_json(source_map, source_map_path)
    write_json(reader_map, reader_map_path)
    if archival_map and archival_map_path:
        write_json(archival_map, archival_map_path)
    structural_diff = generate_structural_diff_report(
        source_map=source_map,
        reader_map=reader_map,
        archival_map=archival_map,
    )
    write_json(structural_diff, structural_diff_path)
    print(f"StructMap: {source_map_path}")
    print(f"ReaderMap: {reader_map_path}")
    if archival_map_path:
        print(f"ArchMap:   {archival_map_path}")
    print(
        "StructDiff:"
        f" {structural_diff_path} "
        f"({structural_diff['summary']['total_issues']} issues)"
    )
    logger.info(
        "Structural maps and diff report generated.",
        source_map=str(source_map_path),
        reader_map=str(reader_map_path),
        archival_map=str(archival_map_path) if archival_map_path else None,
        structural_diff=str(structural_diff_path),
        structural_issues=structural_diff["summary"]["total_issues"],
    )

    # QA Gate (Fix #2)
    passed, failures = qa_gate(qa_report)
    if strict:
        if not passed:
            print(f"\n[QA GATE] FAILED — {len(failures)} violations:")
            for f in failures:
                print(f"  - {f}")
                logger.error(f, stage="qa_gate")
            print("\nExport blocked. Fix issues or run without --strict.")
            print_cost_summary(gemini, claude, gemini_model_name)
            logger.stage_done("QA", "QA gate failed in strict mode.", passed=False)
            logger.close()
            sys.exit(1)
        else:
            print("\n[QA GATE] PASSED — all archival checks OK")
            logger.stage_done("QA", "QA gate passed.", passed=True)
    else:
        logger.stage_done("QA", "QA report generated (strict gate not enforced).", passed=passed)

    # ─── Pass 4: Export ───────────────────────────────────────────────────

    if "epub" in formats or "docx" in formats or "pdf" in formats:
        print("\n" + "━" * 60)
        print("PASS 4: EXPORT")
        print("━" * 60)
        logger.stage_start(
            "Export",
            "Creating reader-friendly output formats.",
            formats=formats,
        )

        from src.export import BookExporter
        exporter = BookExporter(verbose=True)
        exported = exporter.export_book(book, str(output_dir), formats)

        for result in exported.results:
            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"  {result.format}: {status}")
            if result.success:
                logger.info(
                    f"Exported {result.format}.",
                    format=result.format,
                    output_path=result.output_path,
                    bytes=result.file_size,
                )
            else:
                logger.warning(
                    f"Export failed for {result.format}: {result.error}",
                    format=result.format,
                    error=result.error,
                )
        logger.stage_done("Export", "Export step finished.")

    # ─── Summary ──────────────────────────────────────────────────────────

    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.1f}s")
    print_cost_summary(gemini, claude, gemini_model_name, kimi_client=kimi_shadow)
    logger.info("Pipeline completed successfully.", elapsed_seconds=round(elapsed, 2))
    logger.close()

    return book


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project Akshara (FLASH) — Archival Book Digitization Pipeline"
    )
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-p", "--pages", help="Page range (e.g., 1-50)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-cache", action="store_true", help="Disable page caching")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--strict", action="store_true",
                        help="QA gate: block export and exit non-zero on archival violations")
    parser.add_argument(
        "--reader-assembly-mode",
        choices=["deterministic", "llm_guided", "llm_chunks"],
        help="Reader assembly strategy override (default from config: deterministic)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["markdown"],
        choices=["markdown", "epub", "docx", "pdf"],
        help="Export formats (default: markdown)"
    )

    args = parser.parse_args()

    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        if len(parts) == 2:
            page_range = (int(parts[0]), int(parts[1]))
        else:
            page_range = (int(parts[0]), int(parts[0]))

    config = load_config(args.config)

    run_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        page_range=page_range,
        resume=args.resume,
        no_cache=args.no_cache,
        config=config,
        formats=args.formats,
        strict=args.strict,
        reader_assembly_mode_override=args.reader_assembly_mode,
    )


if __name__ == "__main__":
    main()
