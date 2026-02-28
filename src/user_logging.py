"""
Layman-friendly logging utilities for Project Akshara.

Writes:
- Human-readable timeline log (.log)
- Structured event stream (.jsonl)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LogPaths:
    text_log: Path
    jsonl_log: Path


class UserLogger:
    """Simple user-facing logger with text + JSONL sinks."""

    def __init__(self, output_dir: str, book_id: str, variant: str, verbose: bool = True):
        logs_dir = Path(output_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{book_id}_{variant.lower()}_{stamp}"
        self.paths = LogPaths(
            text_log=logs_dir / f"{base}.log",
            jsonl_log=logs_dir / f"{base}.jsonl",
        )
        self.verbose = verbose
        self.book_id = book_id
        self.variant = variant

        self._line(
            f"Session started for '{book_id}' ({variant}). "
            f"Detailed log: {self.paths.text_log}"
        )
        self._event("session_start", {"book_id": book_id, "variant": variant})

    def _line(self, message: str):
        line = f"[{_now()}] {message}"
        with self.paths.text_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.verbose:
            print(f"[Log] {message}")

    def _event(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        data = {
            "ts": _now(),
            "event": event_type,
            "book_id": self.book_id,
            "variant": self.variant,
            "payload": payload or {},
        }
        with self.paths.jsonl_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def info(self, message: str, **payload: Any):
        self._line(message)
        self._event("info", payload)

    def stage_start(self, stage_name: str, plain_note: str = "", **payload: Any):
        msg = f"Starting {stage_name}."
        if plain_note:
            msg += f" {plain_note}"
        self._line(msg)
        self._event("stage_start", {"stage": stage_name, "note": plain_note, **payload})

    def stage_done(self, stage_name: str, plain_result: str = "", **payload: Any):
        msg = f"Completed {stage_name}."
        if plain_result:
            msg += f" {plain_result}"
        self._line(msg)
        self._event("stage_done", {"stage": stage_name, "result": plain_result, **payload})

    def page(self, page_number: int, status: str, words: int = 0, page_type: str = "", **payload: Any):
        ptype = f", type={page_type}" if page_type else ""
        self._line(f"Page {page_number}: {status} (words={words}{ptype}).")
        self._event(
            "page",
            {"page": page_number, "status": status, "words": words, "page_type": page_type, **payload},
        )

    def warning(self, message: str, **payload: Any):
        self._line(f"Warning: {message}")
        self._event("warning", payload)

    def error(self, message: str, **payload: Any):
        self._line(f"Error: {message}")
        self._event("error", payload)

    def qa_summary(self, qa: Dict[str, Any]):
        fn = qa.get("footnote_integrity", {})
        issues = qa.get("issues", [])
        self._line(
            "QA summary: "
            f"{qa.get('total_words', 0):,} words, "
            f"{qa.get('total_chapters', 0)} chapters, "
            f"{qa.get('blank_pages', {}).get('count', 0)} blank pages, "
            f"footnotes={'OK' if fn.get('ok') else 'ISSUES'}."
        )
        if issues:
            self._line(f"QA found {len(issues)} issue(s).")
        self._event("qa_summary", qa)

    def close(self):
        self._line("Session completed.")
        self._event("session_end", {})
