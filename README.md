# Project Akshara

Project Akshara is an archival-first digitization pipeline for scanned public-domain books. It converts scanned PDFs into structured markdown and verification artifacts while preserving the source text as faithfully as possible.

Project site: https://akshara.ink

The project is built around a conservative idea: historical books should be handled like artifacts, not rewritten like generic OCR output.

## Current Model Stack

The current implementation is centered on:
- `Gemini 3 Flash` for primary page extraction
- `Claude Haiku` for fallback and difficult-page recovery

So far, this pairing has worked well:
- Flash is fast and cost-efficient for page-scale transcription
- Claude Haiku is useful as a secondary vision path
- deterministic validation remains necessary around both

## What The Pipeline Does

- renders each PDF page into a stable image representation
- extracts page text with structural hints
- distinguishes true blank pages from extraction failures
- assembles pages into book-level markdown
- generates QA and verification reports for review
- optionally exports into additional publishing formats

## Core Principles

1. No content loss.
2. No content addition.
3. No silent modernization.
4. Uncertainty must be surfaced, not guessed through.
5. Deterministic validation should gate risky model behavior.

## Pipeline Overview

```text
Pass 0: Preflight
    PDF -> normalized page images

Pass 1: Extraction
    page images -> per-page text + metadata + checkpoint JSON

Pass 2: Assembly
    page extractions -> chapter structure -> book markdown

Pass 3: Verification
    QA report + hotspot report + structural maps + verification summary

Pass 4: Export
    optional markdown / EPUB / DOCX / PDF outputs
```

## Repository Structure

```text
.
├── run.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── config/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── assembly.py
│   ├── blank_page_filter.py
│   ├── claude_client.py
│   ├── export.py
│   ├── extraction.py
│   ├── gemini_client.py
│   ├── kimi_client.py
│   ├── preflight.py
│   ├── user_logging.py
│   └── verification_maps.py
└── docs/
    ├── ARCHITECTURE.md
    ├── LEARNINGS.md
    ├── PROJECT_GOALS.md
    ├── RUNBOOK.md
    ├── PUBLISHING_CHECKLIST.md
    └── SECURITY.md
```

## Requirements

- Python 3.11+ recommended
- Linux or macOS preferred
- API access for the providers you intend to use
- `pandoc` only if you want EPUB / DOCX / PDF export
- a local virtual environment is recommended

Main Python dependencies come from:
- PyMuPDF
- Pillow
- OpenCV
- Google GenAI SDK
- Anthropic SDK
- PyYAML

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate the environment variables you need in `.env`.

## Provider Configuration

The pipeline can run with one or more model providers enabled.

Common environment variables:
- `GEMINI_API_KEY` for the primary extraction path
- `ANTHROPIC_API_KEY` for fallback vision extraction
- `MOONSHOT_API_KEY` for optional Kimi assembly experiments
- `LOG_LEVEL` for runtime verbosity
- `OUTPUT_DIR` for the default output root

In normal use:
- Gemini is the default primary provider
- Claude Haiku is the practical fallback for difficult pages
- Kimi is optional and not required for standard extraction runs

If you want the default pipeline behavior, set at least:

```bash
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Then run:

```bash
python run.py path/to/book.pdf
```

## Quick Start

Full-book run:

```bash
python run.py input/book.pdf
```

Resume an interrupted run:

```bash
python run.py input/book.pdf --resume
```

Run only a page slice:

```bash
python run.py input/book.pdf --pages 1-50
```

Strict verification gate:

```bash
python run.py input/book.pdf --strict
```

Use a specific config:

```bash
python run.py input/book.pdf --config config/default.yaml
```

Publish more formats:

```bash
python run.py input/book.pdf --formats markdown epub docx pdf
```

Override the output root:

```bash
python run.py input/book.pdf --output ./output
```

## CLI

```text
python run.py PDF_PATH [options]

Options:
  -o, --output OUTPUT_DIR
  -p, --pages START-END
  --resume
  --no-cache
  --config CONFIG_PATH
  --formats markdown epub docx pdf
  --strict
```

## Configuration

The main configuration lives in [`config/default.yaml`](config/default.yaml).

Important knobs:

### Gemini
- `gemini.model`
- `gemini.extraction_thinking`
- `gemini.structure_thinking`
- `gemini.boundary_thinking`
- `gemini.resolution`
- `gemini.timeout`
- `gemini.max_retries`

### Extraction
- `extraction.detect_structure`
- `extraction.blank_page_filter`

### Assembly
- `assembly.strict_chapter_detection`
- `assembly.llm_assembly_ops`
- `assembly.reader_assembly_mode`
- `assembly.preserve_nonchapter_headings`
- `assembly.reader_aggressive_publish_normalization`
- `assembly.llm_planner_routing`
- `assembly.llm_planner_models`

### Export
- `export.formats`
- `export.write_archival_master`

### Preflight
- `preflight.target_dpi`
- `preflight.enable_deskew`
- `preflight.enable_contrast`
- `preflight.enable_denoise`
- `preflight.image_format`

### Cache
- `cache.enabled`
- `cache.cache_dir`
- `cache.cache_failures`

## Operator Workflow

For a new book, the practical workflow is:

1. Run a small page slice with `--pages`.
2. Inspect per-page markdown under `pages/{book_id}/`.
3. Inspect cached page JSON under `.cache/{book_id}/`.
4. Check the assembled markdown and verification report.
5. Run the full book once extraction quality and page typing look stable.

This matters because most failures in historical books are not uniform across the entire document. A 20-50 page slice usually reveals whether the prompt/config combination is reliable enough for the full run.

## Runtime Output Layout

Each run writes to an isolated directory:

```text
output/runs/full/{book_id}/{run_id}/
output/runs/tests/{book_id}/{run_id}/
```

Typical contents:

```text
run_dir/
├── images/{book_id}/
├── pages/{book_id}/
├── .cache/{book_id}/
├── {book_id}_assembled.md
├── {book_id}_archival_master.md
├── {book_id}_qa_report.json
├── {book_id}_hotspot_report.json
├── {book_id}_verification_report.md
├── {book_id}_source_map.json
├── {book_id}_reader_map.json
├── {book_id}_archival_map.json
├── {book_id}_structural_diff_report.json
└── logs/
```

Operationally:
- `images/` is the preflight render output
- `pages/` is the easiest place to inspect page-by-page extraction quality
- `.cache/` is the resume/checkpoint layer
- `*_assembled.md` is the main assembled output
- `*_archival_master.md` is the more conservative archival companion
- `*_verification_report.md` is the human-readable review summary
- `*_qa_report.json` and `*_structural_diff_report.json` are the main machine-readable QA artifacts

## Extraction Behavior

The extraction pass operates page by page and records a `PageExtraction` object for each page.

Important behavior:
- true blank pages are tracked explicitly as `page_type = "blank"`
- extraction failures are tracked explicitly as `page_type = "error"`
- failures are not allowed to masquerade as blanks
- blocked full-page requests can be retried through segmented LLM extraction
- warnings are preserved for later review

This distinction is important because verification depends on it.

## How The Models Are Used

The current implementation uses models in a constrained way:

- `Gemini 3 Flash` handles primary page extraction
- `Claude Haiku` handles fallback extraction for difficult pages
- segmented LLM retries are used when a full-page request fails but the page may still be recoverable

The important design decision is that the models are not treated as unconstrained rewriting systems. They are used inside a pipeline with:
- deterministic page ordering
- explicit page states
- cached checkpoints
- downstream validation
- verification reports for human review

This is why the system is more reliable than a single-prompt OCR replacement script.

## Technical Architecture Summary

The codebase is organized as a multi-pass pipeline:

- `run.py` orchestrates the entire run, configuration loading, model client setup, run directory creation, and final report generation
- [`src/preflight.py`](src/preflight.py) renders PDF pages into normalized images
- [`src/extraction.py`](src/extraction.py) performs page-level extraction and writes page markdown plus checkpoint JSON
- [`src/assembly.py`](src/assembly.py) assembles extracted pages into book-level outputs and verification artifacts
- [`src/verification_maps.py`](src/verification_maps.py) computes structural maps and diffs for validation
- [`src/export.py`](src/export.py) handles optional downstream publish formats

The architectural reason for this split is simple: page rendering, page transcription, book assembly, and verification fail in different ways and need different controls.

## Failure And Recovery Model

The pipeline is designed to fail explicitly instead of silently.

Important behaviors:
- blank pages and failed pages are separate states
- failed pages remain visible in artifacts and reports
- cache files make reruns resumable
- fallback model usage is recorded
- difficult pages can be retried without discarding the rest of the run

This is especially important for archival work, where a false blank can be worse than an obvious failure.

## Verification Artifacts

The pipeline generates deterministic review artifacts to reduce manual inspection effort.

These include:
- blank-page list
- failed-page list
- fallback-page list
- low-confidence page list
- footnote integrity checks
- hotspot report
- source/output structural maps
- structural diff report
- markdown verification summary

This allows a reviewer to focus on risk points rather than re-reading the entire book output.

## Current Practical Notes

- Flash is currently the best default for cost and speed.
- Claude Haiku is useful for fallback and segmented recovery.
- Verification is still required even when extraction looks good locally.
- The strongest improvements in the project have come from better validation and reporting, not from trying to let models rewrite more.

## Documentation

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Runbook: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Goals and invariants: [docs/PROJECT_GOALS.md](docs/PROJECT_GOALS.md)
- Learnings: [docs/LEARNINGS.md](docs/LEARNINGS.md)
- Security notes: [docs/SECURITY.md](docs/SECURITY.md)
- Publishing checklist: [docs/PUBLISHING_CHECKLIST.md](docs/PUBLISHING_CHECKLIST.md)

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
