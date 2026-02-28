# Runbook

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate the environment variables you need.

## Basic Run

```bash
python run.py input/book.pdf
```

## Resume

```bash
python run.py input/book.pdf --resume
```

## Slice Testing

```bash
python run.py input/book.pdf --pages 1-50
```

Use this when:
- tuning prompts
- validating fallback behavior
- debugging chapter boundary issues
- testing difficult page ranges

## Strict Gate

```bash
python run.py input/book.pdf --strict
```

Use this when you want QA failures to block export.

## Important Runtime Artifacts

Inside a run directory, inspect:
- `pages/{book_id}/` for per-page markdown
- `.cache/{book_id}/` for checkpoint JSON
- `{book_id}_qa_report.json`
- `{book_id}_hotspot_report.json`
- `{book_id}_verification_report.md`
- `{book_id}_structural_diff_report.json`

## When A Page Fails

Check:
1. the per-page markdown file
2. the checkpoint JSON
3. the verification report
4. the run log

Important distinction:
- true blank pages should remain blank
- extraction failures should remain explicit failures

## When A Page Is Policy-Blocked

Current approach:
- full-page extraction first
- fallback model if available
- segmented LLM retry if the full page is blocked or empty

## Config Tuning

The main tuning surface is `config/default.yaml`.

Common knobs:
- model selection
- timeouts and retries
- extraction thinking level
- structure detection behavior
- assembly mode
- strict chapter detection

## Suggested Workflow For A New Book

1. run a small page slice
2. inspect page markdown and cache JSON
3. inspect chapter assembly
4. inspect verification artifacts
5. run full book once the slice looks stable
