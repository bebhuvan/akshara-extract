# Architecture

## Objective

Project Akshara is designed for faithful reconstruction of scanned books. The architecture is deliberately multi-pass because the failure modes of historical documents are structural, not just lexical.

The system optimizes for:
- completeness
- determinism
- reviewability
- bounded use of LLMs

It does not optimize first for:
- minimum latency
- minimum number of prompts
- freeform rewriting quality

## System Design

At a high level:

```text
PDF
  -> preflight page rendering
  -> page-level extraction
  -> assembly into chapters/book
  -> verification and structural diagnostics
  -> optional export
```

Each stage writes inspectable artifacts so the pipeline can be resumed and audited.

## Main Components

### `run.py`
Top-level orchestrator.

Responsibilities:
- loads environment variables
- loads YAML config
- derives a stable `book_id`
- allocates run directory
- initializes model clients
- coordinates all passes
- writes final reports

### `src/preflight.py`
Prepares the PDF for extraction.

Responsibilities:
- open PDF with PyMuPDF
- render pages to images
- normalize image output naming
- optionally apply image cleanup

Outputs:
- one page image per source page

### `src/extraction.py`
Core page-level transcription logic.

Responsibilities:
- prompt Gemini for page transcription
- detect blank-page commentary leakage
- classify page type
- record structural hints
- distinguish blank pages from failed pages
- retry blocked pages using segmented LLM extraction
- write per-page markdown and checkpoint JSON

This module defines the `PageExtraction` record, which is the central handoff object for later stages.

### `src/assembly.py`
Book assembly and verification layer.

Responsibilities:
- classify page roles for assembly
- group pages into chapter structures
- preserve and normalize footnotes
- emit archival and reader outputs
- apply deterministic cleanup
- apply validator-gated LLM operations where allowed
- generate QA, hotspot, and verification summaries

### `src/verification_maps.py`
Structural verification layer.

Responsibilities:
- build source maps from extracted pages
- build output maps from assembled markdown
- compute structural diffs

This is useful for:
- detecting likely omissions
- comparing archival and reader outputs
- seeding targeted repair loops

### `src/blank_page_filter.py`
Regex-based filter for model commentary.

Responsibilities:
- detect patterns like “this page is blank”
- prevent those lines from entering archival output

### Model Clients

#### `src/gemini_client.py`
Primary extraction client.

Responsibilities:
- text and image requests
- fallback persona retries
- retry/backoff
- token and cost tracking

#### `src/claude_client.py`
Fallback and supporting vision client.

Responsibilities:
- image requests
- retry/backoff
- cost tracking

#### `src/kimi_client.py`
Optional shadow / challenger client for assembly experiments.

### `src/export.py`
Optional downstream format conversion layer.

## Pass-by-Pass Technical Flow

## Pass 0: Preflight

Input:
- one PDF

Output:
- page images under `images/{book_id}/`

Technical details:
- uses PyMuPDF rendering
- page numbering remains stable across reruns
- image naming is deterministic
- large books are handled page by page rather than as a monolith

## Pass 1: Extraction

Input:
- page images

Output:
- page markdown files
- checkpoint JSON files
- in-memory `PageExtraction` list

Technical details:
- Gemini 3 Flash is the primary extractor
- structure detection is separate from plain text extraction
- blank-page commentary is filtered
- blocked full-page extractions can be retried through segmented LLM crops
- failures remain explicit errors

Important states:
- `blank`
- `text`
- `title`
- `toc`
- `frontmatter`
- `backmatter`
- `illustration`
- `error`

## Pass 2: Assembly

Input:
- ordered page extractions

Output:
- assembled markdown
- archival markdown companion

Technical details:
- deterministic page classification first
- chapter grouping second
- footnote extraction and normalization third
- bounded LLM planning only where useful
- validators remain in control of application

Why this matters:
- many assembly defects are not OCR defects
- running headers, opener duplication, and footnote linkage are assembly problems

## Pass 3: Verification

Input:
- assembled book
- page extractions

Output:
- QA report
- hotspot report
- verification markdown
- source/output maps
- structural diff report

Technical details:
- blank-page counts come from explicit page states
- failed pages are listed separately
- fallback usage is recorded
- footnote integrity is checked deterministically
- suspicious regions are surfaced for human review

## Pass 4: Export

Optional conversion layer.

Outputs may include:
- markdown
- EPUB
- DOCX
- PDF

## Data Structures

### `PageExtraction`

Fields include:
- `page_number`
- `text`
- `raw_text`
- `page_type`
- `has_footnotes`
- `has_indic_script`
- `has_tables`
- `has_poetry`
- `chapter_heading`
- `word_count`
- `confidence`
- `fidelity`
- `extraction_model`
- `warnings`
- `error`
- `success`

Operational meaning:
- this is the canonical per-page unit passed into assembly
- it is also checkpoint-serializable

### `AssembledBook`

Carries:
- title
- author
- chapters
- page count
- word count
- blank pages
- warnings
- pages needing review
- assembly operation ledger

## Run Isolation And Directory Model

Each execution writes into its own run directory. This is important for three reasons:

- intermediate artifacts stay inspectable
- interrupted runs can be resumed
- experiments on the same book do not overwrite each other

Typical layout:

```text
output/runs/full/{book_id}/{run_id}/
output/runs/tests/{book_id}/{run_id}/
```

Within a run:
- `images/{book_id}/` stores preflight renders
- `pages/{book_id}/` stores per-page markdown
- `.cache/{book_id}/` stores checkpoint JSON
- assembled and verification artifacts live at the run root

The design intent is that a run directory should be a complete forensic record of what happened.

## Resume Semantics

The pipeline is resumable because page extraction is checkpointed independently.

Operationally:
- successful page results are cached as JSON
- reruns can reuse completed work
- failed pages can be retried without reprocessing the entire book
- assembly and verification can be regenerated after extraction state changes

This is especially important for long books where model latency or provider failures make single-shot execution fragile.

## Failure Semantics

The architecture treats page states explicitly.

Important distinctions:
- `blank` means the source page is intentionally blank
- `error` means extraction failed or was blocked
- fallback use is metadata, not a hidden implementation detail

This matters because:
- false blanks hide data loss
- explicit failures can be routed to retry or review
- verification reports become trustworthy only when these states are separated

## LLM Routing Strategy

The system uses LLMs in layers rather than as one undifferentiated backend.

Current practical routing:
- Gemini 3 Flash for primary page extraction
- Claude Haiku for fallback on difficult pages
- segmented vision retries when a full-page request fails but the page still appears recoverable

The routing philosophy is conservative:
- use the fastest reliable primary model for the common case
- use a secondary model only when needed
- keep validation deterministic
- never let fallback behavior silently alter archival guarantees

## Assembly Philosophy

Assembly is not just string concatenation. It is where many book-specific failures show up:

- chapter openers can be duplicated
- running headers can leak into body text
- footnotes can detach from references
- contents pages and indexes need different handling from prose pages

Because of this, assembly is split into:
- deterministic classification and grouping
- bounded LLM planning where useful
- validator-gated application of risky operations

This lets the system benefit from model assistance without turning the book into a freeform rewrite.

## Verification Contract

The verification layer exists to answer a practical question:

`What should a human reviewer inspect first?`

To support that, the architecture emits:
- QA JSON for machine-readable review
- verification markdown for human review
- structural maps for source/output comparison
- hotspot reports for likely problem areas

This contract is a core part of the system, not a post-processing extra.

## Current Practical Conclusion

In current use:
- Gemini 3 Flash has worked well as the primary extraction model
- Claude Haiku has worked well as the fallback model
- the biggest quality gains have come from validation, reporting, and explicit failure handling

Project site: `https://akshara.ink`

## Run Directory Model

The pipeline uses unique run directories for isolation:

```text
output/runs/full/{book_id}/{run_id}/
output/runs/tests/{book_id}/{run_id}/
```

Benefits:
- reproducible reruns
- cache locality
- side-by-side experiment comparison
- clear audit trail per run

## Resume Model

Per-page checkpoints are stored under:

```text
.cache/{book_id}/page_0001.json
```

Resume behavior:
- successful pages can be loaded from cache
- reruns do not need to repeat the whole book
- targeted page regeneration is possible by removing specific cache files

## LLM Usage Philosophy

The project does use LLMs, but under constraints.

LLMs are used for:
- page transcription
- structure hints
- bounded planning tasks
- segmented fallback on blocked pages

LLMs are not trusted for:
- unrestricted chapter rewriting
- historical normalization
- silent gap filling

## Reliability Strategy

### Deterministic first
Deterministic code should own the final shape of the book wherever possible.

### Reviewable artifacts
Every major stage writes inspectable artifacts.

### Explicit failure states
Blank pages and failed pages are separate.

### Bounded fallback paths
Fallback behavior is recorded, not hidden.

## Current Practical Conclusion

At the moment:
- Gemini 3 Flash is the strongest default primary extractor in this repository
- Claude Haiku is an effective secondary model in difficult cases
- the best quality gains have come from verification and fallback design, not simply from increasing prompt complexity
