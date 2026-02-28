# Learnings

## 1. Archival work is not generic OCR

Historical books break in ways that simple OCR metrics do not capture:
- chapter boundaries can be misdetected
- running headers can leak into prose
- footnotes can detach from references
- structured pages like contents and indexes require different handling from prose pages

This project therefore uses a multi-pass pipeline instead of a single extraction prompt.

## 2. Blank pages and failed pages must be separate states

One of the most important corrections in the pipeline was separating:
- true blank pages
- extraction failures

If a blocked or failed page is mislabeled as blank, the review trail becomes unreliable and silent data loss becomes easier to miss.

## 3. Gemini 3 Flash works well as the primary extraction model

In current use, Flash has been a strong default because it is:
- fast
- cost-efficient
- good enough for large-scale page transcription

For many pages, it produces usable archival text directly with minimal intervention.

## 4. Claude Haiku is a useful fallback

Claude Haiku has worked well as a secondary vision path in cases where:
- the first extraction pass returns empty output
- a second model perspective is helpful
- segmented page retries are needed

It is not a replacement for deterministic validation, but it has been useful in difficult cases.

## 5. Deterministic validation matters more than adding more prompts

When pages fail, the best fix is rarely just “more prompting.” The durable improvements have come from:
- better state tracking
- better reporting
- better page classification
- better fallback routing
- better validation

Prompt changes help at the margin. Validation changes improve the system.

## 6. Segmented LLM retries can recover blocked full-page extractions

Some pages that fail as full-page requests can be recovered by splitting the page into vertical segments and retrying with the same vision models. This has worked better than accepting false blanks and avoids dropping directly into low-quality OCR.

## 7. Verification artifacts are essential

The pipeline is strongest when it produces inspectable outputs such as:
- per-page markdown
- per-page checkpoint JSON
- QA reports
- hotspot reports
- verification summaries

This makes the system auditable and lets a reviewer focus on high-risk pages.

## 8. Assembly needs both deterministic logic and bounded LLM assistance

The most effective pattern so far has been:
- deterministic assembly for the core structure
- constrained LLM assistance for targeted planning tasks
- validator-gated application of model suggestions

This keeps the system flexible without letting freeform rewriting corrupt the source.

## 9. The right metric is publishable fidelity, not raw model output quality

A page-level result can look acceptable but still create downstream damage if:
- footnotes are malformed
- a heading is misclassified
- a running header leaks into the body
- a false blank enters the verification record

The quality bar is therefore end-to-end integrity, not just local extraction quality.
