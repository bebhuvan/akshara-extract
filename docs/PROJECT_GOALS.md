# Project Goals And Invariants

## Mission

Project Akshara aims to make public-domain scanned books usable as living texts again without corrupting the historical artifact.

Public site: `https://akshara.ink`

## Non-Negotiable Invariants

1. Preserve all source content that belongs to the book.
2. Do not add text that is not grounded in the scan.
3. Do not modernize spelling, punctuation, or wording.
4. Preserve structure, headings, tables, verse, and notes as faithfully as practical.
5. Preserve Indic scripts exactly as printed.
6. Surface uncertainty rather than guessing.

## What Must Be Preserved

- title pages
- publication metadata
- prefaces and introductions
- chapters and sections in order
- footnotes and endnotes
- appendices and bibliographies
- indexes and backmatter
- textual marginalia where present

## What The System Must Not Do

- summarize
- paraphrase
- editorialize
- normalize archaic language
- infer missing text as if certain
- silently drop difficult sections

## Why A Multi-Pass Pipeline Exists

Large scanned books fail in subtle ways:
- chunk boundaries can lose content
- footnotes can detach from references
- running headers can leak into prose
- blank pages and failures can be confused

A conservative multi-pass design exists to catch those failures and make them reviewable.
