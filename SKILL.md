---
name: ocr-book-to-kb
description: >
  Turn a directory of book-page screenshots (phone photos of a physical book) into a
  structured knowledge-base entry — same principles/arguments/frameworks/failure-modes/
  playbook schema as PDF/EPUB books, never a raw verbatim text dump. OCR via tesseract
  by default, or PaddleOCR's PP-StructureV3 (--tables, optionally --workers N for
  multi-core machines) when tables/diagrams matter enough to justify the much slower
  layout-aware pass. Domain is always asked, never assumed.
  Trigger phrases: "OCR these book screenshots into my KB", "add this book from photos",
  "turn these page photos into a KB entry", "I have screenshots of a book to add",
  "this book has tables/diagrams that need to come through cleanly".
---

# ocr-book-to-kb

Automated workflow for adding a physically-photographed book to your knowledge base.
Mirrors `add-pdf-to-kb`'s Stage 1 (mechanics) / Stage 2 (Claude-driven structuring) split —
the only difference is Stage 1 starts from images + OCR instead of a PDF/EPUB file.

**Never stores raw OCR text as the KB artifact.** The OCR output is an intermediate file
that gets distilled into principles/arguments/frameworks/failure-modes by the same
`bookrag build` pipeline every other book in this KB goes through — quotes in the final
notes are short and cited, not full pages.

Repo: `~/Documents/ai-tools/ocr-book-to-kb`. First time on a new machine, run
`./install.sh` there once — checks/installs tesseract, sets up `.venv-ocr` for the
`--tables` engine, installs this skill, and adds the `ocr-to-kb` shell function.

## Workflow

### Step 1 — Locate the Screenshots

Accept a directory path from the user containing the book-page images (jpg/png/etc, one
file per page). Confirm the page count looks right (`ls <dir> | wc -l`) before proceeding —
catches an accidental wrong-folder mistake before burning an OCR pass on it.

### Step 2 — Gather Metadata and Ask for the Domain

1. Infer **title/author** from the folder name or the first page's OCR'd title/cover text
   if visible — confirm with the user, don't guess silently.
2. **Always ask which domain** this book belongs in — never default or infer silently:
   - `engineering-practices`, `software-architecture`, `engineering-leadership`,
     `llm-engineering`, `software-craft`, `functional-programming`, `ml-data`, or a new domain
   - If the domain doesn't exist yet, register it first (same as `add-pdf-to-kb` Step 2):
     ```bash
     uv run --directory ~/Documents/ai-tools/skills-mono-repo \
       bookrag domain-create {domain-name} \
       --display-name "{display name}" \
       --keywords "{comma,separated,keywords}" \
       --description "{brief description}" \
       --settings ~/Documents/Obsidian-Vault/05-Knowledge-Base/config/settings.toml
     ```
3. Generate the book slug: `{title-slug}` (kebab-case).
4. **Confirm with user**: "Adding '{title}' by {author} to domain '{domain}', OCR'd from
   {N} screenshots. Proceed?"

### Step 3 — Run OCR

**Ask which engine** if the book's content isn't obviously plain prose — tables, grids,
forms, or diagram-heavy pages (workbooks, textbooks with exercises, reference tables) lose
real information under flat OCR. Default to tesseract; use `--tables` when the user says
tables/figures matter, or after monitoring shows a page's table/diagram content came out
scrambled.

**Default (tesseract, fast, ~1s/page):**
```bash
cd ~/Documents/ai-tools/ocr-book-to-kb
python3 ocr_book_to_kb.py "{image_dir}" \
  --domain "{domain}" --name "{book-slug}" --title "{title}"
```
Uses `tesseract` (default lang `por+eng` — pass `--lang` to override). Flattens each page
to plain text — tables come out as scrambled runs of words, figures are usually noise or
lost.

**Table/figure-aware (PaddleOCR PP-StructureV3, slow, ~100s/page):**
```bash
cd ~/Documents/ai-tools/ocr-book-to-kb
python3 ocr_book_to_kb.py "{image_dir}" \
  --domain "{domain}" --name "{book-slug}" --title "{title}" \
  --tables --paddle-lang pt
```
Layout-aware — real headings/paragraphs, and tables come out as actual markdown tables
instead of scrambled text. Figures are replaced with a `_[figure/diagram on this page]_`
placeholder (not OCR'd, not captioned — this pipeline stays text-only, same as everywhere
else in this KB). `--paddle-lang` is a PaddleOCR language code (`pt`, `en`, ...), separate
from tesseract's `--lang`.

~100s/page means a 100+ page book takes hours on a single core — **warn the user before
starting** and let them decide if it's worth it for this book. Each page's result is cached
under `{book-slug}/.paddle-cache/` as it completes, so an interrupted run resumes from
where it left off instead of restarting — safe to re-run the same command if it gets killed.

**On a multi-core machine, add `--workers N`** to process N pages concurrently (separate
processes, each capped to a slice of the machine's threads):
```bash
python3 ocr_book_to_kb.py "{image_dir}" --domain "{domain}" --name "{book-slug}" \
  --tables --paddle-lang pt --workers 6
```
**Run `benchmark.py` first** to check whether this machine benefits and to pick N — don't
just guess a large number, thread contention can make too-high a worker count *slower*
than fewer workers:
```bash
.venv-ocr/bin/python benchmark.py "{image_dir}" --pages 3
```
This times a few real pages and prints an estimated full-book runtime at several
`--workers` values.

Requires a one-time setup (separate from tesseract, only needed the first time `--tables`
is used) — `./install.sh` handles this, or manually:
```bash
cd ~/Documents/ai-tools/ocr-book-to-kb
python3.12 -m venv .venv-ocr
.venv-ocr/bin/pip install paddlepaddle paddleocr "paddlex[ocr]" pandas tabulate
# on Linux x86_64 pin paddle instead:
.venv-ocr/bin/pip install "paddlepaddle==3.0.0" paddleocr "paddlex[ocr]" pandas tabulate
```
(paddlepaddle needs Python ≤3.13 — use `python3.12` explicitly if the system default is
newer.) First `--tables`/`benchmark.py` run also downloads ~1-2GB of PP-StructureV3
models; that download counts against the ~100s/page estimate for page 1 only, then is
cached in `~/.paddlex/`.

**Two dependency traps, both verified the hard way on a Linux x86_64 box:**

- **`paddlepaddle==3.0.0` is required on Linux x86_64**, not preferred. 3.3.1's oneDNN
  kernel raises `NotImplementedError: ConvertPirAttribute2RuntimeAttribute` on the
  layout-detection model, and passing `enable_mkldnn=False` to work around it costs **6x
  the time and 3.4x the peak memory for byte-identical output** (9.0s/3.4GB vs
  56.6s/11.7GB on the same page). At 11.7GB/worker even 3 workers OOM a 16GB machine.
  macOS/ARM has no oneDNN path and is unaffected.
- **`tabulate` is not optional.** Without it `df.to_markdown()` raises ImportError, the
  per-table `except Exception` swallows it, and every table in the book silently lands as
  raw HTML — the exact failure `--tables` was chosen to avoid. `paddle_structure_ocr.py`
  now preflights for it and refuses to start rather than degrade quietly.

Output (either engine): `05-Knowledge-Base/domains/{domain}/markdown/{book-slug}/{book-slug}.md`

**Monitor for**: pages that OCR'd mostly blank or garbled (check the output file — a page
with <10 words when the source photo clearly has a paragraph is a signal to re-photograph
that page rather than let bad OCR feed into structuring). With tesseract specifically, a
page that's mostly numbers/short fragments in a grid-like pattern is a signal it contains a
table — worth flagging to the user as a `--tables` re-run candidate rather than treating it
as a photo-quality problem.

### Step 4 — Run bookrag build (same pipeline as add-pdf-to-kb)

```bash
uv run --directory ~/Documents/ai-tools/skills-mono-repo \
  bookrag build \
  --input "05-Knowledge-Base/domains/{domain}/markdown/{book-slug}" \
  --domain "{domain}" \
  --name "{book-slug}" \
  --settings ~/Documents/Obsidian-Vault/05-Knowledge-Base/config/settings.toml
```

This is the exact same Stage 2/3 pipeline used for PDF/EPUB books — LLM structuring into
`00_book_summary.md` / `01_core_principles.md` / `02_arguments.md` / `03_frameworks_models.md`
/ `04_failure_modes.md` / `05_application_playbook.md`, then indexing into `bookrag.db`.
Requires `ANTHROPIC_API_KEY` (same as `add-pdf-to-kb`).

**Handle failures**: identical to `add-pdf-to-kb`'s Error Handling section (missing API key,
domain not found, `uv` not installed).

### Step 5 — Verify Indexing

```bash
uv run --directory ~/Documents/ai-tools/skills-mono-repo \
  bookrag query-hybrid "key topic from the book" \
  --domain "{domain}" \
  --settings ~/Documents/Obsidian-Vault/05-Knowledge-Base/config/settings.toml \
  --stdout
```

### Step 6 — Rebuild Master ask-kb Index (Required)

Without this, the book won't appear in `/ask-kb` results:
```bash
uv run --directory ~/Documents/ai-tools/skills-mono-repo \
  bookrag obsidian-ingest \
  --vault-path ~/Documents/Obsidian-Vault \
  --db ~/Documents/ai-tools/skills-mono-repo/master-kb/domains/obsidian-vault/bookrag.db \
  --settings ~/Documents/ai-tools/skills-mono-repo/bookrag/config/settings.toml
```
Takes 20–60 minutes.

### Step 7 — Create Obsidian Vault Reference Note

Same as `add-pdf-to-kb` Step 5 — `mcp__ultimate-obsidian__create_or_update_note`:
- **filepath**: `01-Reference/Books/{book-slug}.md`
- **content**: same template as `add-pdf-to-kb`, plus a `source: {N} page photos (OCR'd via {engine})`
  frontmatter field instead of `source: {filename}.pdf`, so it's clear this book's KB entry
  came from photos, not a clean digital source (useful context if OCR quality questions come up later).

### Step 8 — Report to User

```
✅ Successfully added to your knowledge base!

Book:       {Title} by {Author}
Domain:     {domain}
Source:     {N} page photos, OCR'd via {tesseract | PaddleOCR PP-StructureV3}
DB:         ~/Documents/Obsidian-Vault/05-Knowledge-Base/domains/{domain}/bookrag.db
Vault note: 01-Reference/Books/{book-slug}.md

Note: Run Step 6 to make this book findable via /ask-kb.
```

## Error Handling

**No images found**:
```
❌ no images found in <dir>
```

**tesseract not installed**:
```
❌ tesseract not found.
Install: brew install tesseract   (or ./install.sh)
```

**`.venv-ocr` not found** (`--tables` used without one-time setup):
```
❌ .venv-ocr not found. Set it up with:
  cd ~/Documents/ai-tools/ocr-book-to-kb && ./install.sh
```

**Low-quality OCR page detected** (Step 3 monitoring):
```
⚠️ Page {N} OCR'd to under 10 words — likely a blurry/skewed photo.
Consider re-photographing and re-running before Step 4 structures it.
```

Other failures (API key, domain not found, uv missing): identical to `add-pdf-to-kb`'s
Error Handling section.

## Dependencies

**Required**:
- `tesseract`: `brew install tesseract` (or `./install.sh`)
- `uv` + `bookrag` (same as `add-pdf-to-kb`)
- `ANTHROPIC_API_KEY` (structuring stage)
- **ultimate-obsidian MCP** (Step 7)

**Optional** (only for `--tables`):
- `.venv-ocr` — dedicated Python 3.12 venv with `paddlepaddle` (**pinned to `==3.0.0` on
  Linux x86_64**), `paddleocr`, `paddlex[ocr]`, `pandas`, **`tabulate`** (required — its
  absence silently turns every table into raw HTML). paddlepaddle doesn't support Python
  3.14, hence the separate venv instead of the system/project Python — set up by
  `./install.sh`, which applies the platform pin and verifies every import.

## Quality Assurance

Before finalizing, verify:
- [ ] Page count in OCR output matches the number of source screenshots
- [ ] No page OCR'd to near-empty text without a corresponding blank source page
- [ ] Domain was explicitly confirmed with the user, never assumed
- [ ] `books.yaml`/`kb-registry.yaml` updated with the new book entry
- [ ] Sample `bookrag query-hybrid` returns relevant, distilled chunks — not raw page text
- [ ] Vault reference note created, `source:` field reflects OCR provenance

## Example Interaction

```
User: I have screenshots of a book, help me OCR them into my KB — they're in ~/Desktop/book-pages

Claude:
1. ✓ Found 142 images in ~/Desktop/book-pages
2. Which domain should this go in? (engineering-practices, ml-data, software-craft, ...)
User: ml-data
3. Adding to domain 'ml-data'. Proceed? — yes
4. Running OCR (tesseract)... 142/142 pages
   ⚠️ Page 87 OCR'd to 3 words — may be blurry, continuing anyway
5. Running bookrag build... Stage 2 (structuring) ✅ Stage 3 (indexing) ✅
6. ✓ Sample query returns relevant chunks
7. ✓ Vault note: 01-Reference/Books/{slug}.md

✅ Successfully added to your knowledge base!
```

## Example Interaction (tables matter, multi-core machine)

```
User: OCR ~/Desktop/math-workbook into educacao-matematica — it's full of tables and grids

Claude:
1. ✓ Found 185 images in ~/Desktop/math-workbook
2. Domain: educacao-matematica. Proceed? — yes
3. This book has tables — tesseract will scramble them. Using --tables (PaddleOCR
   PP-StructureV3) instead: ~100s/page. Let me check this machine's speed first.
   Running benchmark.py... avg 42s/page. Sequential: ~130 min. With --workers 6: ~25 min.
   Use --workers 6? — yes
4. Running OCR (PP-StructureV3, 6 workers)... 185/185 pages (26 min)
5. Running bookrag build... Stage 2 ✅ Stage 3 ✅
6. ✓ Sample query returns relevant chunks, tables render correctly
7. ✓ Vault note: 01-Reference/Books/{slug}.md

✅ Successfully added to your knowledge base!
```
