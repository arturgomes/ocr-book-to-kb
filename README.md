# ocr-book-to-kb

OCR a directory of book-page screenshots (phone photos of a physical book) into markdown
ready for a [bookrag](https://github.com/) knowledge-base `build` pipeline — same
principles/arguments/frameworks/failure-modes/playbook schema as PDF/EPUB books. Never
stores raw OCR'd text as the final KB artifact; this only does the OCR → markdown step,
the LLM structuring/indexing/vault-note steps are a separate downstream pipeline (see
`SKILL.md` for the full Claude Code skill workflow).

Two OCR engines:

| | tesseract (default) | PaddleOCR PP-StructureV3 (`--tables`) |
|---|---|---|
| Speed | ~1s/page | ~100s/page (CPU) |
| Output | flat plain text | layout-aware: real headings/paragraphs, tables as markdown tables |
| Tables | scrambled runs of text | preserved as actual markdown tables |
| Figures | usually noise or lost | replaced with a `_[figure/diagram]_` placeholder (not OCR'd/captioned — stays text-only) |
| Setup | `brew install tesseract` | dedicated `.venv-ocr` (Python 3.12) |

Default to tesseract. Use `--tables` when a book has enough tables/diagrams/grids that flat
OCR loses real information (workbooks, textbooks with exercises, reference tables).

## Setup

```bash
git clone https://github.com/arturgomes/ocr-book-to-kb.git
cd ocr-book-to-kb
./install.sh
```

`install.sh` is idempotent — safe to re-run. It:
1. Checks/installs `tesseract` (Homebrew on macOS, apt on Debian/Ubuntu)
2. Creates `.venv-ocr` (Python 3.12 — paddlepaddle doesn't support 3.14+) and installs
   `paddlepaddle`, `paddleocr`, `paddlex[ocr]`, `pandas`
3. Installs `SKILL.md` to `~/.claude/skills/ocr-book-to-kb/` so Claude Code auto-discovers it
4. Adds an `ocr-to-kb()` shell function to `~/.zshrc` for one-line invocation

## Usage

```bash
# tesseract (default, fast)
python3 ocr_book_to_kb.py <image_dir> --domain <domain> --name <book-slug>

# PP-StructureV3 (tables/figures preserved, slow)
python3 ocr_book_to_kb.py <image_dir> --domain <domain> --name <book-slug> --tables

# PP-StructureV3 with page-level parallelism on multi-core machines
python3 ocr_book_to_kb.py <image_dir> --domain <domain> --name <book-slug> \
  --tables --workers 6
```

Output: `{kb_root}/domains/{domain}/markdown/{book-slug}/{book-slug}.md`
(`--kb-root` defaults to `~/Documents/Obsidian-Vault/05-Knowledge-Base`.)

Or, after `install.sh` + `source ~/.zshrc`, drive the whole thing (OCR → bookrag build →
indexing → vault note) through Claude Code in one line:
```bash
ocr-to-kb ~/Desktop/book-pages ml-data
```

### Benchmarking `--tables` on a new machine

`--tables` is slow enough (~100s/page) that it's worth checking a machine's actual speed
before committing to a multi-hour run:

```bash
.venv-ocr/bin/python benchmark.py <image_dir> --pages 3
```

Times a few real pages and estimates full-book runtime at several `--workers` values. Run
this first — thread contention means guessing a large `--workers` number can be *slower*
than a smaller one, especially on machines with fewer cores.

### `--workers N`

Only applies to `--tables`. Runs N pages concurrently in separate processes, each capped to
`cpu_count() // N` threads so workers don't all fight for every core at once. Each worker
loads its own copy of the PP-StructureV3 models (memory cost scales with N — watch RAM on
machines with less of it). Defaults to 1 (sequential, same behavior as before this flag
existed).

## How `--tables` avoids common OCR-to-table pitfalls

- **Figures**: PP-StructureV3 detects figure/diagram regions and would emit `<img>` tags
  pointing at crop files. This pipeline doesn't save those crops (stays text-only, same
  policy as the rest of the KB tooling) — they're replaced with a
  `_[figure/diagram on this page — not OCR'd]_` placeholder instead of leaking broken image
  tags into the markdown.
- **Tables**: PP-StructureV3 emits detected tables as embedded HTML
  (`<table>...</table>`), which is converted to a real markdown table via
  `pandas.read_html(..., thousands=None)`. The `thousands=None` matters — pandas defaults
  to treating `,` as a thousands separator, which silently corrupts decimal-comma notation
  common outside English locales (e.g. Portuguese "28,3" → "283"). Verified against a real
  Portuguese math textbook table during development.
- **Resumability**: each page's result caches to `{book-slug}/.paddle-cache/<page>.md` as
  it completes. A killed/interrupted multi-hour run picks up where it left off on re-run
  instead of starting over.

## Files

- `ocr_book_to_kb.py` — main CLI, both engines
- `paddle_structure_ocr.py` — PP-StructureV3 engine + `--workers` pool, run inside
  `.venv-ocr` as a subprocess of the main script
- `benchmark.py` — per-machine timing check, run inside `.venv-ocr`
- `install.sh` — one-shot setup (idempotent)
- `SKILL.md` — Claude Code skill definition (installed by `install.sh`)

## Status

Implemented and in use. Tested end-to-end against a real photographed Portuguese math
textbook (185 pages) with both engines; table-conversion correctness verified against a
real multi-row table with decimal notation.
