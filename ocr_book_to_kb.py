#!/usr/bin/env python3
"""
ocr_book_to_kb.py — OCR book-page screenshots into the markdown format the
bookrag `build` pipeline expects, so photographed book pages go through the
SAME structured distillation as PDF/EPUB books (00_book_summary.md,
01_core_principles.md, 02_arguments.md, ...) instead of being stored as a raw
verbatim text dump. This script only does the OCR → markdown step; the actual
KB structuring/indexing/vault-note steps are the existing bookrag pipeline,
run by the ocr-book-to-kb skill (mirrors add-pdf-to-kb's Stage 1 vs
Stage 2/3 split).

Default engine: tesseract (same approach as ~/.claude/scripts/ocr_images.py) —
fast (~1s/page), flattens everything to plain text, so tables come out as
scrambled runs of text and figures are usually noise or missing.

Optional engine: PaddleOCR's PP-StructureV3 (--tables), a layout-aware
pipeline that renders tables as real markdown tables and keeps headings/
paragraphs intact. Much slower (~100s/page on CPU), runs in a dedicated
.venv-ocr (Python 3.12) via paddle_structure_ocr.py. Use it when a book has
enough tables/diagrams that flat OCR loses real information. Run install.sh
once to set up .venv-ocr, or see README.md for manual setup.

--workers N (only with --tables) processes N pages concurrently in separate
processes on multi-core machines. Run benchmark.py first to see whether this
is worth it on a given machine and pick a sane N.

Usage:
    python ocr_book_to_kb.py <image_dir> --domain <domain> --name <book-slug>
    python ocr_book_to_kb.py <image_dir> --domain ml-data --name my-book --title "My Book"
    python ocr_book_to_kb.py <image_dir> --domain ml-data --name my-book --tables --workers 6

Output:
    {kb_root}/domains/{domain}/markdown/{book-slug}/{book-slug}.md

    Ready to feed straight into the existing bookrag structuring pipeline:
    uv run --directory ~/Documents/ai-tools/skills-mono-repo bookrag build \\
      --input {kb_root}/domains/{domain}/markdown/{book-slug} \\
      --domain {domain} --name {book-slug} \\
      --settings {kb_root}/config/settings.toml
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_KB_ROOT = Path.home() / "Documents" / "Obsidian-Vault" / "05-Knowledge-Base"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
PADDLE_VENV_PYTHON = SCRIPT_DIR / ".venv-ocr" / "bin" / "python3"
PADDLE_HELPER = SCRIPT_DIR / "paddle_structure_ocr.py"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def collect_images(directory: Path) -> list:
    images = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda p: p.name.lower())
    return images


def ocr_pages(images: list, lang: str) -> list:
    """Returns a list of page texts, one per image, in order."""
    pages = []
    for image_path in images:
        print(f"  OCR: {image_path.name}")
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", lang, "--psm", "3"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pages.append(f"[OCR error on {image_path.name}: {result.stderr.strip()}]")
        else:
            pages.append(result.stdout.strip())
    return pages


def ocr_pages_structure(image_dir: Path, cache_dir: Path, lang: str, workers: int) -> list:
    """Layout-aware OCR via PP-StructureV3, run inside .venv-ocr. Slow but
    preserves tables (as markdown tables) and headings/paragraphs instead of
    flattening the page to scrambled text."""
    if not PADDLE_VENV_PYTHON.exists():
        print("❌ .venv-ocr not found. Set it up with:", file=sys.stderr)
        print(f"  cd {SCRIPT_DIR} && ./install.sh", file=sys.stderr)
        print("  (or see README.md for manual setup)", file=sys.stderr)
        raise SystemExit(1)

    print(f"  Running PP-StructureV3 (slow, ~100s/page, cached in {cache_dir.name}/)...")
    result = subprocess.run(
        [
            str(PADDLE_VENV_PYTHON), str(PADDLE_HELPER), str(image_dir), str(cache_dir),
            "--lang", lang, "--workers", str(workers),
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("❌ PP-StructureV3 OCR failed", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(result.stdout)["pages"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR book-page screenshots into bookrag-ready markdown."
    )
    parser.add_argument("image_dir", type=Path, help="Directory of book-page screenshot images")
    parser.add_argument(
        "--domain", required=True, help="Destination KB domain (e.g. ml-data, software-craft)"
    )
    parser.add_argument(
        "--name", default=None, help="Book slug (default: image_dir folder name, slugified)"
    )
    parser.add_argument("--title", default=None, help="Human title (default: --name)")
    parser.add_argument(
        "--kb-root",
        type=Path,
        default=DEFAULT_KB_ROOT,
        help=f"KB root directory (default: {DEFAULT_KB_ROOT})",
    )
    parser.add_argument("--lang", default="por+eng", help="tesseract --lang value (default: por+eng)")
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Use PP-StructureV3 instead of tesseract — preserves tables/headings, "
        "much slower (~100s/page). Requires .venv-ocr, see module docstring.",
    )
    parser.add_argument(
        "--paddle-lang",
        default="pt",
        help="PaddleOCR language code, only used with --tables (default: pt)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="pages to process concurrently, only used with --tables (default: 1 = "
        "sequential). Run benchmark.py first to pick a sane number for this machine.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        print(f"❌ not a directory: {image_dir}", file=sys.stderr)
        return 1

    images = collect_images(image_dir)
    if not images:
        print(f"❌ no images found in {image_dir}", file=sys.stderr)
        return 1

    slug = args.name or slugify(image_dir.name)
    title = args.title or slug

    print(f"Found {len(images)} image(s) in {image_dir}")

    out_dir = args.kb_root / "domains" / args.domain / "markdown" / slug

    if args.tables:
        cache_dir = out_dir / ".paddle-cache"
        pages = ocr_pages_structure(image_dir, cache_dir, args.paddle_lang, args.workers)
    else:
        pages = ocr_pages(images, args.lang)

    markdown = f"# {title}\n\n" + "\n\n---\n\n".join(pages) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.md"
    out_path.write_text(markdown, encoding="utf-8")

    print(f"\n✅ OCR complete: {len(images)} pages -> {out_path}")
    print("\nNext (structures this into principles/arguments/frameworks — not a raw dump):")
    print(
        f"  uv run --directory ~/Documents/ai-tools/skills-mono-repo bookrag build \\\n"
        f"    --input {out_dir} --domain {args.domain} --name {slug} \\\n"
        f"    --settings {args.kb_root}/config/settings.toml"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
