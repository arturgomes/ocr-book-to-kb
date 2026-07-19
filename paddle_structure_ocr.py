#!/usr/bin/env python3
"""
paddle_structure_ocr.py — layout-aware page transcription via PaddleOCR's
PP-StructureV3, for books where tables/figures matter enough to justify the
cost (Tesseract mangles table layout into scrambled flat text).

Must run inside the dedicated .venv-ocr (Python 3.12) environment — see
install.sh / README.md for setup.

Slow: ~100s/page on CPU (layout detection + text rec + table structure rec
run as separate models per page). Each page's markdown is cached to
<cache_dir>/<page-stem>.md so an interrupted run resumes instead of
restarting from page 1.

--workers N runs N pages concurrently in separate processes, each capped to
a fraction of the machine's threads (os.cpu_count() // N) to avoid every
worker fighting for all cores at once. Worth using on many-core machines;
on a low-core-count machine (e.g. a laptop with 8 threads) workers beyond
2-3 usually just cause thread contention instead of speeding things up —
run benchmark.py first to check.

Usage:
    .venv-ocr/bin/python paddle_structure_ocr.py <image_dir> <cache_dir> --lang pt
    .venv-ocr/bin/python paddle_structure_ocr.py <image_dir> <cache_dir> --lang pt --workers 6

Progress goes to stderr (for live monitoring on long runs); the final
{"pages": [...]} JSON goes to stdout.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from io import StringIO
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# PP-StructureV3 emits <img> tags pointing at crop files we don't save (this
# pipeline is text-only, same as the rest of the KB — see ocr_book_to_kb.py).
_IMG_TAG_RE = re.compile(r"<div[^>]*>\s*<img[^>]*>\s*</div>|<img[^>]*/?>")

# Tables come out as a <div><html><body><table>...</table></body></html></div>
# wrapper — capture the <table> so it can be converted to a real markdown table.
_TABLE_BLOCK_RE = re.compile(
    r"<div[^>]*>\s*(?:<html>\s*<body>\s*)?(<table\b.*?</table>)(?:\s*</body>\s*</html>)?\s*</div>",
    re.DOTALL | re.IGNORECASE,
)

# Set per-worker in _init_worker() before paddle is imported in that process;
# left at their defaults (whole machine) in the sequential (--workers 1) path.
_ENGINE = None


def collect_images(directory: Path) -> list:
    images = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda p: p.name.lower())
    return images


def _table_html_to_markdown(match: re.Match) -> str:
    import pandas as pd

    html = match.group(1)
    try:
        # thousands=None: pandas defaults to treating ',' as a thousands
        # separator, which silently mangles Portuguese decimal notation,
        # e.g. "28,3" -> "283". Book tables are prose, not numeric data.
        df = pd.read_html(StringIO(html), thousands=None)[0]
        return "\n" + df.to_markdown(index=False) + "\n"
    except Exception:
        return html  # leave raw HTML if pandas can't parse it — still readable, not lost


def clean_markdown(text: str) -> str:
    text = _TABLE_BLOCK_RE.sub(_table_html_to_markdown, text)
    text = _IMG_TAG_RE.sub("\n_[figure/diagram on this page — not OCR'd]_\n", text)
    return text


def _build_engine(lang: str):
    from paddleocr import PPStructureV3

    return PPStructureV3(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )


def _ocr_one(engine, image_path: Path) -> str:
    result = list(engine.predict(str(image_path)))[0]
    return clean_markdown(result.markdown.get("markdown_texts", ""))


def _init_worker(lang: str, threads_per_worker: int) -> None:
    # Must happen before paddle is imported in this (forked/spawned) process,
    # otherwise it locks in the default (all-cores) thread count.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CPU_NUM"):
        os.environ[var] = str(threads_per_worker)
    global _ENGINE
    _ENGINE = _build_engine(lang)


def _process_one(task: tuple) -> tuple:
    index, image_path_str, cache_file_str, total = task
    cache_file = Path(cache_file_str)
    t0 = time.time()
    text = _ocr_one(_ENGINE, Path(image_path_str))
    cache_file.write_text(text, encoding="utf-8")
    elapsed = time.time() - t0
    print(
        f"  [{index + 1}/{total}] {Path(image_path_str).name} ({elapsed:.0f}s) [pid {os.getpid()}]",
        file=sys.stderr,
    )
    return index, text


def run(image_dir: Path, cache_dir: Path, lang: str, workers: int) -> list:
    images = collect_images(image_dir)
    if not images:
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)

    pages = [None] * len(images)
    todo = []
    for i, image_path in enumerate(images):
        cache_file = cache_dir / f"{image_path.stem}.md"
        if cache_file.exists():
            pages[i] = cache_file.read_text(encoding="utf-8")
            print(f"  [{i + 1}/{len(images)}] {image_path.name} (cached)", file=sys.stderr)
        else:
            todo.append((i, image_path, cache_file))

    if not todo:
        return pages

    if workers <= 1:
        print("  loading PP-StructureV3 models (first page only)...", file=sys.stderr)
        engine = _build_engine(lang)
        for i, image_path, cache_file in todo:
            t0 = time.time()
            text = _ocr_one(engine, image_path)
            cache_file.write_text(text, encoding="utf-8")
            pages[i] = text
            print(f"  [{i + 1}/{len(images)}] {image_path.name} ({time.time() - t0:.0f}s)", file=sys.stderr)
        return pages

    total_threads = os.cpu_count() or workers
    threads_per_worker = max(1, total_threads // workers)
    print(
        f"  {workers} workers x {threads_per_worker} threads each "
        f"(of {total_threads} logical CPUs) — loading models per worker...",
        file=sys.stderr,
    )
    tasks = [(i, str(image_path), str(cache_file), len(images)) for i, image_path, cache_file in todo]
    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(lang, threads_per_worker)) as pool:
        for index, text in pool.imap_unordered(_process_one, tasks):
            pages[index] = text
    return pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--lang", default="pt", help="PaddleOCR language code (default: pt)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="pages to process concurrently (default: 1 = sequential). "
        "Run benchmark.py first to pick a sane number for this machine.",
    )
    args = parser.parse_args(argv)

    pages = run(args.image_dir, args.cache_dir, args.lang, args.workers)
    print(json.dumps({"pages": pages}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
