#!/usr/bin/env python3
"""
benchmark.py — measure PP-StructureV3 per-page inference time on this
machine, before committing to a full --tables run. Run this on any new
machine to get a real number instead of guessing.

Must run inside .venv-ocr (same environment paddle_structure_ocr.py uses):
    .venv-ocr/bin/python benchmark.py <image_dir> [--pages 3] [--lang pt]

Times a warmup page (excluded from the average — includes one-time model
load/JIT overhead) plus N more pages, then estimates full-book runtime at
different --workers values. The --workers estimate is a straight-line
projection (avg_time * pages / workers) — real scaling will be a bit worse
than that due to thread contention, but it's a reasonable first guess for
picking a starting N.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(directory: Path) -> list:
    images = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda p: p.name.lower())
    return images


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--pages", type=int, default=3, help="pages to time, excluding warmup (default: 3)")
    parser.add_argument("--lang", default="pt")
    parser.add_argument("--book-pages", type=int, default=200, help="book size to estimate for (default: 200)")
    args = parser.parse_args(argv)

    images = collect_images(args.image_dir)
    if len(images) < args.pages + 1:
        print(f"❌ need at least {args.pages + 1} images in {args.image_dir} (1 warmup + {args.pages} timed)", file=sys.stderr)
        return 1
    images = images[: args.pages + 1]

    print(f"Machine:  {platform.machine()} | {platform.system()} {platform.release()}")
    print(f"CPUs:     {os.cpu_count()} logical")
    print(f"Python:   {platform.python_version()}")
    print()
    print("Loading PP-StructureV3 models (one-time)...")

    from paddleocr import PPStructureV3

    engine = PPStructureV3(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )

    t0 = time.time()
    list(engine.predict(str(images[0])))
    print(f"  warmup page: {time.time() - t0:.1f}s (includes model init overhead — ignore this one)\n")

    times = []
    for image_path in images[1:]:
        t0 = time.time()
        list(engine.predict(str(image_path)))
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  {image_path.name}: {elapsed:.1f}s")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg:.1f}s/page")

    n = args.book_pages
    print(f"\nEstimated total time for a {n}-page book:")
    print(f"  --workers 1 (sequential): {avg * n / 60:.0f} min")
    for w in (2, 4, 8, 14, 28):
        if w >= (os.cpu_count() or 1) * 2:
            continue
        print(f"  --workers {w} (straight-line estimate, real scaling will be somewhat worse): {avg * n / 60 / w:.0f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
