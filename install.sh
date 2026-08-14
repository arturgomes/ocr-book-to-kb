#!/usr/bin/env bash
# install.sh — one-shot setup for ocr-book-to-kb on a new machine:
#   1. checks/installs tesseract (default OCR engine)
#   2. creates .venv-ocr with PaddleOCR PP-StructureV3 (--tables engine)
#   3. installs the Claude Code skill under ~/.claude/skills/
#   4. adds the ocr-to-kb shell function to ~/.zshrc (idempotent)
#
# Safe to re-run — every step skips work that's already done.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$HOME/.claude/skills/ocr-book-to-kb"
ZSHRC="$HOME/.zshrc"

echo "==> Checking tesseract..."
if ! command -v tesseract >/dev/null 2>&1; then
  echo "tesseract not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing via Homebrew..."
    brew install tesseract tesseract-lang
  elif command -v apt-get >/dev/null 2>&1; then
    echo "Installing via apt..."
    sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-por
  else
    echo "❌ Could not auto-install tesseract on this system." >&2
    echo "   Install it manually, then re-run this script." >&2
    exit 1
  fi
else
  echo "  tesseract OK ($(tesseract --version 2>&1 | head -1))"
fi

echo "==> Setting up .venv-ocr (PaddleOCR PP-StructureV3, Python 3.12)..."
PY312="$(command -v python3.12 || true)"
if [ -z "$PY312" ]; then
  echo "❌ python3.12 not found (paddlepaddle doesn't support 3.14+)." >&2
  echo "   macOS: brew install python@3.12" >&2
  echo "   Linux: use your distro's package for python3.12, or pyenv install 3.12" >&2
  exit 1
fi

if [ ! -d "$REPO_DIR/.venv-ocr" ]; then
  "$PY312" -m venv "$REPO_DIR/.venv-ocr"
  echo "  created .venv-ocr"
else
  echo "  .venv-ocr already exists, reusing"
fi

"$REPO_DIR/.venv-ocr/bin/pip" install -q --upgrade pip

# paddlepaddle is pinned to 3.0.0 on Linux x86_64. On that platform 3.3.1's oneDNN kernel
# raises NotImplementedError (ConvertPirAttribute2RuntimeAttribute) on the layout-detection
# model, and the enable_mkldnn=False workaround costs 6x the time and 3.4x the peak memory
# for byte-identical output (measured: 9.0s/3.4GB vs 56.6s/11.7GB on the same page).
# macOS/ARM has no oneDNN path and is unaffected, so it stays on the latest release.
PADDLE_SPEC="paddlepaddle"
if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
  PADDLE_SPEC="paddlepaddle==3.0.0"
  echo "  Linux x86_64 detected — pinning $PADDLE_SPEC (3.3.1 breaks on oneDNN)"
fi

# tabulate is not optional: without it pandas' df.to_markdown() raises ImportError, which
# paddle_structure_ocr's per-table `except Exception` swallows — every table in the book
# then lands as raw HTML instead of markdown, with nothing in the log to say so.
"$REPO_DIR/.venv-ocr/bin/pip" install -q "$PADDLE_SPEC" paddleocr "paddlex[ocr]" pandas tabulate
echo "  .venv-ocr dependencies installed"

echo "==> Verifying the --tables toolchain imports..."
"$REPO_DIR/.venv-ocr/bin/python" - <<'PYEOF'
import sys

missing = []
for mod in ("paddle", "paddleocr", "pandas", "tabulate"):
    try:
        __import__(mod)
    except ImportError as exc:
        missing.append(f"{mod} ({exc})")
if missing:
    print("❌ --tables toolchain incomplete: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)

import paddle

print(f"  paddlepaddle {paddle.__version__}, pandas + tabulate OK")
PYEOF
echo "  (PP-StructureV3 models — ~1-2GB — download on first --tables/benchmark.py run, then cache in ~/.paddlex/)"

echo "==> Installing Claude Code skill..."
mkdir -p "$SKILLS_DIR"
cp "$REPO_DIR/SKILL.md" "$SKILLS_DIR/SKILL.md"
echo "  installed at $SKILLS_DIR/SKILL.md"

echo "==> Adding ocr-to-kb() shell function..."
MARKER="# ocr-book-to-kb (installed by install.sh)"
if [ -f "$ZSHRC" ] && grep -qF "$MARKER" "$ZSHRC"; then
  echo "  already present in $ZSHRC, skipping"
else
  cat >> "$ZSHRC" <<'EOF'

# ocr-book-to-kb (installed by install.sh)
ocr-to-kb() {
  local image_dir="$1"
  local domain="$2"
  if [[ -z "$image_dir" || -z "$domain" ]]; then
    echo "Usage: ocr-to-kb <image_dir> <domain>"
    echo "  e.g. ocr-to-kb ~/Desktop/book-pages ml-data"
    return 1
  fi
  claude -p "OCR the book screenshots in $image_dir into my knowledge base under the '$domain' domain. Run the full ocr-book-to-kb skill end-to-end: infer the title/author from the folder name or cover page (only ask me if it's genuinely ambiguous), OCR the pages (tesseract by default; if the book has tables/diagrams that matter, use --tables and mention the runtime tradeoff before starting), run bookrag build to structure it into principles/arguments/frameworks/failure-modes (never a raw text dump), verify indexing with a sample query, rebuild the ask-kb master index, and create the Obsidian vault reference note. The domain is already given ($domain) — do not ask me for it. Proceed straight through without pausing for confirmation, unless a step fails or an OCR page looks broken/near-empty, in which case tell me and keep going with the rest."
}
EOF
  echo "  added to $ZSHRC — run 'source ~/.zshrc' or open a new terminal to use it"
fi

echo
echo "✅ Done."
echo "   Script:    $REPO_DIR/ocr_book_to_kb.py"
echo "   Benchmark: $REPO_DIR/.venv-ocr/bin/python $REPO_DIR/benchmark.py <image_dir>"
echo "   Skill:     $SKILLS_DIR/SKILL.md"
