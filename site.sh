#!/usr/bin/env bash
# Course website commands: ./site.sh preview | build | check
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-preview}"
if [[ "$MODE" = check ]]; then exec python3 scripts/check_site.py; fi
case "$MODE" in preview|build) ;; *) echo 'Usage: ./site.sh [preview|build|check]' >&2; exit 2 ;; esac
QUARTO="$(command -v quarto || true)"
if [[ -z "$QUARTO" && -x "$ROOT/../.tools/bin/quarto" ]]; then QUARTO="$ROOT/../.tools/bin/quarto"; fi
if [[ -z "$QUARTO" ]]; then echo 'Install Quarto 1.10.18 from https://quarto.org/docs/get-started/ before building.' >&2; exit 127; fi
# Includes must exist before Quarto discovers render targets on a clean checkout.
python3 scripts/prepare_site.py
if [[ "$MODE" = preview ]]; then
    exec "$QUARTO" preview --no-browser
else
    "$QUARTO" render
    python3 scripts/check_site.py
fi
