#!/usr/bin/env bash
# Works from any directory; see --help for file, week, and whole-course builds.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/scripts/build_slides.py" "$@"
