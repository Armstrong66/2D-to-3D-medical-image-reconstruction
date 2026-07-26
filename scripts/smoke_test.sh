#!/usr/bin/env bash
# Run all synthetic-data smoke tests, CPU-only. Meant to be run before every
# handoff to the execution environment (Kaggle / workstation) -- if this
# doesn't pass, don't bother spending real GPU time yet.
set -euo pipefail
cd "$(dirname "$0")/.."
pytest tests/ -x -q "$@"
