#!/usr/bin/env bash
set -euo pipefail

slow_workers="${PYTEST_SLOW_WORKERS:-4}"

uv run pytest -q -m 'not slow and not wolfram'
uv run pytest -q -n "${slow_workers}" -m slow
