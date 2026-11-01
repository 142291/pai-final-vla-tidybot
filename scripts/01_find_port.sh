#!/usr/bin/env bash
# モーターバス基板の USB ポートを特定する。指示に従って USB を抜き差しする。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/.venv/bin/activate"
lerobot-find-port
