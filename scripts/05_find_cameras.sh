#!/usr/bin/env bash
# 接続中のカメラ一覧(index、解像度、fps)。結果を config.env の CAMERAS に反映する。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/.venv/bin/activate"
lerobot-find-cameras opencv
