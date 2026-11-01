# 各スクリプトから source される共通設定
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
if [ -f "$ROOT/config.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/config.env"
else
  echo "!! $ROOT/config.env がありません。config.env.example をコピーして編集してください" >&2
  exit 1
fi
