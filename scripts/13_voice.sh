#!/usr/bin/env bash
# 音声でアームに指示する。
#
# 使い方: scripts/13_voice.sh [録音秒数]
#
# 流れ:
#   マイクで録音 → mlx-whisper で文字起こし(日本語) → 意図を判定 →
#   ポリシーが学習した英語の指示文に変換 → H200 経由で実行
#
# ■ カメラとマイクの権限の都合で「ターミナル.app」から実行すること。
# ■ 事前に H200 で policy_server が動いていること。
#
# ■ 正直な但し書き:
#   学習データのタスクは 1 種類だけなので、ポリシー自体は指示を区別できない。
#   ここでの「意図判定」はキーワードによる対応付けであって、言語接地ではない。
#   複数タスクを録画(Phase 3)して初めて、本物の言語条件付けになる。
#   その段階では、この対応表がタスク選択の入口としてそのまま使える。
set -euo pipefail
cd "$(dirname "$0")/.."
SEC="${1:-4}"
MIC="${MIC_DEVICE:-2}"          # ffmpeg -f avfoundation -list_devices true -i "" で確認
WAV="${TMPDIR:-/tmp}/voice_cmd.wav"

echo "▶ ${SEC} 秒間、話してください（例: 「赤いキューブを取って」）"
echo
ffmpeg -hide_banner -loglevel error -f avfoundation -i ":${MIC}" -t "$SEC" -ar 16000 -ac 1 -y "$WAV"
echo "✓ 録音完了"

echo "▶ 文字起こし中..."
HEARD=$(python3 - "$WAV" <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
import mlx_whisper
r = mlx_whisper.transcribe(sys.argv[1], path_or_hf_repo="mlx-community/whisper-small-mlx", language="ja")
print(r["text"].strip())
PY
)
echo "  聞き取り: 「$HEARD」"

# 意図の判定(いまはタスク 1 種なので単純な対応付け)
TASK=""
case "$HEARD" in
  *赤*|*レッド*|*red*)  TASK="Pick up the red cube and put it in the box" ;;
esac
if [ -z "$TASK" ]; then
  case "$HEARD" in
    *取*|*とっ*|*掴*|*つかん*|*持*|*もっ*|*入れ*|*運*)
      TASK="Pick up the red cube and put it in the box"
      echo "  ! 色は聞き取れませんでしたが、取る指示と判断しました" ;;
  esac
fi
if [ -z "$TASK" ]; then
  echo
  echo "✗ 指示として解釈できませんでした。"
  echo "  「赤いキューブを取って」のように言ってください。"
  exit 1
fi

echo "  → 指示: $TASK"
echo
exec scripts/12_language_remote.sh "$TASK"
