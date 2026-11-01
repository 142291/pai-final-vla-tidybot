#!/usr/bin/env bash
# 自然言語でアームを動かす(SmolVLA)。
#
# 使い方: scripts/10_language.sh "<指示(英語)>" [秒数]
# 例:    scripts/10_language.sh "Pick up the red cube and put it in the box"
#
# 環境変数:
#   POLICY  使うモデル(既定は SmolVLA の最終チェックポイント)
#   FPS     制御周波数(既定 30。学習データと揃えること)
#
# ■ カメラ権限の都合で「ターミナル.app」から実行すること。
#
# ■ 起動直後の約 1.2 秒はアームが動かない(MPS のカーネルコンパイル)。故障ではない。
#
# ■ 中身の設定(実測にもとづく、変更しないこと):
#   POLICY_DEVICE=mps  … CPU は forward 1 回に約 7 秒かかり成立しない。MPS は 742ms
#   INFER=rtc          … 推論を裏スレッドへ。無いと 1.67 秒ごとに 0.74 秒フリーズする
#   RENAME_MAP         … SmolVLA のベースは camera1/camera2 を期待するので変換が要る
#
# ■ 学習データのタスクは 1 種類だけなので、違う指示を出しても同じ動作をする。
#   指示を区別させるには複数タスクの録画(Phase 3)が必要。
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?指示を英語で。例: \"Pick up the red cube and put it in the box\"}"
DUR="${2:-60}"
POLICY="${POLICY:-outputs/train/smolvla_so101_cube_v2/checkpoints/last/pretrained_model}"

if [ ! -d "$POLICY" ]; then
  echo "モデルが無い: $POLICY" >&2
  echo "H200 からの回収がまだか、パスが違う。" >&2
  exit 1
fi

echo "指示  : $TASK"
echo "モデル: $POLICY"
echo "停止  : Ctrl+C (非常時はフォロワーの電源 OFF)"
echo

POLICY_DEVICE=mps \
INFER=rtc \
ROLLOUT_FPS="${FPS:-30}" \
RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  scripts/08_rollout.sh "$POLICY" "$TASK" "$DUR"
