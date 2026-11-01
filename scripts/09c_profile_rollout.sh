#!/usr/bin/env bash
# 08_rollout.sh と同じ動作をしつつ、制御ループの区間別所要時間を計測する。
# 使い方: scripts/09c_profile_rollout.sh <policy path> "<タスク説明>" [秒数]
#
# 環境変数:
#   POLICY_DEVICE=cpu|mps   推論デバイス(既定 mps)
#   ROLLOUT_FPS=N           制御周波数(既定 30)
#   N_ACTION_STEPS=N        1回の予測を何ステップ実行するか(既定は学習時の設定)
#   NO_CAMERAS=1            カメラを外して実行し、カメラの影響を切り分ける
#
# カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
cd "$(dirname "$0")/.."
source "scripts/env.sh"
POLICY="${1:?policy path}"; TASK="${2:?task}"; DUR="${3:-20}"

ARGS=(
  --strategy.type=base
  --policy.path="$POLICY"
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false
  --task="$TASK"
  --policy.device="${POLICY_DEVICE:-mps}"
  --fps="${ROLLOUT_FPS:-30}"
  --duration="$DUR"
)
# NO_CAMERAS=1 のときはカメラ引数を渡さない(ポリシーは画像を欲しがるので失敗するが、
# get_observation の所要時間だけは比較できる)
if [ -z "${NO_CAMERAS:-}" ]; then
  ARGS+=(--robot.cameras="$CAMERAS")
fi
[ -n "${N_ACTION_STEPS:-}" ] && ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")

python scripts/09c_profile_rollout.py "${ARGS[@]}"
