#!/usr/bin/env bash
# リーダーでフォロワーを動かす。--cam を付けるとカメラ映像と関節値を rerun で表示。
# 安全: TELEOP_MAX_REL = 1 ステップあたりの最大移動量(±100 の正規化単位)。
#   既定は「無制限」。録画(06_record.sh)も無制限なので条件を揃えてある。
#   この値が小さいとサーボへの指令が現在位置の近くに制限され、フォロワーの速度自体が落ちる。
#   2026-09-27: 15 にしていたため肘が平均 47 度遅れ「上がらない」ように見えた。
#   慎重に動かしたいときだけ TELEOP_MAX_REL=20 などを指定する。
#   TELEOP_MAX_REL=40 scripts/04_teleop.sh
# 非常停止はフォロワーの電源を切る。Ctrl+C でも止まる(トルクは切れる)。
set -euo pipefail
source "$(dirname "$0")/env.sh"
MAXREL="${TELEOP_MAX_REL-}"
PGAIN="${TELEOP_P-16}"   # フォロワーの位置 P ゲイン。lerobot 既定 16(柔らかめ、重力で垂れやすい)、Feetech 既定 32
# use_degrees=false: 関節値を可動域で頭打ちにする方式。手首回転がエンコーダの切れ目を
# またいでも値が飛ばず、フォロワーが 1 回転する暴走が起きない(2026-09-26 に対策)
ARGS=(
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false
  --teleop.type=so101_leader  --teleop.port="$LEADER_PORT"  --teleop.id="$LEADER_ID"  --teleop.use_degrees=false
)
if [ -n "$MAXREL" ]; then ARGS+=( --robot.max_relative_target="$MAXREL" ); fi
ARGS+=( --robot.position_p_coefficient="$PGAIN" )
if [ "${1:-}" = "--cam" ]; then
  ARGS+=( --robot.cameras="$CAMERAS" --display_data=true )
fi
echo "起動: max_relative_target=${MAXREL:-なし} P=$PGAIN  (停止は Ctrl+C、非常時はフォロワー電源 OFF)"
lerobot-teleoperate "${ARGS[@]}"
