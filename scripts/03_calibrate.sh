#!/usr/bin/env bash
# キャリブレーション。全関節を可動域の中央にして Enter → 各関節を端から端まで動かす → Enter。
# 使い方: scripts/03_calibrate.sh follower | leader | both
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "${1:-}" in
  follower) lerobot-calibrate --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" ;;
  leader)   lerobot-calibrate --teleop.type=so101_leader  --teleop.port="$LEADER_PORT"  --teleop.id="$LEADER_ID" ;;
  both)     "$0" follower && echo && echo "---- 次はリーダー ----" && echo && "$0" leader ;;
  *) echo "usage: $0 follower|leader|both" >&2; exit 1 ;;
esac
