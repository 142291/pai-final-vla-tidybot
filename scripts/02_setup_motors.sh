#!/usr/bin/env bash
# モーターに ID とボーレートを書き込む(EEPROM、1 回だけ)。
# 使い方: scripts/02_setup_motors.sh follower | leader | both
# 画面の指示どおり gripper(6) → wrist_roll(5) → ... → shoulder_pan(1) の順に 1 個ずつ基板に接続する。
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "${1:-}" in
  follower) lerobot-setup-motors --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" ;;
  leader)   lerobot-setup-motors --teleop.type=so101_leader  --teleop.port="$LEADER_PORT" ;;
  both)     "$0" follower && echo && echo "---- 次はリーダー ----" && echo && "$0" leader ;;
  *) echo "usage: $0 follower|leader|both" >&2; exit 1 ;;
esac
