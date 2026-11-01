#!/usr/bin/env bash
# モーター 1 個だけに ID・ボーレートを書き込む(非対話)。基板にそのモーターだけをつないだ状態で実行する。
# 使い方: scripts/02b_setup_one_motor.sh <follower|leader> <gripper|wrist_roll|wrist_flex|elbow_flex|shoulder_lift|shoulder_pan>
# 順番の目安: gripper(6) → wrist_roll(5) → wrist_flex(4) → elbow_flex(3) → shoulder_lift(2) → shoulder_pan(1)
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; MOTOR="${2:?motor name}"
case "$ARM" in
  follower) PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID" ;;
  leader)   PORT="$LEADER_PORT";   ID="$LEADER_ID" ;;
  *) echo "usage: $0 follower|leader <motor>" >&2; exit 1 ;;
esac
python - "$ARM" "$PORT" "$ID" "$MOTOR" <<'PY'
import sys, logging
logging.disable(logging.WARNING)
arm, port, rid, motor = sys.argv[1:5]
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid))
if motor not in dev.bus.motors:
    sys.exit(f"unknown motor '{motor}'. choose from: {list(dev.bus.motors)}")
dev.bus.setup_motor(motor)
print(f"OK: '{motor}' -> id {dev.bus.motors[motor].id} ({arm}, {port})")
PY
