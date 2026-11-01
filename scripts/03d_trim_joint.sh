#!/usr/bin/env bash
# 1 関節の対応ずれを補正する。キャリブレーションの range_min/range_max を同じ量だけずらし(幅は不変)、
# その関節の 0 度(=中点)を動かす。フォロワー側で使うと「同じリーダー姿勢に対するフォロワーの角度」が delta 度ぶん変わる。
# 使い方: scripts/03d_trim_joint.sh <follower|leader> <motor名> <delta度>   例: scripts/03d_trim_joint.sh follower shoulder_pan -3.5
# 目安: テレオペ中にフォロワーが「リーダーより +5 度」にいるなら follower に +5 を与える(フォロワーの読みが 5 度小さくなり、追従先が 5 度戻る)
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; MOTOR="${2:?motor}"; DELTA="${3:?delta degrees}"
case "$ARM" in
  follower) PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID" ;;
  leader)   PORT="$LEADER_PORT";   ID="$LEADER_ID" ;;
  *) echo "usage: $0 follower|leader <motor> <delta_deg>" >&2; exit 1 ;;
esac
python - "$ARM" "$PORT" "$ID" "$MOTOR" "$DELTA" <<'PY'
import sys, logging
logging.disable(logging.WARNING)
arm, port, rid, motor, delta = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
if motor not in dev.calibration: sys.exit(f"unknown motor {motor}")
ticks = round(delta * 4095 / 360)
c = dev.calibration[motor]
old = (c.range_min, c.range_max)
c.range_min += ticks; c.range_max += ticks
if not (0 <= c.range_min and c.range_max <= 4095): sys.exit(f"範囲外になります: {c.range_min}..{c.range_max}")
bus.connect(handshake=True)
try:
    # EEPROM(Min/Max_Position_Limit, Homing_Offset)はトルク ON だと電源断で元に戻ることがあるため、必ずトルクを切ってから書く
    bus.disable_torque()
    bus.write_calibration(dev.calibration)
    dev._save_calibration()
    chk = bus.read_calibration()[motor]
    print(f"motor readback: homing={chk.homing_offset} min={chk.range_min} max={chk.range_max}")
finally:
    bus.disconnect(disable_torque=False)
print(f"[{arm}] {motor}: range {old} -> ({c.range_min}, {c.range_max})  ({delta:+.1f} deg = {ticks:+d} ticks)  保存: {dev.calibration_fpath}")
PY
