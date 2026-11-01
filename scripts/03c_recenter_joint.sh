#!/usr/bin/env bash
# 1 関節だけ中央(2048)を取り直し、キャリブレーションファイルの homing_offset も更新する。
# 使い方: scripts/03c_recenter_joint.sh <follower|leader> <motor名>
# 例: 両アームの手首をまっすぐにした状態で
#     scripts/03c_recenter_joint.sh leader wrist_roll && scripts/03c_recenter_joint.sh follower wrist_roll
# 可動域(range_min/max)は変えないので、wrist_roll(0〜4095 固定)以外で使うときは range もやり直すこと。
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; MOTOR="${2:?motor name}"
case "$ARM" in
  follower) PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID" ;;
  leader)   PORT="$LEADER_PORT";   ID="$LEADER_ID" ;;
  *) echo "usage: $0 follower|leader <motor>" >&2; exit 1 ;;
esac
python - "$ARM" "$PORT" "$ID" "$MOTOR" <<'PY'
import sys, time, logging
logging.disable(logging.WARNING)
arm, port, rid, motor = sys.argv[1:5]
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
if motor not in bus.motors: sys.exit(f"unknown motor {motor}: {list(bus.motors)}")
if not dev.calibration: sys.exit("キャリブレーションファイルがありません。先に 03b で mid/range を実行")
bus.connect(handshake=True)
try:
    bus.disable_torque(motor)
    before = int(bus.read("Present_Position", motor, normalize=False))
    comm, err = bus.packet_handler.write1ByteTxRx(bus.port_handler, bus.motors[motor].id, 40, 128)
    if comm != 0: sys.exit(f"128 書き込み失敗: {bus.packet_handler.getTxRxResult(comm)}")
    time.sleep(0.3)
    bus.disable_torque(motor)
    after = int(bus.read("Present_Position", motor, normalize=False))
    off = int(bus.read("Homing_Offset", motor, normalize=False))
    dev.calibration[motor].homing_offset = off
    bus.write_calibration(dev.calibration)
    dev._save_calibration()
    print(f"[{arm}] {motor}: pos {before} -> {after}, homing_offset={off}  保存: {dev.calibration_fpath}")
    print("is_calibrated:", dev.is_calibrated)
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
