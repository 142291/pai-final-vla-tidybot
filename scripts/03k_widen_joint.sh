#!/usr/bin/env bash
# 1 関節だけ可動域を測り直す。その関節を端から端まで動かしている間の実測値で置き換える。
# 頭打ちになって動かない関節がある場合に使う。
# 使い方: scripts/03k_widen_joint.sh <follower|leader> <関節名> [秒数]
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; J="${2:?関節名}"; SECS="${3:-40}"
if [ "$ARM" = follower ]; then PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID"; else PORT="$LEADER_PORT"; ID="$LEADER_ID"; fi
python - "$ARM" "$PORT" "$ID" "$J" "$SECS" <<'PY' 2>&1 | grep -v "^objc\["
import sys, time, logging
logging.disable(logging.WARNING)
arm, port, rid, J, secs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
bus.connect(handshake=True)
try:
    bus.disable_torque()
    c = dev.calibration[J]
    old = (c.range_min, c.range_max)
    bus.write("Min_Position_Limit", J, 0, normalize=False)
    bus.write("Max_Position_Limit", J, 4095, normalize=False)
    time.sleep(0.2)
    r = lambda: int(bus.read("Present_Position", J, normalize=False))
    lo = hi = r()
    print(f"[{arm}] {J} を {int(secs)} 秒間、**止まるところまで**両方向に動かしてください(現在 {lo}、旧 {old[0]}..{old[1]})")
    t0 = time.time()
    while time.time() - t0 < secs:
        v = r(); lo = min(lo, v); hi = max(hi, v); time.sleep(0.04)
    if hi - lo < 200: sys.exit(f"ほとんど動いていません({lo}..{hi})。保存せず終了")
    c.range_min, c.range_max = lo, hi
    bus.disable_torque()
    bus.write_calibration(dev.calibration); dev._save_calibration()
    print(f"  旧 {old[0]}..{old[1]} (幅 {old[1]-old[0]})")
    print(f"  新 {lo}..{hi} (幅 {hi-lo}, {hi-lo-(old[1]-old[0]):+d})")
    print(f"  保存: {dev.calibration_fpath}")
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
