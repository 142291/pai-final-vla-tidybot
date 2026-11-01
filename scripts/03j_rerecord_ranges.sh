#!/usr/bin/env bash
# 可動域(range_min/range_max)を取り直す。homing_offset と wrist_roll の設定は変更しない。
# 頭打ち方式(use_degrees=false)では、登録した可動域が狭いとフォロワーが端で止まってしまうため、
# 各関節を「機械的に止まるところ」まで確実に押し当てて記録し直す。
#
# 使い方: scripts/03j_rerecord_ranges.sh <follower|leader> [秒数]
# 実行したら、wrist_roll 以外の 5 関節を端から端まで、止まるところまでしっかり動かす。
# 動きが止まって 8 秒で自動終了して保存。
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; MAXSEC="${2:-120}"
if [ "$ARM" = follower ]; then PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID"; else PORT="$LEADER_PORT"; ID="$LEADER_ID"; fi
python - "$ARM" "$PORT" "$ID" "$MAXSEC" <<'PY' 2>&1 | grep -v "^objc\["
import sys, time, logging
logging.disable(logging.WARNING)
arm, port, rid, maxsec = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
SKIP = "wrist_roll"
bus.connect(handshake=True)
try:
    bus.disable_torque()
    names = [m for m in bus.motors if m != SKIP]
    old = {m: (dev.calibration[m].range_min, dev.calibration[m].range_max) for m in names}
    # 可動域を一時的に最大にして、端まで読めるようにする
    for m in names:
        bus.write("Min_Position_Limit", m, 0, normalize=False)
        bus.write("Max_Position_Limit", m, 4095, normalize=False)
    time.sleep(0.2)
    read = lambda: bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
    start = read(); mins = dict(start); maxes = dict(start); last = dict(start)
    print(f"[{arm}] 記録開始。{SKIP} 以外の 5 関節を、**止まるところまで**端から端まで動かしてください")
    print(f"       最大 {int(maxsec)} 秒、動きを止めて 8 秒で自動終了\n")
    t0 = time.time(); moved = False; still = None
    while time.time() - t0 < maxsec:
        pos = read()
        if any(abs(int(pos[m]) - int(last[m])) > 12 for m in names):
            moved = True; still = None
        elif moved:
            still = still or time.time()
            if time.time() - still >= 8: break
        for m in names:
            mins[m] = min(int(mins[m]), int(pos[m])); maxes[m] = max(int(maxes[m]), int(pos[m]))
        last = pos
        time.sleep(0.03)
    if not moved: sys.exit("動きが検出されませんでした。保存せず終了")
    print(f"{'joint':14s} {'旧 min-max':>14s} {'新 min-max':>14s} {'幅の変化':>10s}")
    for m in names:
        lo, hi = int(mins[m]), int(maxes[m])
        dev.calibration[m].range_min = lo; dev.calibration[m].range_max = hi
        o = old[m]
        print(f"{m:14s} {o[0]:6d}-{o[1]:<7d} {lo:6d}-{hi:<7d} {(hi-lo)-(o[1]-o[0]):+10d}")
    bus.disable_torque()
    bus.write_calibration(dev.calibration); dev._save_calibration()
    print(f"\n保存: {dev.calibration_fpath}")
    print(f"{SKIP} は変更していません({dev.calibration[SKIP].range_min}..{dev.calibration[SKIP].range_max})")
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
