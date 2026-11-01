#!/usr/bin/env bash
# 手首回転(wrist_roll)の生値と角度を連続表示し、エンコーダの切れ目(値が飛ぶ点)を探す。
# 使い方: scripts/03h_watch_wrist_roll.sh [follower|leader] [秒数]
# 実行したら、その関節をゆっくり端から端まで回す。値が急に 4000 近く飛んだら、そこが切れ目。
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:-leader}"; SECS="${2:-40}"
if [ "$ARM" = follower ]; then PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID"; else PORT="$LEADER_PORT"; ID="$LEADER_ID"; fi
python - "$ARM" "$PORT" "$ID" "$SECS" <<'PY' 2>&1 | grep -v "^objc\["
import sys, time, logging
logging.disable(logging.WARNING)
arm, port, rid, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
m = "wrist_roll"
bus.connect(handshake=True)
try:
    bus.disable_torque(m)
    h = int(bus.read("Homing_Offset", m, normalize=False))
    print(f"[{arm}] wrist_roll homing_offset={h}  可動域={dev.calibration[m].range_min}..{dev.calibration[m].range_max}")
    print(f"{int(secs)} 秒間、この関節をゆっくり端から端まで回してください\n")
    prev = None; jumps = []; lo = hi = None
    t0 = time.time()
    while time.time() - t0 < secs:
        raw = int(bus.read(m, "Present_Position", normalize=False) if False else bus.read("Present_Position", m, normalize=False))
        deg = float(bus.read("Present_Position", m, normalize=True))
        lo = raw if lo is None else min(lo, raw); hi = raw if hi is None else max(hi, raw)
        mark = ""
        if prev is not None and abs(raw - prev) > 500:
            jumps.append((prev, raw)); mark = f"   <<< 飛んだ！ {prev} -> {raw}"
        print(f"  raw={raw:5d}  {deg:+7.1f}deg{mark}", flush=True)
        prev = raw
        time.sleep(0.25)
    print(f"\n観測した生値の範囲: {lo}..{hi}")
    if jumps:
        print(f"切れ目を {len(jumps)} 回またぎました:")
        for a, b in jumps: print(f"  {a} -> {b}")
        print("→ この関節の常用範囲が切れ目に近すぎる。対策が必要")
    else:
        print("切れ目はまたぎませんでした。常用範囲は安全側にあります")
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
