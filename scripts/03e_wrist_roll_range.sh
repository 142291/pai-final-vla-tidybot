#!/usr/bin/env bash
# 手首回転(wrist_roll)を「端から端」で両アーム共通の基準にする。
# 前提: 03c で両アームの wrist_roll を「同じ端」で 2048 に揃えてある。
# 使い方: 両アームの手首を反対側の端まで回した状態で  scripts/03e_wrist_roll_range.sh
# 動作: 反対側の端の位置(2048+R)を読み、可動域を 2048 中心に来るよう homing_offset を R/2 ずらし、
#       キャリブレーションファイルの wrist_roll range を [2048-|R|/2, 2048+|R|/2] にする(0〜4095 固定をやめる)。
set -euo pipefail
source "$(dirname "$0")/env.sh"
python - "$LEADER_PORT" "$LEADER_ID" "$FOLLOWER_PORT" "$FOLLOWER_ID" <<'PY'
import sys, time, logging
logging.disable(logging.WARNING)
lp, lid, fp, fid = sys.argv[1:5]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
devs = [("leader", SO101Leader(SO101LeaderConfig(port=lp, id=lid))), ("follower", SO101Follower(SO101FollowerConfig(port=fp, id=fid)))]
m = "wrist_roll"
for name, dev in devs:
    bus = dev.bus; bus.connect(handshake=True)
    try:
        bus.disable_torque(m)
        pos = int(bus.read("Present_Position", m, normalize=False))
        R = pos - 2048
        if abs(R) < 300: sys.exit(f"[{name}] 反対側の端まで回っていないようです(2048 からの移動 {R})")
        half = abs(R) // 2
        old_off = int(bus.read("Homing_Offset", m, normalize=False))
        # present = actual - offset。端Aが 2048、端Bが 2048+R。中央を 2048 にするには offset += R/2
        new_off = old_off + (R // 2)
        if abs(new_off) > 2047: sys.exit(f"[{name}] homing_offset が範囲外 {new_off}")
        bus.write("Homing_Offset", m, new_off, normalize=False)
        time.sleep(0.2)
        c = dev.calibration[m]
        c.homing_offset = int(bus.read("Homing_Offset", m, normalize=False))
        c.range_min = 2048 - half; c.range_max = 2048 + half
        bus.write_calibration(dev.calibration); dev._save_calibration()
        now = int(bus.read("Present_Position", m, normalize=False)); deg = float(bus.read("Present_Position", m, normalize=True))
        print(f"[{name}] 端B raw={pos} (R={R:+d}) -> homing {old_off}->{c.homing_offset}, range=({c.range_min},{c.range_max}), いまの読み raw={now} / {deg:+.1f} deg  保存: {dev.calibration_fpath}")
    finally:
        bus.disconnect(disable_torque=False)
print("両アームとも、端A/端B がそれぞれ -R/2 / +R/2 度に対応。中央(0 度)が同じ物理向きになる")
PY
