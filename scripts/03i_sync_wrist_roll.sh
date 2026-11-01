#!/usr/bin/env bash
# 両アームの手首回転(wrist_roll)を「同時に」0 点登録する。
# 片方ずつ登録すると、その間にもう片方が動いてずれるため、必ずこちらを使う。
#
# 使い方: 両アームの手首を同じ向き(作業で使う向き)に揃えてから
#   scripts/03i_sync_wrist_roll.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
python - "$FOLLOWER_PORT" "$FOLLOWER_ID" "$LEADER_PORT" "$LEADER_ID" <<'PY' 2>&1 | grep -v "^objc\["
import sys, time, logging
logging.disable(logging.WARNING)
fp, fid, lp, lid = sys.argv[1:5]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
M = "wrist_roll"; ADDR_TORQUE = 40
devs = [("follower", SO101Follower(SO101FollowerConfig(port=fp, id=fid))),
        ("leader",   SO101Leader(SO101LeaderConfig(port=lp, id=lid)))]
for _, d in devs: d.bus.connect(handshake=True)
try:
    # 1) 両方まとめてトルクを切る
    for _, d in devs: d.bus.disable_torque(M)
    # 2) 両方に 128(現在位置を 2048 にする)を連続で送る
    for nm, d in devs:
        comm, _ = d.bus.packet_handler.write1ByteTxRx(d.bus.port_handler, d.bus.motors[M].id, ADDR_TORQUE, 128)
        if comm != 0: sys.exit(f"[{nm}] 128 書き込み失敗")
    time.sleep(0.3)
    # 3) 両方の homing を読んでファイルへ保存
    for nm, d in devs:
        d.bus.disable_torque(M)
        off = int(d.bus.read("Homing_Offset", M, normalize=False))
        raw = int(d.bus.read("Present_Position", M, normalize=False))
        d.calibration[M].homing_offset = off
        d.bus.write_calibration(d.calibration); d._save_calibration()
        print(f"[{nm}] raw={raw}  homing={off}  切れ目まで {min(raw,4095-raw)*360/4095:.0f} 度")
    # 4) 一致確認
    fd = devs[0][1].bus.read("Present_Position", M, normalize=True)
    ld = devs[1][1].bus.read("Present_Position", M, normalize=True)
    print(f"\n手首回転: leader={float(ld):+.1f}度  follower={float(fd):+.1f}度  差={float(fd)-float(ld):+.1f}度")
    print("差が 3 度以内なら成功" if abs(float(fd)-float(ld)) < 3 else "!! ずれている。もう一度揃えて再実行")
finally:
    for _, d in devs:
        try: d.bus.disconnect(disable_torque=False)
        except Exception: pass
PY
