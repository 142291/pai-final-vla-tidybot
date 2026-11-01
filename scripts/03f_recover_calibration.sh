#!/usr/bin/env bash
# モーターの EEPROM に残っている設定からキャリブレーションファイルを復元する。
# キャリブレーションファイル(~/.cache/huggingface/lerobot/calibration/...)を消してしまった場合の救済。
# アームを動かす必要はない。読み取って書き出すだけ。
#
# 使い方: scripts/03f_recover_calibration.sh [follower|leader|both]   省略時 both
#
# 注意: これはファイルを作り直すだけで、モーター側の値は変えない。
#       アーム自体を組み直した/中央合わせをやり直した場合は使えない(03b からやり直すこと)。
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "${1:-both}" in
  follower) ARMS=(follower) ;;
  leader)   ARMS=(leader) ;;
  both)     ARMS=(follower leader) ;;
  *) echo "usage: $0 [follower|leader|both]" >&2; exit 1 ;;
esac
for ARM in "${ARMS[@]}"; do
  if [ "$ARM" = follower ]; then PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID"; else PORT="$LEADER_PORT"; ID="$LEADER_ID"; fi
  python - "$ARM" "$PORT" "$ID" <<'PY' 2>&1 | grep -v "^objc\["
import sys, logging
logging.disable(logging.WARNING)
arm, port, rid = sys.argv[1:4]
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
bus.connect(handshake=True)
try:
    bus.disable_torque()
    calib = bus.read_calibration()          # EEPROM から読む
    bad = [m for m, c in calib.items() if c.range_min == 0 and c.range_max == 4095 and m != "wrist_roll"]
    print(f"[{arm}] モーターから読んだ設定:")
    for m, c in calib.items():
        print(f"  {m:14s} homing={c.homing_offset:6d} range={c.range_min:5d}..{c.range_max:<5d}")
    if bad:
        print(f"  !! 可動域が初期値のままの関節: {bad} → EEPROM も失われている。03b から取り直しが必要")
    dev.calibration = calib
    dev._save_calibration()
    print(f"  復元: {dev.calibration_fpath}")
    print(f"  is_calibrated: {dev.is_calibrated}")
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
done
