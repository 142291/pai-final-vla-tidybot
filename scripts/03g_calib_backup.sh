#!/usr/bin/env bash
# キャリブレーションの検証・バックアップ・復元。
# ~/.cache/huggingface/lerobot/ は消えやすいので、プロジェクト内(calib/)に控えを置く。
#
#   scripts/03g_calib_backup.sh verify    モーターの EEPROM とファイルが一致しているか確認
#   scripts/03g_calib_backup.sh backup    キャッシュのファイルを calib/ にコピー(git 管理)
#   scripts/03g_calib_backup.sh restore   calib/ からキャッシュへ戻し、モーターにも書き込む
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE="${1:-verify}"
CACHE="$HOME/.cache/huggingface/lerobot/calibration"
BK="$ROOT/calib"
F_REL="robots/so_follower/${FOLLOWER_ID}.json"
L_REL="teleoperators/so_leader/${LEADER_ID}.json"

case "$MODE" in
  backup)
    for rel in "$F_REL" "$L_REL"; do
      [ -f "$CACHE/$rel" ] || { echo "!! ない: $CACHE/$rel" >&2; continue; }
      mkdir -p "$BK/$(dirname "$rel")"; cp "$CACHE/$rel" "$BK/$rel"; echo "バックアップ: calib/$rel"
    done
    ;;
  restore)
    for rel in "$F_REL" "$L_REL"; do
      [ -f "$BK/$rel" ] || { echo "!! 控えがない: calib/$rel" >&2; continue; }
      mkdir -p "$CACHE/$(dirname "$rel")"; cp "$BK/$rel" "$CACHE/$rel"; echo "キャッシュへ復元: $rel"
    done
    python - "$FOLLOWER_PORT" "$FOLLOWER_ID" "$LEADER_PORT" "$LEADER_ID" <<'PY' 2>&1 | grep -v "^objc\["
import sys, logging
logging.disable(logging.WARNING)
fp, fid, lp, lid = sys.argv[1:5]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
for name, dev in [("follower", SO101Follower(SO101FollowerConfig(port=fp, id=fid))),
                  ("leader",   SO101Leader(SO101LeaderConfig(port=lp, id=lid)))]:
    if not dev.calibration: print(f"[{name}] ファイルが読めない"); continue
    dev.bus.connect(handshake=True)
    try:
        dev.bus.disable_torque()          # EEPROM 定着に必須
        dev.bus.write_calibration(dev.calibration)
        print(f"[{name}] モーターへ書き込み完了")
    finally:
        dev.bus.disconnect(disable_torque=False)
PY
    ;;
  verify)
    python - "$FOLLOWER_PORT" "$FOLLOWER_ID" "$LEADER_PORT" "$LEADER_ID" <<'PY' 2>&1 | grep -v "^objc\["
import sys, logging
logging.disable(logging.WARNING)
fp, fid, lp, lid = sys.argv[1:5]
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
allok = True
for name, dev in [("follower", SO101Follower(SO101FollowerConfig(port=fp, id=fid))),
                  ("leader",   SO101Leader(SO101LeaderConfig(port=lp, id=lid)))]:
    if not dev.calibration:
        print(f"[{name}] !! キャリブレーションファイルがない ({dev.calibration_fpath})"); allok=False; continue
    dev.bus.connect(handshake=True)
    try:
        motor = dev.bus.read_calibration()
        diffs = []
        for m, c in dev.calibration.items():
            mm = motor[m]
            for field in ("homing_offset", "range_min", "range_max"):
                a, b = getattr(c, field), getattr(mm, field)
                if a != b: diffs.append(f"{m}.{field}: ファイル={a} モーター={b}")
        if diffs:
            allok = False
            print(f"[{name}] !! 不一致 {len(diffs)} 件")
            for d in diffs: print("   ", d)
            print("    → scripts/03g_calib_backup.sh restore で直す")
        else:
            print(f"[{name}] OK — ファイルとモーターが一致")
    finally:
        dev.bus.disconnect(disable_torque=False)
print("\n" + ("すべて一致。テレオペ/録画を始めてよい。" if allok else "不一致あり。直してから進むこと。"))
PY
    ;;
  *) echo "usage: $0 verify|backup|restore" >&2; exit 1 ;;
esac
