#!/usr/bin/env bash
# 基板につながっているモーターを読み取り専用でスキャンし、ボーレートと ID を表示する。
# 使い方: scripts/00b_scan_bus.sh [follower|leader|both]   (省略時 both)
# 用途: 電源が来ているか / ベンダーが ID を設定済みか / デイジーチェーンが全部見えるか の確認。
set -euo pipefail
source "$(dirname "$0")/env.sh"
PORTS=()
case "${1:-both}" in
  follower) PORTS=("$FOLLOWER_PORT") ;;
  leader)   PORTS=("$LEADER_PORT") ;;
  both)     PORTS=("$FOLLOWER_PORT" "$LEADER_PORT") ;;
  *) echo "usage: $0 [follower|leader|both]" >&2; exit 1 ;;
esac
python - "${PORTS[@]}" <<'PY'
import sys, logging
logging.disable(logging.WARNING)
from lerobot.motors.feetech import FeetechMotorsBus
for port in sys.argv[1:]:
    print(f"\n=== {port} ===")
    try:
        found = FeetechMotorsBus.scan_port(port)
    except Exception as e:
        print("ERROR:", type(e).__name__, e); continue
    if not found:
        print("応答なし: モーター未接続 / 電源なし / 3ピンケーブル不良")
    for baud, ids in found.items():
        print(f"baud {baud}: id {ids}   (lerobot 標準は baud 1000000、ID 1〜6)")
PY
