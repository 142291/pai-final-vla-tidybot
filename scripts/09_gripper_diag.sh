#!/usr/bin/env bash
# フォロワーのグリッパ診断。テレオペを止めてから実行する(同じポートを同時に開かない)。
#   scripts/09_gripper_diag.sh regs            設定レジスタと電圧・温度・負荷を表示
#   scripts/09_gripper_diag.sh where           8 秒間、手で開閉した位置(0〜100%)を表示 → どちらが閉か分かる
#   scripts/09_gripper_diag.sh hold <目標%> [秒] 物を挟んだ状態で目標%まで閉じてトルクを入れ、負荷/電流/電圧/温度を記録
set -euo pipefail
source "$(dirname "$0")/env.sh"
if pgrep -f lerobot-teleoperate >/dev/null; then echo "lerobot-teleoperate が動作中。先に止めてください" >&2; exit 1; fi
MODE="${1:?regs|where|hold}"; TARGET="${2:-0}"; SECS="${3:-10}"
python - "$FOLLOWER_PORT" "$FOLLOWER_ID" "$MODE" "$TARGET" "$SECS" <<'PY'
import sys, time, logging
logging.disable(logging.WARNING)
port, rid, mode, target, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
dev = SO101Follower(SO101FollowerConfig(port=port, id=rid)); bus = dev.bus
g = "gripper"
bus.connect(handshake=True)
try:
    def r(name, norm=False): return bus.read(name, g, normalize=norm)
    if mode == "regs":
        for k in ["Max_Torque_Limit","Torque_Limit","Protection_Current","Protective_Torque","Protection_Time",
                  "Overload_Torque","Over_Current_Protection_Time","P_Coefficient","I_Coefficient","D_Coefficient",
                  "Min_Position_Limit","Max_Position_Limit","Present_Position","Present_Load","Present_Current",
                  "Present_Voltage","Present_Temperature","Status","Torque_Enable"]:
            try: print(f"{k:28s} {int(r(k))}")
            except Exception as e: print(f"{k:28s} ERR {type(e).__name__}")
        print(f"{'Present_Position(%)':28s} {float(r('Present_Position', True)):.1f}")
        print("電圧は 0.1V 単位。7.4V 系なら 50〜84、12V 系なら 90〜126 が正常")
    elif mode == "where":
        bus.disable_torque(g)
        print("手でグリッパを開閉してください(8 秒)")
        t0 = time.time()
        while time.time() - t0 < secs if secs != 10 else time.time() - t0 < 8:
            print(f"  gripper {float(r('Present_Position', True)):6.1f}%  (raw {int(r('Present_Position'))})", flush=True)
            time.sleep(0.5)
    elif mode == "hold":
        print(f"目標 {target:.0f}% に閉じてトルク保持、{secs:.0f} 秒記録")
        bus.enable_torque(g)
        bus.write("Goal_Position", g, target, normalize=True)
        t0 = time.time()
        print(f"{'t':>5s} {'pos%':>6s} {'load':>6s} {'curr':>6s} {'V':>5s} {'temp':>5s} {'status':>6s}")
        while time.time() - t0 < secs:
            print(f"{time.time()-t0:5.1f} {float(r('Present_Position', True)):6.1f} {int(r('Present_Load')):6d} {int(r('Present_Current')):6d} {int(r('Present_Voltage'))/10:5.1f} {int(r('Present_Temperature')):5d} {int(r('Status')):6d}", flush=True)
            time.sleep(0.5)
        bus.disable_torque(g)
        print("トルク OFF")
    else:
        sys.exit("mode は regs|where|hold")
finally:
    try: bus.disconnect(disable_torque=False)
    except Exception: pass
PY
