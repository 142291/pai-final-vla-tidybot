#!/usr/bin/env bash
# チャット越し(非対話)で進めるキャリブレーション。公式 lerobot-calibrate と同じ手順を 2 段階に分割したもの。
#   scripts/03b_calibrate_chat.sh <follower|leader> mid            全関節を可動域の中央にした状態で実行(ホーミング設定)
#   scripts/03b_calibrate_chat.sh <follower|leader> range [最大秒]  実行後に各関節を端から端まで動かす。動きが止まって 8 秒で自動終了し保存
#   scripts/03b_calibrate_chat.sh <follower|leader> show           保存済みキャリブレーションを表示
#   scripts/03b_calibrate_chat.sh <follower|leader> check          現在位置を表示(中央姿勢で 2048 付近か、電源再投入後の確認用)
# wrist_roll は 1 回転モーター扱いで 0〜4095 固定(公式と同じ)。
set -euo pipefail
source "$(dirname "$0")/env.sh"
ARM="${1:?follower|leader}"; PHASE="${2:?mid|range|show|check}"; MAXSEC="${3:-180}"
case "$ARM" in
  follower) PORT="$FOLLOWER_PORT"; ID="$FOLLOWER_ID" ;;
  leader)   PORT="$LEADER_PORT";   ID="$LEADER_ID" ;;
  *) echo "usage: $0 follower|leader mid|range|show" >&2; exit 1 ;;
esac
python - "$ARM" "$PORT" "$ID" "$PHASE" "$MAXSEC" "$ROOT/outputs/calib_tmp" <<'PY'
import sys, json, time, logging, pathlib
logging.disable(logging.WARNING)
arm, port, rid, phase, maxsec, tmpdir = sys.argv[1:7]
maxsec = float(maxsec); tmpdir = pathlib.Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
from lerobot.motors import MotorCalibration
try:
    from lerobot.motors.feetech import OperatingMode
except ImportError:
    from lerobot.motors.feetech.tables import OperatingMode
if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower as Cls, SO101FollowerConfig as Cfg
else:
    from lerobot.teleoperators.so_leader import SO101Leader as Cls, SO101LeaderConfig as Cfg
dev = Cls(Cfg(port=port, id=rid)); bus = dev.bus
homing_file = tmpdir / f"{arm}_homing.json"
FULL_TURN = "wrist_roll"

if phase == "show":
    fp = dev.calibration_fpath
    print("file:", fp, "(exists)" if fp.exists() else "(missing)")
    if fp.exists(): print(fp.read_text())
    sys.exit(0)

bus.connect(handshake=False)
try:
    if phase == "mid":
        # Feetech サーボ内蔵の中央合わせ: Torque_Enable(addr 40) に 128 を書くと現在位置が 2048 になり、
        # ファームウェアが Homing_Offset を自動設定する(lerobot の set_half_turn_homings は位置が負のとき失敗するため使わない)。
        bus.disable_torque()
        for m in bus.motors:
            bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
            bus.write("Min_Position_Limit", m, 0, normalize=False)
            bus.write("Max_Position_Limit", m, 4095, normalize=False)
        ADDR_TORQUE = 40
        for m, mm in bus.motors.items():
            comm, err = bus.packet_handler.write1ByteTxRx(bus.port_handler, mm.id, ADDR_TORQUE, 128)
            if comm != 0: print(f"  !! {m}: 128 書き込み失敗 {bus.packet_handler.getTxRxResult(comm)}")
        time.sleep(0.3)
        bus.disable_torque()
        pos = bus.sync_read("Present_Position", normalize=False, num_retry=5)
        homing = {m: int(bus.read("Homing_Offset", m, normalize=False)) for m in bus.motors}
        homing_file.write_text(json.dumps(homing, indent=2))
        bad = [m for m in bus.motors if abs(int(pos[m]) - 2048) > 30]
        print(f"[{arm}] 中央合わせ(Torque_Enable=128)実行")
        for m in bus.motors: print(f"  {m:14s} pos={int(pos[m]):5d}  homing_offset={homing[m]}")
        if bad:
            print("!! 2048 になっていない関節:", bad, "→ その姿勢のまま電源を入れ直してから再度 'mid'")
        else:
            print("全関節 2048 OK。保存:", homing_file)
            print("次: 'range' を実行し、wrist_roll 以外の全関節を端から端までゆっくり動かす")
    elif phase == "check":
        pos = bus.sync_read("Present_Position", normalize=False, num_retry=5)
        print(f"[{arm}] 現在位置(中央姿勢なら全部 2048 付近のはず)")
        for m in bus.motors: print(f"  {m:14s} pos={int(pos[m]):5d}  ({int(pos[m])-2048:+d})")
    elif phase == "range":
        if not homing_file.exists():
            sys.exit("先に 'mid' を実行してください(ホーミング未設定)")
        homing = json.loads(homing_file.read_text())
        bus.disable_torque()
        names = [m for m in bus.motors if m != FULL_TURN]
        def read(): return bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
        start = read(); mins = dict(start); maxes = dict(start); last = dict(start)
        t0 = time.time(); moved_once = False; still_since = None
        MOVE_TH, STILL_SEC = 15, 8.0
        print(f"[{arm}] 記録開始。各関節を端から端まで動かしてください(最大 {int(maxsec)} 秒、止めて {int(STILL_SEC)} 秒で自動終了)")
        while time.time() - t0 < maxsec:
            pos = read()
            moved = any(abs(pos[m] - last[m]) > MOVE_TH for m in names)
            for m in names:
                mins[m] = min(mins[m], pos[m]); maxes[m] = max(maxes[m], pos[m])
            last = pos
            if moved:
                moved_once = True; still_since = None
            elif moved_once:
                still_since = still_since or time.time()
                if time.time() - still_since >= STILL_SEC: break
            time.sleep(0.03)
        else:
            print("注意: 最大秒数に達したので打ち切りました")
        if not moved_once: sys.exit("動きが検出されませんでした。保存せず終了")
        print(f"{'NAME':<15} | {'MIN':>6} | {'MAX':>6} | {'RANGE':>6}")
        warn = []
        for m in names:
            r = maxes[m] - mins[m]
            print(f"{m:<15} | {mins[m]:>6} | {maxes[m]:>6} | {r:>6}")
            if r < 400: warn.append(m)
            if mins[m] < 150 or maxes[m] > 3945: print(f"!! {m}: 可動域がエンコーダの端(0/4095)に近い。中央合わせをやり直すか、ホーン取り付け角度の見直しが必要")
        mins[FULL_TURN] = 0; maxes[FULL_TURN] = 4095
        calib = {m: MotorCalibration(id=mm.id, drive_mode=0, homing_offset=int(homing[m]),
                                     range_min=int(mins[m]), range_max=int(maxes[m]))
                 for m, mm in bus.motors.items()}
        dev.calibration = calib
        bus.write_calibration(calib)
        dev._save_calibration()
        print(f"保存: {dev.calibration_fpath}")
        if warn: print("注意: 可動域が小さい関節があります(端まで動かせていない可能性):", warn, "→ 必要なら 'range' をやり直し")
    else:
        sys.exit("phase は mid|range|show|check")
finally:
    try: bus.disconnect()
    except Exception: pass
PY
