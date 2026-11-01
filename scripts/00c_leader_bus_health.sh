#!/usr/bin/env bash
# リーダーのサーボバスを連続 ping して、どの関節で通信が切れるかを特定する。
#
# 使い方: scripts/00c_leader_bus_health.sh [秒数]
#
# 2026-09-29、録画中にリーダーが 3 回連続で落ちた。症状は常に
# 「リセット中(=リーダーを手で動かしているとき)」で、電圧は 4.9V(5V 電源なので正常)。
# デイジーチェーンのケーブル接触不良が疑われる。
#
# 使い方の手順:
#   1. このスクリプトを起動する
#   2. リーダーの関節を根元から順に、ゆっくり大きく動かす
#   3. どの関節を動かしたときに、どの ID の成功率が落ちるかを見る
#
# 落ちる ID より先(番号が大きい側)のケーブルが怪しい。
# 全部同時に落ちるなら、基板とモーター1 の間か USB 側。
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./config.env; set +a
SEC="${1:-60}"

python3 - "$LEADER_PORT" "$LEADER_ID" "$SEC" <<'PY'
import sys, time, logging
logging.disable(logging.WARNING)
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig

port, ident, sec = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = SOLeaderConfig(port=port, use_degrees=False); cfg.id = ident; cfg.calibration_dir = None
dev = SOLeader(cfg); dev.bus.connect()
names = list(dev.bus.motors)
print(f"対象: {names}")
print(f"{sec:.0f} 秒間 ping します。リーダーの関節を根元から順にゆっくり動かしてください。\n")

ok = {n: 0 for n in names}; ng = {n: 0 for n in names}
t0 = time.perf_counter(); last = t0
try:
    while time.perf_counter() - t0 < sec:
        for n in names:
            try:
                r = dev.bus.ping(n, num_retry=0, raise_on_error=False)
                (ok if r is not None else ng)[n] += 1
            except Exception:
                ng[n] += 1
        if time.perf_counter() - last >= 2.0:
            last = time.perf_counter()
            line = []
            for n in names:
                t = ok[n] + ng[n]
                rate = ok[n] / t * 100 if t else 0
                mark = "OK" if rate > 99 else ("..." if rate > 80 else "NG!")
                line.append(f"{n[:9]}:{rate:5.1f}%{mark}")
            print(f"[{time.perf_counter()-t0:5.1f}s] " + "  ".join(line), flush=True)
except KeyboardInterrupt:
    pass
finally:
    dev.bus.disconnect()

print("\n=== 累計 ===")
worst = None
for n in names:
    t = ok[n] + ng[n]; rate = ok[n] / t * 100 if t else 0
    print(f"  {n:<14} 成功 {ok[n]:5d} / 失敗 {ng[n]:5d}  = {rate:5.1f}%")
    if worst is None or rate < worst[1]: worst = (n, rate)
print()
if worst[1] > 99:
    print("判定: ○ 全モーターが安定して応答。動かしながらでも切れていない。")
    print("      再現しなかっただけの可能性もあるので、実際に落ちた動きを再現してみること。")
else:
    print(f"判定: ✗ {worst[0]} の成功率が {worst[1]:.1f}%。")
    print(f"      {worst[0]} とその手前のケーブル・コネクタを確認すること。")
PY
