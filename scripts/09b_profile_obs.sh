#!/usr/bin/env bash
# 実行ループの内訳を測る。observation 取得のどこに時間がかかっているかを特定する。
# カメラ権限の都合で「ターミナル.app」から実行すること。
# 使い方: scripts/09b_profile_obs.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
python - "$FOLLOWER_PORT" "$FOLLOWER_ID" "$CAMERAS" <<'PY' 2>&1 | grep -viE "^objc\[|AVFFrame|AVFAudio"
import sys, time, logging, statistics
logging.disable(logging.WARNING)
port, rid, cams_str = sys.argv[1:4]
import draccus
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
cfg = draccus.parse(TeleoperateConfig, args=[
  "--robot.type=so101_follower", f"--robot.port={port}", f"--robot.id={rid}",
  f"--robot.cameras={cams_str}", "--robot.use_degrees=false",
  "--teleop.type=so101_leader", "--teleop.port=/dev/null", "--teleop.id=x"])
robot = SO101Follower(cfg.robot)
robot.connect()
def timeit(fn, n=40):
    fn(); ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), max(ts)
try:
    print(f"{'処理':38s} {'中央値':>9s} {'最大':>9s}")
    m,x = timeit(lambda: robot.bus.sync_read("Present_Position", num_retry=2))
    print(f"{'モーター 6 個の読み取り':38s} {m:7.1f}ms {x:7.1f}ms")
    for name, cam in robot.cameras.items():
        m,x = timeit(lambda c=cam: c.read_latest())
        print(f"{'カメラ read_latest: '+name:38s} {m:7.1f}ms {x:7.1f}ms")
    m,x = timeit(lambda: robot.get_observation())
    print(f"{'get_observation 全体':38s} {m:7.1f}ms {x:7.1f}ms")
    print(f"\n→ get_observation が {m:.0f}ms なら上限 {1000/m:.0f}Hz。30Hz には 33ms 以内が必要")
finally:
    robot.disconnect()
PY
