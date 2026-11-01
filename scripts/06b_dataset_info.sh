#!/usr/bin/env bash
# ローカルのデータセットの中身を確認する(エピソード数、タスク、特徴量)。
# 使い方: scripts/06b_dataset_info.sh <データセット名>
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="${1:?データセット名}"
python - "$ROOT/data/$NAME" "local/$NAME" <<'PY' 2>&1 | grep -v "^objc\["
import sys, logging, pathlib
logging.disable(logging.WARNING)
root, repo_id = pathlib.Path(sys.argv[1]), sys.argv[2]
if not root.exists(): sys.exit(f"ありません: {root}")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id, root=root)
m = ds.meta
print(f"データセット: {repo_id}")
print(f"  保存先      : {root}")
print(f"  エピソード数: {m.total_episodes}")
print(f"  フレーム数  : {m.total_frames}")
print(f"  fps         : {m.fps}")
print(f"  タスク数    : {m.total_tasks}")
for i, t in (m.tasks.items() if hasattr(m.tasks,'items') else enumerate(m.tasks)):
    print(f"    [{i}] {t}")
print("  特徴量:")
for k, v in m.features.items():
    print(f"    {k:32s} {v.get('dtype')} {v.get('shape')}")
PY
