"""通し動作(16_orchestrator)の記録 outputs/runs/*/events.jsonl から、自律化率などを集計する。

使い方:
    .venv/bin/python scripts/19_autonomy.py            # 全部をモデル・ラウンドごとに
    .venv/bin/python scripts/19_autonomy.py round1     # ROUND=round1 で動かした分だけ

指標(1 色 = 1 サブタスク):
  自律化率   … 人の手を借りずに箱へ入った割合(auto_success / サブタスク数)
  完了率     … 人の手助けも含めて箱へ入った割合
  1 回目成功 … やり直しなしで入った割合
  平均試行   … 自力で入ったサブタスクの、入るまでの試行回数
  介入       … 人が手伝った回数
ROUND は 17_orchestrate.sh を起動するときに環境変数で付ける(例: ROUND=round1)。

2026-10-25: ラウンドごとに「人の手を借りずに片付けられる割合」が上がることを示すために作った。
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main(only=None):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs", "runs", "*", "events.jsonl"))):
        ev = [json.loads(l) for l in open(f)]
        start = next((e for e in ev if e["event"] == "start"), None)
        if start is None:
            continue
        key = (start.get("round") or "-", f"{start.get('run', '?')}/{start.get('ckpt', '?')}", start.get("sync_k", "?"))
        if only and key[0] != only:
            continue
        steps = [e for e in ev if e["event"] == "step"]
        groups[key].append((os.path.basename(os.path.dirname(f)), start.get("instruction", ""), steps))
    if not groups:
        print("記録なし")
        return
    for (rnd, model, k), runs in sorted(groups.items()):
        steps = [s for _, _, ss in runs for s in ss]
        n = len(steps)
        auto = [s for s in steps if s["outcome"] == "auto_success"]
        done = [s for s in steps if s["outcome"] in ("auto_success", "human_success")]
        first = [s for s in auto if s["attempts"] == 1]
        human = [s for s in steps if s["outcome"].startswith("human")]
        p, lo, hi = wilson(len(auto), n)
        print(f"■ ラウンド {rnd}  モデル {model}  SYNC_K={k}  通し {len(runs)} 回・サブタスク {n} 個")
        print(f"  自律化率 {len(auto)}/{n} = {p:.0%} [95%CI {lo:.0%}–{hi:.0%}]   完了率 {len(done)}/{n}   "
              f"1 回目成功 {len(first)}/{n}   平均試行 {sum(s['attempts'] for s in auto) / max(1, len(auto)):.2f}   介入 {len(human)}")
        for name, instr, ss in runs:
            marks = " ".join(f"{s['color']}:{'○' if s['outcome'] == 'auto_success' else '✋' if s['outcome'] == 'human_success' else '×'}{s['attempts']}"
                             for s in ss)
            print(f"    {name}  「{instr}」  {marks}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
