"""ロボットが扱う物の一覧(名前・π0.5 への指示文・画面の色)と、「もう教わった物」の記録。

教わった物は outputs/skills.json に書く(無ければキューブ 3 色だけ)。
新しい物を教えて学習し直したら、次で加える:
    .venv/bin/python scripts/objects.py add eraser

2026-10-28: キューブ 3 色だけだった段取り・評価・通し動作を、消しゴムを教える場面のために一般化した。
キューブは色検出で箱の中かを判定できるが、消しゴムなど新しい物は検出器が無いので、Claude に画像で判定させる
(16_orchestrator.py の vlm_in_box)。
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SKILLS = os.path.join(ROOT, "outputs", "skills.json")

OBJECTS = {
    "red": {"ja": "赤", "task": "Pick up the red cube and put it in the box", "draw": (0, 0, 255), "cube": True},
    "yellow": {"ja": "黄", "task": "Pick up the yellow cube and put it in the box", "draw": (0, 220, 255), "cube": True},
    "purple": {"ja": "紫", "task": "Pick up the purple cube and put it in the box", "draw": (220, 0, 200), "cube": True},
    "eraser": {"ja": "消しゴム", "task": "Pick up the eraser and put it in the box", "draw": (255, 255, 255), "cube": False},
}
CUBES = [k for k, v in OBJECTS.items() if v["cube"]]
JA = {k: v["ja"] for k, v in OBJECTS.items()}


def task(obj: str) -> str:
    return OBJECTS[obj]["task"]


def taught() -> list:
    try:
        return [k for k in json.load(open(SKILLS))["taught"] if k in OBJECTS]
    except (OSError, ValueError, KeyError):
        return list(CUBES)


def add(obj: str):
    t = taught()
    if obj not in t:
        t.append(obj)
    os.makedirs(os.path.dirname(SKILLS), exist_ok=True)
    json.dump({"taught": t}, open(SKILLS, "w"), ensure_ascii=False, indent=1)
    return t


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "add":
        print("教わった物:", add(sys.argv[2]))
    else:
        print("教わった物:", taught())
