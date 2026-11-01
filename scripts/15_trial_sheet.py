"""評価の試行ごとの録画(outputs/eval/rec/)から、経過を 1 枚に並べた画像を作る。

使い方:
    .venv/bin/python scripts/15_trial_sheet.py [録画フォルダ ...]   # 省略時は rec/ の全試行

上段 = 俯瞰カメラ、下段 = 手首カメラ。左から時間順に N 枚(既定 8)。最終列は最後の観測。
1 枚目の俯瞰画像には、色検出したキューブの位置(色名と座標)を描く。
出力: outputs/eval/sheets/<試行名>.jpg

2026-10-19: キューブの置き場所は P1〜P5 の指定どおりではなく手で自由に置いているため、
位置は画像から判断する必要がある。また「少しずれた所を掴み続ける」のずれ方を確かめるため。
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

audit = import_module("06d_audit_dataset")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REC = os.path.join(ROOT, "outputs", "eval", "rec")
OUT = os.path.join(ROOT, "outputs", "eval", "sheets")
N = int(os.environ.get("SHEET_N", "8"))
DRAW = {"red": (0, 0, 255), "yellow": (0, 255, 255), "purple": (255, 0, 200)}


def cam_names(d):
    """録画のカメラ名(方策ごとに違う: pi05 = base_0_rgb/left_wrist_0_rgb、smolvla = camera1/camera2、act = top/wrist)。"""
    for top, wrist in (("base_0_rgb", "left_wrist_0_rgb"), ("camera1", "camera2"), ("top", "wrist")):
        if glob.glob(os.path.join(d, f"*_{top}.jpg")):
            return top, wrist
    return "base_0_rgb", "left_wrist_0_rgb"


def cube_positions(top_bgr):
    return {c: audit.find_cube(top_bgr, c) for c in audit.HSV}


def sheet(d):
    lines = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    obs = [l for l in lines if l["kind"] == "obs"]
    if not obs:
        return None
    t0 = obs[0]["obs_t"]
    pick = sorted({round(i) for i in np.linspace(0, len(obs) - 1, N)})
    cols = []
    cam_top, cam_wrist = cam_names(d)
    for j, i in enumerate(pick):
        o = obs[i]
        top = cv2.imread(os.path.join(d, f"{o['step']:05d}_{cam_top}.jpg"))
        wr = cv2.imread(os.path.join(d, f"{o['step']:05d}_{cam_wrist}.jpg"))
        if top is None or wr is None:
            continue
        if j == 0:
            for c, p in cube_positions(top).items():
                if p is not None:
                    cv2.circle(top, (int(p[0]), int(p[1])), 14, DRAW[c], 2)
                    cv2.putText(top, f"{c} ({p[0]:.0f},{p[1]:.0f})", (int(p[0]) - 40, int(p[1]) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW[c], 2)
        col = np.vstack([cv2.resize(top, (320, 180)), cv2.resize(wr, (320, 180))])
        cv2.putText(col, f"t={o['obs_t'] - t0:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cols.append(col)
    img = np.hstack(cols)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, os.path.basename(d.rstrip("/")) + ".jpg")
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(REC, "*")))
    for d in dirs:
        print(sheet(d))
