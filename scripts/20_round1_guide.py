"""ラウンド 1 の追加録画の置き場所を、色ごとに番号つきの図と表で示す。

使い方:
    .venv/bin/python scripts/20_round1_guide.py            # outputs/record_r1/guide_<色>.jpg を作り、表を表示
    .venv/bin/python scripts/20_round1_guide.py red        # 赤の表だけ表示

狙い(2026-10-25 の評価と通し動作から):
  1. 学習データのすき間を埋める: アームの真下の中央、手前の端、かたまりの間(学習 150 本の置き場所は
     左奥・右側の縦一列・真ん中手前の 3 つに偏っていた。失敗はすき間に集まった)
  2. 箱にキューブが入った状態から始める回を入れる: 学習データには 1 本も無く、通しで片付けると
     1 個入った後に動かなくなった
  3. 真上から下ろして掴む: 近づくときにキューブを弾いて失敗した回があった
置き場所は色ごとに少しずらして、同じ点に重ならないようにする。
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

place_name = import_module("14_eval").place_name

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "outputs", "record_r1")
JA = {"red": "赤", "yellow": "黄", "purple": "紫"}
SHIFT = {"red": (0, 0), "yellow": (15, -10), "purple": (-15, 10)}
# (x, y, 説明)。俯瞰画像 640x360 の座標
BASE = [
    (330, 150, "アームの真下の中央"),
    (290, 200, "中央やや左"),
    (380, 215, "中央やや右"),
    (300, 320, "手前の端・中央"),
    (420, 320, "手前の端・やや右"),
    (240, 95, "左奥と中央の間"),
    (250, 165, "箱の右上"),
    (430, 90, "奥・右寄り"),
    (450, 180, "中央と右の間"),
    (600, 320, "右手前の端"),
    (470, 270, "手前・中央と右の間"),
    (360, 270, "手前・中央"),
    (590, 60, "右奥の端"),
    (130, 60, "左奥(比較用)"),
    (530, 160, "右中央(比較用)"),
]
# 何本目で、ほかの色を先に箱へ入れておくか(1 個 / 2 個)
BOX_ONE = {3, 6, 9, 12, 15}
BOX_TWO = {5, 10}


def plan(color):
    others = [c for c in JA if c != color]
    rows = []
    for i, (x, y, desc) in enumerate(BASE, 1):
        dx, dy = SHIFT[color]
        x, y = x + dx, y + dy
        if i in BOX_TWO:
            box = "、".join(JA[o] for o in others) + "を箱に入れておく"
        elif i in BOX_ONE:
            box = JA[others[i % 2]] + "を箱に入れておく"
        else:
            box = "箱は空"
        rows.append((i, x, y, f"{place_name(x, y)}({desc})", box))
    return rows


def draw(color, rows):
    ref = cv2.resize(cv2.imread(os.path.join(ROOT, "outputs", "cam_test", "ref_top_so101_cube_v3_s1.jpg")), (640, 360))
    for x in (213, 427):
        cv2.line(ref, (x, 0), (x, 360), (200, 200, 200), 1)
    for y in (120, 240):
        cv2.line(ref, (0, y), (640, y), (200, 200, 200), 1)
    col = {"red": (0, 0, 255), "yellow": (0, 220, 255), "purple": (220, 0, 200)}[color]
    for i, x, y, _, box in rows:
        cv2.circle(ref, (x, y), 12, col, 2)
        cv2.putText(ref, str(i), (x - 7, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        if box != "箱は空":
            cv2.circle(ref, (x, y), 16, (255, 255, 255), 1)   # 二重丸 = 箱に入れておく回
    cv2.putText(ref, f"{color}: double circle = put other cube(s) in the box first", (8, 352),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"guide_{color}.jpg")
    cv2.imwrite(path, ref)
    return path


if __name__ == "__main__":
    colors = sys.argv[1:] or list(JA)
    for c in colors:
        rows = plan(c)
        print(f"\n■ {JA[c]}(図: {draw(c, rows)})")
        for i, x, y, name, box in rows:
            print(f"  {i:2d}. {name:<22s} {box}")
