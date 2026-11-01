"""評価の結果(results.csv)と試行ごとの録画を突き合わせ、置き場所・箱の位置・照明ごとに成否を集計する。

使い方:
    .venv/bin/python scripts/18_eval_analysis.py <条件名>        # 例: pi05_v3full+round0
出力:
    標準出力に 1 試行 1 行の表(対象の位置、学習時からの箱のずれ、明るさ、手首のコントラスト、結果、時間)
    outputs/eval/analysis_<条件名>.jpg … 学習時の俯瞰画像に、対象の初期位置を ○成功 ×失敗 で描いた図

置き場所は P1〜P5 の指定どおりではないので、録画の最初の俯瞰画像から色検出で求める。
2026-10-24: 夜の照明(手首のコントラストが学習データより高い)と箱のずれ(約 55px)が失敗の
原因候補になったため、条件を毎回記録して比べられるようにした。
"""

import csv
import glob
import json
import os
import sys
from importlib import import_module

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
audit = import_module("06d_audit_dataset")
place_name = import_module("14_eval").place_name   # 「左奥」「真ん中手前」などの呼び名
TRAIN_BOX = (76, 185, 227, 286)        # 学習データ 150 本の箱(白い紙)の外接矩形
TRAIN_WRIST_CONTRAST = (37.5, 51.6)    # 学習データの手首カメラのコントラスト(明るさの標準偏差)p5〜p95


def cam_names(d):
    """録画のカメラ名(方策ごとに違う: pi05 = base_0_rgb/left_wrist_0_rgb、smolvla = camera1/camera2、act = top/wrist)。"""
    for top, wrist in (("base_0_rgb", "left_wrist_0_rgb"), ("camera1", "camera2"), ("top", "wrist")):
        if glob.glob(os.path.join(d, f"*_{top}.jpg")):
            return top, wrist
    return "base_0_rgb", "left_wrist_0_rgb"


def first_frames(d):
    obs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if '"obs"' in l]
    if not obs:
        return None, None
    s = obs[0]["step"]
    # 手首カメラは学習データと同じく開始 1 秒後の画像で比べる(ホーム姿勢のままだと机が映らない)
    w = next((o for o in obs if o["obs_t"] - obs[0]["obs_t"] >= 1.0), obs[-1])["step"]
    top, wrist = cam_names(d)
    return cv2.imread(os.path.join(d, f"{s:05d}_{top}.jpg")), cv2.imread(os.path.join(d, f"{w:05d}_{wrist}.jpg"))


def main(label):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "eval", "results.csv"))) if r["run"] == label]
    ref = cv2.resize(cv2.imread(os.path.join(ROOT, "outputs", "cam_test", "ref_top_so101_cube_v3_s1.jpg")), (640, 360))
    print(f"{'#':>2} {'色':6} {'対象の位置':>16} {'箱のずれ':>8} {'明るさ':>6} {'手首ｺﾝﾄﾗｽﾄ':>9} 結果 時間")
    for r in rows:
        tag = f"{label}_{r['ckpt']}_t{int(r['trial']):02d}_"
        dirs = sorted(glob.glob(os.path.join(ROOT, "outputs", "eval", "rec", tag + "*")))
        if not dirs:
            print(f"{r['trial']:>2} {r['color']:6} (録画なし) {r['result']}")
            continue
        top, wr = first_frames(dirs[-1])   # やり直し(x)があれば最後の録画が記録された試行
        if top is None:
            continue
        p = audit.find_cube(top, r["color"])
        box = audit.find_box(top)
        shift = "?" if box is None else f"{box[0] - TRAIN_BOX[0]:+d},{box[1] - TRAIN_BOX[1]:+d}"
        bright = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)[..., 2].mean()
        contrast = cv2.cvtColor(wr, cv2.COLOR_BGR2HSV)[..., 2].std()
        flag = "" if TRAIN_WRIST_CONTRAST[0] <= contrast <= TRAIN_WRIST_CONTRAST[1] else "!"
        pos = "未検出" if p is None else f"{place_name(*p)}({p[0]:.0f},{p[1]:.0f})"
        t = r["note"].split(" ")[0].replace("t=", "")
        print(f"{r['trial']:>2} {r['color']:6} {pos:>16} {shift:>8} {bright:6.0f} {contrast:8.1f}{flag:1} {r['result']:>3}  {t}")
        if p is not None:
            c = (0, 200, 0) if r["result"] == "s" else (0, 0, 255)
            if r["result"] == "s":
                cv2.circle(ref, (int(p[0]), int(p[1])), 10, c, 2)
            else:
                cv2.drawMarker(ref, (int(p[0]), int(p[1])), c, cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(ref, r["trial"], (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    k = sum(r["result"] == "s" for r in rows)
    print(f"成功 {k}/{len(rows)}   (! = 手首カメラのコントラストが学習データの範囲外)")
    out = os.path.join(ROOT, "outputs", "eval", f"analysis_{label.replace('+', '_')}.jpg")
    cv2.imwrite(out, ref)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pi05_v3full+round0")
