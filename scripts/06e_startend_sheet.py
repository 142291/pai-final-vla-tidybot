"""エピソードの開始フレームと終了フレームを並べた一覧画像を作る(監査の目視確認用)。

使い方: .venv/bin/python scripts/06e_startend_sheet.py <データセット名> <最初のep> <最後のep> [出力パス]
既定の出力: outputs/audit/<データセット名>_<最初>-<最後>_startend.jpg

06d_audit_dataset.py の自動判定は、強い日差し(赤が橙・桃に見える)や夜(箱の中が一番明るくない)で
誤検出が多い。✗ が出たら必ずこれで「指示した色のキューブが最後に箱の中にあるか」を目で確かめる。
"""

import glob
import importlib.util
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("aud", os.path.join(HERE, "06d_audit_dataset.py"))
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)


def main():
    name, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    root = os.path.join(HERE, "..", "data", name)
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "..", "outputs", "audit", f"{name}_{lo}-{hi}_startend.jpg")
    meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)))
    meta = meta.set_index("episode_index")
    tiles = []
    for e in range(lo, hi + 1):
        f0, f1, _ = aud.decode_episode_frames(root, "observation.images.top", meta.loc[e])
        task = aud.color_of(str(meta.loc[e, "tasks"][0]))

        def lab(img, t):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (640, 34), (0, 0, 0), -1)
            cv2.putText(img, t, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            return cv2.resize(img, (300, 169))

        tiles.append(np.hstack([lab(f0, f"{e} {task} S"), lab(f1, f"{e} E")]))
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]))
    print(out)


if __name__ == "__main__":
    main()
