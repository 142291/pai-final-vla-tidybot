"""記録した観測(画像 2 枚 + 関節角)をモデルに見せ、予測した行動チャンクで「どれだけ動こうとするか」を測る(H200 で動かす)。

使い方:
    python scripts/remote/replay_probe.py <probe フォルダ> <チェックポイント> [<チェックポイント> ...]
probe フォルダには <名前>_top.jpg / <名前>_wrist.jpg / <名前>.json({"state": {...}, "task": "..."})を置く。

出力: 観測ごと・モデルごとに、チャンク 50 ステップの中で現在の姿勢から最も離れる量(関節ごとの最大差の最大)
と、肩の持ち上げ(shoulder_lift)の最大変化。ほぼ 0 なら「動かない」と予測している。

2026-10-26: 通し動作で 1 個目が箱に入った後、π0.5 が「その場から動かない」予測を出し続けた(学習データに
箱に入った状態から始まる回が 0 本)。「箱に入った状態から」の手本を足したラウンド 1 で直ったかを、
実機を使わずに確かめるために作った。
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def load(path):
    img = cv2.imread(path)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)


def main(probe_dir, ckpts):
    probes = sorted(p[:-5] for p in glob.glob(os.path.join(probe_dir, "*.json")))
    for ck in ckpts:
        policy = get_policy_class("pi05").from_pretrained(ck).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=ck,
                                             preprocessor_overrides={"device_processor": {"device": "cuda"}},
                                             postprocessor_overrides={"device_processor": {"device": "cuda"}})
        print(f"### {ck}")
        for p in probes:
            meta = json.load(open(p + ".json"))
            s = np.array([meta["state"][j] for j in JOINTS], dtype=np.float32)
            batch = {"observation.images.top": load(p + "_top.jpg"), "observation.images.wrist": load(p + "_wrist.jpg"),
                     "observation.state": torch.from_numpy(s).unsqueeze(0), "task": [meta["task"]]}
            moves = []
            for _ in range(3):   # flow matching はノイズから始まるので 3 回の平均
                with torch.no_grad():
                    a = policy.predict_action_chunk(pre(batch))
                    a = torch.stack([post(a[:, t]) for t in range(a.shape[1])], dim=1)[0].float().cpu().numpy()
                d = np.abs(a - s[None, :])
                moves.append((d[:, :5].max(), d[:, 1].max()))
            m = np.mean(moves, 0)
            print(f"  {os.path.basename(p):28s} 最大の動き {m[0]:6.1f}   肩の持ち上げ {m[1]:6.1f}   ({meta.get('note', '')})")
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
