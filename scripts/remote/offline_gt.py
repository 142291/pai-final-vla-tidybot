"""学習済みモデルに記録データの映像を見せ、出した行動を人の操作(正解)と比べる(H200 で動かす)。

使い方(H200 で):
    python scripts/remote/offline_gt.py <チェックポイントのディレクトリ> [--dataset so101_cube_v3_train] [--n 40]

各エピソードから等間隔にフレームを取り、そのフレームでの行動チャンク(50 ステップ)を予測して、
記録された行動(正解)との差を関節ごとに出す。単位は記録と同じ(関節は -100〜100、グリッパは 0〜100)。

  first … チャンクの最初の 1 ステップの誤差(今すぐどこへ動くか)
  chunk … チャンク全体(50 ステップ)の平均誤差(この先 1.7 秒の軌道)

差が小さい  → モデルは学習できている。実機で失敗するなら動かし方(遅延・混ぜ方)側の問題。
差が大きい  → 学習そのものの問題(学習量・設定)。

2026-10-19: 本番モデルが実機で 4 試行中 1 成功だったので、原因をモデル側か実行側かに切り分けるために作った。
"""

import argparse
import os
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

JOINTS = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--dataset", default="so101_cube_v3_train")
    ap.add_argument("--policy", default="pi05", help="pi05 / smolvla / act(比較実験用、2026-10-25)")
    ap.add_argument("--n", type=int, default=40, help="評価するフレーム数")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--steps", type=int, default=0, help="flow matching の反復回数を上書き(既定は学習時の 10)")
    ap.add_argument("--episodes", default="", help="例: 18,50,100,144(学習に使っていない回で汎化を見る)")
    args = ap.parse_args()

    root = os.path.expanduser(f"~/ai-arm/data/{args.dataset}")
    eps = [int(x) for x in args.episodes.split(",")] if args.episodes else None
    ds = LeRobotDataset(f"local/{args.dataset}", root=root, episodes=eps,
                        delta_timestamps={"action": [i / 30 for i in range(args.chunk)]})
    policy_cls = get_policy_class(args.policy)
    policy = policy_cls.from_pretrained(args.ckpt).to("cuda").eval()
    if args.steps and hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = args.steps
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}},
                                         postprocessor_overrides={"device_processor": {"device": "cuda"}})

    rng = np.random.default_rng(0)
    idxs = sorted(rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist())
    first, chunk, horizon, moving, infer_ms = [], [], [], [], []
    for i in idxs:
        item = ds[i]
        gt = item["action"].numpy()                      # (chunk, 6) 記録された行動
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in item.items()
                 if k.startswith("observation") or k == "task"}
        with torch.no_grad():
            b = pre(batch)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pred = policy.predict_action_chunk(b)        # (1, chunk, 6) 正規化された行動
            torch.cuda.synchronize(); infer_ms.append((time.perf_counter() - t0) * 1000)
            pred = torch.stack([post(pred[:, t]) for t in range(pred.shape[1])], dim=1)[0].float().cpu().numpy()
        n = min(len(gt), len(pred))
        horizon.append(np.abs(pred[:n] - gt[:n]).mean(axis=1))       # (n,) 予測の何ステップ目でどれだけ外れるか
        moving.append(np.abs(gt - gt[0]).mean() > 2.0)                 # 人が動かしている区間か(止まっている区間を除くため)
        first.append(np.abs(pred[0] - gt[0]))
        chunk.append(np.abs(pred[:n] - gt[:n]).mean(axis=0))
    first, chunk = np.array(first), np.array(chunk)
    print(f"{args.ckpt}  ({len(idxs)} フレーム)  反復 {getattr(policy.config, "num_inference_steps", getattr(policy.config, "num_steps", "-"))} 回  推論 p50 {np.median(infer_ms[1:]):.0f}ms")
    print("          " + " ".join(f"{j:>7s}" for j in JOINTS))
    print("first 平均 " + " ".join(f"{x:7.2f}" for x in first.mean(0)))
    print("chunk 平均 " + " ".join(f"{x:7.2f}" for x in chunk.mean(0)))
    print("chunk 最悪 " + " ".join(f"{x:7.2f}" for x in chunk.max(0)))
    # 比較の物差し: 50 ステップの間に記録の行動自体がどれだけ動くか(何も予測しないで「今の値のまま」とした場合の誤差)
    base = []
    for i in idxs:
        gt = ds[i]["action"].numpy()
        base.append(np.abs(gt - gt[0]).mean(axis=0))
    H = np.array([h[:args.chunk] for h in horizon if len(h) >= args.chunk])
    M = np.array([m for h, m in zip(horizon, moving) if len(h) >= args.chunk])
    for label, sel in (("全フレーム", np.ones(len(H), bool)), ("動いている区間", M)):
        if sel.sum():
            h = H[sel].mean(0)
            print(f"予測ステップ別の誤差({label} {sel.sum()} 個): " + " ".join(f"{a}-{b}:{h[a:b].mean():.1f}" for a, b in ((0, 5), (5, 15), (15, 25), (25, 35), (35, 50))))
    print("参考: 静止予測 " + " ".join(f"{x:7.2f}" for x in np.array(base).mean(0)) + "   ← chunk 平均がこれより十分小さければ学習できている")


if __name__ == "__main__":
    main()
