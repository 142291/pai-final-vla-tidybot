"""人の介入の記録(flywheel_human)から、学習に悪い部分を切り落とした新しいデータセットを作る。

使い方:
    .venv/bin/python scripts/22_trim_episodes.py flywheel_human flywheel_human_clean

切り落とすもの:
  1. 先頭の「追いつき」: 人の操作に切り替えた瞬間はリーダーとフォロワーの姿勢が違い、フォロワーが
     リーダーへ急いで寄る(指令と姿勢の差が 16〜19。普通の録画は約 2)。指令と姿勢の差が LEAD_OK 以下に
     収まるまでを捨てる(最大 LEAD_MAX フレーム)。
  2. 末尾: 箱で放した後の静止(3〜4 秒あった)と、ホームへ戻す動き。放した所(後半でグリッパが最も
     開いた所)から TAIL_KEEP フレームだけ残す(CUT_AT_RELEASE=0 で戻す動きは残す)。
  3. SKIP=5 のように指定した回(目視で手などが写り込んでいた回)。
残りが MIN_LEN フレーム未満の回は捨てる。画像は元の動画から読み直して、学習データと同じ符号化で書く。

2026-10-26: ラウンド 2 用の人の手本 11 本のうち 8 本で先頭の急な動き、3 本で末尾の静止が見つかったため作った。
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LEAD_OK = float(os.environ.get("LEAD_OK", "5"))
LEAD_MAX = int(os.environ.get("LEAD_MAX", "45"))
TAIL_KEEP = int(os.environ.get("TAIL_KEEP", "10"))
MIN_LEN = int(os.environ.get("MIN_LEN", "90"))
CUT_AT_RELEASE = os.environ.get("CUT_AT_RELEASE", "0") == "1"   # 既定は切らない(学習データも 87% がホームへ戻ってから終わっている)
SKIP = {int(x) for x in os.environ.get("SKIP", "").split(",") if x.strip()}   # 目視で除く回(例: SKIP=5)


def main(src, dst):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{src}", root=os.path.join(ROOT, "data", src))
    ref = json.load(open(os.path.join(ROOT, "data", src, "meta", "info.json")))["features"]
    feats = {k: dict(v, shape=tuple(v["shape"])) for k, v in ref.items()
             if k in ("action", "observation.state", "observation.images.top", "observation.images.wrist")}
    out = LeRobotDataset.create(f"local/{dst}", 30, feats, root=os.path.join(ROOT, "data", dst), robot_type="so_follower",
                                use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264_videotoolbox", crf=23),
                                streaming_encoding=True, encoder_threads=4)
    eps = ds.meta.episodes
    for ep in range(ds.meta.total_episodes):
        if ep in SKIP:
            print(f"  ep{ep}: 目視で除外")
            continue
        a, b = eps["dataset_from_index"][ep], eps["dataset_to_index"][ep]
        items = [ds[i] for i in range(a, b)]
        act = np.stack([it["action"].numpy() for it in items])
        st = np.stack([it["observation.state"].numpy() for it in items])
        gap = np.abs(act[:, :5] - st[:, :5]).max(1)
        # 先頭 LEAD_MAX フレームの中で差が LEAD_OK を超えた最後のフレームの次から使う
        # (切り替えた直後は差が小さく、少し遅れて急に寄る回もあるため「最初に収まった所」では足りない)
        over = [i for i in range(min(LEAD_MAX, len(items))) if gap[i] > LEAD_OK]
        start = over[-1] + 1 if over else 0
        moved = np.where(np.abs(np.diff(st[:, :5], axis=0)).max(1) > 0.5)[0]
        end = min(len(items), (moved[-1] + 1 + TAIL_KEEP) if len(moved) else len(items))
        # 箱で放した後にホームへ戻す動きも切る(学習データは「箱で放した所」で終わっている)。
        # 2026-10-26: 人の手本 11 本すべてで、h を押す前にリーダーをホームへ戻していた。
        # 放した所 = 俯瞰カメラで対象が箱の中に見え始め、そのまま最後まで箱の中にある最初のフレーム
        # (アームが箱から離れ始めた所)。そこから TAIL_KEEP フレームだけ残す。
        # (グリッパの開きで探す方法は、掴む前の開きを拾って早すぎる所で切ったのでやめた)
        if CUT_AT_RELEASE:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from importlib import import_module
            audit = import_module("06d_audit_dataset")
            color = items[0]["task"].split()[3]
            inbox = []
            for it in items:
                top = (it["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
                inbox.append(audit.inside(audit.find_cube(top, color), audit.find_box(top)))
            settled = next((i for i in range(len(inbox)) if all(inbox[i:])), None)
            if settled is not None:
                end = min(end, settled + TAIL_KEEP)
        keep = items[start:end]
        task = items[0]["task"]
        if len(keep) < MIN_LEN:
            print(f"  ep{ep} ({task.split()[3]}): 残り {len(keep)} フレームで短すぎるので捨てる")
            continue
        for it in keep:
            out.add_frame({
                "observation.state": it["observation.state"].numpy(),
                "observation.images.top": (it["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                "observation.images.wrist": (it["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                "action": it["action"].numpy(),
                "task": task,
            })
        out.save_episode()
        print(f"  ep{ep} ({task.split()[3]}): {len(items)} → {len(keep)} フレーム(先頭 {start}・末尾 {len(items) - end} を削除)")
    out.finalize()
    print(f"保存: data/{dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
