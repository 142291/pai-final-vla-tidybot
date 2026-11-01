"""学習に入れてよいエピソードかを 1 本ずつ監査する(映像を全フレーム復号して確かめる)。

使い方: .venv/bin/python scripts/06d_audit_dataset.py <データセット名> [<データセット名> ...]
出力:   outputs/audit/<データセット名>.csv  と  outputs/audit/<データセット名>_end.jpg(終了フレームの一覧)

確かめること(✗ が 1 つでもあれば除外候補):
  ラベル   指示文の色のキューブが、終了時に箱の中にある(=言語ラベルと実際の動作が一致)
  巻き添え 指示されていない色のキューブが動いていない / 箱に入っていない
  開始     対象のキューブが開始時に箱の外にある
  画角     箱の位置がデータセット全体の中央値から大きく外れていない
  映像     動画のフレーム数が表データと一致し、固まり(同一フレームの連続)や真っ暗がない
  数値     NaN が無い、タイムスタンプが 1/fps 刻み、関節値が想定範囲内
  遅れ     リーダー(action)→フォロワー(state)の遅れがデータセット内でそろっている
  静止     開始時・終了時に 2 秒以上止まっていない

2026-10-18: so101_cube_v2 にカメラ画角の違う 27 本が混ざっていたことを受けて作った。
"""

import glob
import json
import os
import sys

import av
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HSV = {  # (下限, 上限) のリスト。赤は色相が 0 付近と 180 付近に分かれる
    "red": [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (180, 255, 255))],
    # 18 からだと明るい赤キューブの橙色を拾った(10-18)。10-25: ライトで白飛びした黄(彩度 64〜100)を拾えず
    # 通し動作で「黄が無い」と判断していたので、彩度の下限を 130→70 に下げた(学習 75 枚で位置は変わらず、未検出 17→8)
    "yellow": [((22, 70, 120), (36, 255, 255))],
    "purple": [((125, 50, 35), (165, 255, 255))],
}
KERNEL = np.ones((5, 5), np.uint8)


def color_of(task: str) -> str:
    for c in HSV:
        if c in task:
            return c
    return "?"


def find_cube(bgr, color):
    """色マスクを開演算して細い配線を消し、最大の塊の重心を返す(無ければ None)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = sum(cv2.inRange(hsv, lo, hi) for lo, hi in HSV[color])
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    # キューブらしい塊だけ: 面積 120〜2500px、縦横比 0.5〜2(配線やアームの部品を除く)
    cand = [i for i in range(1, n)
            if 120 <= stats[i, cv2.CC_STAT_AREA] <= 2500
            and 0.5 <= stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[i, cv2.CC_STAT_HEIGHT]) <= 2.0]
    if not cand:
        return None
    i = max(cand, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    return tuple(cents[i])


def find_box(bgr):
    """箱 = 画面左側の、明るく(V>=150)色の薄い(S<=60)、四角く詰まった大きな塊(白い紙)。その外接矩形を返す。

    2026-10-24: 以前は左半分を大津の二値化で分けて最大の明るい塊を取っていたが、夜の照明(S3)では
    ライトで明るくなった床と混ざり、S3 の 50 本すべてと 10/24 の評価で箱を誤検出していた
    (例: (197,139,340,360))。彩度と形(詰まり具合・縦横比)で床を除くようにした。
    学習データ 150 本では (76,185,227,286) 付近で安定する。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= 150) & (hsv[..., 1] <= 60)).astype(np.uint8) * 255
    m[:, 360:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 8000 <= area <= 45000 and area / float(w * h) >= 0.75 and 1.1 <= w / h <= 2.2:
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    return (int(x), int(y), int(x + w), int(y + h))


def inside(pt, box, margin=-6):
    if pt is None or box is None:
        return False
    x0, y0, x1, y1 = box
    return x0 - margin <= pt[0] <= x1 + margin and y0 - margin <= pt[1] <= y1 + margin


def decode_episode_frames(root, key, ep_meta):
    """エピソードの全フレームを復号し、(先頭 BGR, 末尾 BGR, 小さいグレー列) を返す。"""
    chunk = int(ep_meta[f"videos/{key}/chunk_index"])
    fidx = int(ep_meta[f"videos/{key}/file_index"])
    t0 = float(ep_meta[f"videos/{key}/from_timestamp"])
    t1 = float(ep_meta[f"videos/{key}/to_timestamp"])
    path = os.path.join(root, "videos", key, f"chunk-{chunk:03d}", f"file-{fidx:03d}.mp4")
    container = av.open(path)
    stream = container.streams.video[0]
    container.seek(int(max(0.0, t0 - 0.5) / stream.time_base), stream=stream, backward=True, any_frame=False)
    first = last = None
    smalls = []
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t < t0 - 1e-3:
            continue
        if t > t1 - 1e-3:
            break
        img = frame.to_ndarray(format="bgr24")
        if first is None:
            first = img
        last = img
        smalls.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA))
    container.close()
    return first, last, smalls


def lag_frames(a, s):
    x = a - a.mean()
    y = s - s.mean()
    best, bestc = 0, -2
    for k in range(0, 15):
        c = np.dot(x[: len(x) - k], y[k:]) / (np.linalg.norm(x[: len(x) - k]) * np.linalg.norm(y[k:]) + 1e-9)
        if c > bestc:
            best, bestc = k, c
    return best


def audit(name):
    root = os.path.join(ROOT, "data", name)
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    fps = info["fps"]
    meta = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)))
    meta = meta.sort_values("episode_index").reset_index(drop=True)
    rng = os.environ.get("AUDIT_EPS")  # 例: AUDIT_EPS=50-66  (追記した分だけ確かめる)
    if rng:
        lo, hi = (int(x) for x in rng.split("-"))
        meta = meta[(meta["episode_index"] >= lo) & (meta["episode_index"] <= hi)].reset_index(drop=True)
    data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)))
    rows, ends = [], []
    for _, ep in meta.iterrows():
        e = int(ep["episode_index"])
        task = str(ep["tasks"][0])
        target = color_of(task)
        d = data[data["episode_index"] == e].sort_values("frame_index")
        A = np.stack(d["action"].to_numpy())
        S = np.stack(d["observation.state"].to_numpy())
        ts = d["timestamp"].to_numpy()
        r = {"ep": e, "task": target, "len_s": round(len(d) / fps, 1)}
        problems = []

        # 数値
        if np.isnan(A).any() or np.isnan(S).any():
            problems.append("NaN")
        dt = np.diff(ts)
        if len(dt) and (np.abs(dt - 1 / fps).max() > 1e-3):
            problems.append("時刻の刻みが不正")
        if np.abs(A[:, :5]).max() > 100.5 or A[:, 5].min() < -0.5 or A[:, 5].max() > 100.5:
            problems.append("関節値が範囲外")

        # 静止
        v = np.linalg.norm(np.diff(A[:, :5], axis=0), axis=1)
        moving = v > 0.15
        r["idle_start_s"] = round(np.argmax(moving) / fps, 1) if moving.any() else r["len_s"]
        r["idle_end_s"] = round(np.argmax(moving[::-1]) / fps, 1) if moving.any() else r["len_s"]
        if r["idle_start_s"] > 2:
            problems.append(f"開始時に{r['idle_start_s']}s静止")
        if r["idle_end_s"] > 2:
            problems.append(f"終了時に{r['idle_end_s']}s静止")

        # フレーム落ちの目安: 1 フレームで関節が 12 単位以上飛ぶ(通常の操作では 1 フレーム数単位)
        jump = np.abs(np.diff(A[:, :5], axis=0)).max() if len(A) > 1 else 0
        r["max_jump"] = round(float(jump), 1)
        if jump > 12:
            problems.append(f"動作が1フレームで{jump:.0f}飛んだ(フレーム落ちの疑い)")

        # 遅れ(肩上下)
        r["lag_f"] = lag_frames(A[:, 1], S[:, 1])

        # 映像
        top0, top1, top_small = decode_episode_frames(root, "observation.images.top", ep)
        _, _, wr_small = decode_episode_frames(root, "observation.images.wrist", ep)
        r["n_top"], r["n_wrist"] = len(top_small), len(wr_small)
        if abs(len(top_small) - len(d)) > 1 or abs(len(wr_small) - len(d)) > 1:
            problems.append(f"フレーム数不一致(表{len(d)}/俯瞰{len(top_small)}/手首{len(wr_small)})")
        for label, sm in (("俯瞰", top_small), ("手首", wr_small)):
            if not sm:
                problems.append(f"{label}映像なし")
                continue
            arr = np.stack(sm).astype(np.int16)
            diffs = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
            run, longest = 0, 0
            for x in diffs:
                run = run + 1 if x < 0.05 else 0
                longest = max(longest, run)
            if longest >= 15:
                problems.append(f"{label}映像が{longest}フレーム固まっている")
            if arr.mean() < 15:
                problems.append(f"{label}映像が暗すぎる")

        # 箱・キューブ
        box0, box1 = find_box(top0), find_box(top1)
        r["box"] = box0
        cubes0 = {c: find_cube(top0, c) for c in HSV}
        cubes1 = {c: find_cube(top1, c) for c in HSV}
        r["target_start"] = cubes0.get(target)
        r["target_end"] = cubes1.get(target)
        if box0 is None:
            problems.append("箱を検出できない")
        if cubes0.get(target) is None:
            problems.append("開始時に対象キューブが見えない")
        elif inside(cubes0[target], box0):
            problems.append("開始時に対象がすでに箱の中")
        if not inside(cubes1.get(target), box1 or box0, margin=4):
            problems.append("終了時に対象が箱の中に無い(失敗 or ラベル違い)")
        for c in HSV:
            if c == target:
                continue
            p0, p1 = cubes0.get(c), cubes1.get(c)
            if p1 is not None and inside(p1, box1 or box0) and not inside(p0, box0):
                problems.append(f"{c}も箱に入った(ラベル違いの疑い)")
            elif p0 is not None and p1 is not None and np.hypot(p0[0] - p1[0], p0[1] - p1[1]) > 25:
                problems.append(f"{c}が{np.hypot(p0[0]-p1[0], p0[1]-p1[1]):.0f}px動いた")
        r["problems"] = " / ".join(problems)
        rows.append(r)
        ends.append((e, target, top1, cubes1.get(target), box1 or box0, bool(problems)))
        print(f"  ep{e:3d} {target:6s} {'✗ ' + r['problems'] if problems else '✓'}", flush=True)

    df = pd.DataFrame(rows)
    # 画角: 箱の左上・右下が中央値から 12px 以上ずれていたら外れ値
    boxes = np.array([b for b in df["box"] if b is not None])
    med = np.median(boxes, axis=0)
    for i, b in enumerate(df["box"]):
        if b is not None and np.abs(np.array(b) - med).max() > 12:
            df.loc[i, "problems"] = (df.loc[i, "problems"] + " / " if df.loc[i, "problems"] else "") + \
                f"画角が外れ値(箱 {tuple(int(x) for x in b)} vs 中央値 {tuple(int(x) for x in med)})"
    # 遅れ: データセットの最頻値から 2 フレーム以上違う
    mode = int(df["lag_f"].mode()[0])
    for i, l in enumerate(df["lag_f"]):
        if abs(l - mode) >= 2:
            df.loc[i, "problems"] = (df.loc[i, "problems"] + " / " if df.loc[i, "problems"] else "") + f"遅れが{l}フレーム(普段{mode})"

    os.makedirs(os.path.join(ROOT, "outputs", "audit"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "outputs", "audit", f"{name}{'_' + rng if rng else ''}.csv"), index=False)

    # 終了フレームの一覧(目視確認用): 緑枠=箱、丸=対象キューブ、赤い帯=問題あり
    tiles = []
    for e, target, img, pt, box, bad in ends:
        t = img.copy()
        if box:
            cv2.rectangle(t, box[:2], box[2:], (0, 255, 0), 2)
        if pt is not None:
            cv2.circle(t, (int(pt[0]), int(pt[1])), 14, (255, 255, 255), 3)
        prob = df.loc[df["ep"] == e, "problems"].iloc[0]
        cv2.rectangle(t, (0, 0), (640, 34), (0, 0, 200) if prob else (40, 40, 40), -1)
        cv2.putText(t, f"ep{e} {target}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(t, (320, 180)))
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    suffix = f"_{rng}" if rng else ""
    cv2.imwrite(os.path.join(ROOT, "outputs", "audit", f"{name}{suffix}_end.jpg"), sheet)

    print(f"\n=== {name}: {len(df)} 本 / 問題なし {int((df['problems'] == '').sum())} 本 ===")
    print(f"箱の位置(中央値) {tuple(int(x) for x in med)} / 遅れの最頻値 {mode} フレーム")
    for _, r in df[df["problems"] != ""].iterrows():
        print(f"  ✗ ep{r['ep']} ({r['task']}): {r['problems']}")
    return df


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(f"### {n}")
        audit(n)
