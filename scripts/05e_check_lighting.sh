#!/usr/bin/env bash
# いまの照明が学習データの分布内にあるかを測る。
#
# 使い方: scripts/05e_check_lighting.sh [データセット名]
#
# 模倣学習のポリシーは学習時と同じ見え方でしか信頼できない。
# 照明が分布から外れていると、両方のポリシーが「毎回同じ方向に外す」という
# 症状が出る(2026-09-29 に実際に遭遇)。動かす前にこれで確認する。
#
# ■ 判定に使うのは固定カメラ(top)だけ。手首カメラはアームと一緒に動くので、
#   写る中身が姿勢に依存し、学習データ(エピソード開始時=ホーム姿勢)とは比較にならない。
#   参考値としては出すが、判定には含めない。
#
# ■ カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
NAME="${1:-so101_cube_v2}"

python3 - "$NAME" "$CAMERAS" <<'PY'
import sys, re, glob, json, os, subprocess, numpy as np, cv2, pandas as pd
name, cams = sys.argv[1], sys.argv[2]
idx = {m[0]: int(m[1]) for m in re.findall(r"(\w+):\s*\{[^}]*index_or_path:\s*(\d+)", cams)}
cache = f"data/{name}/.lighting_stats.json"

# 学習データの統計(初回だけ計算してキャッシュ)
if os.path.exists(cache):
    ref = json.load(open(cache))
else:
    meta = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"data/{name}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    ref = {}
    for cam in idx:
        k = f"videos/observation.images.{cam}"
        if k + "/chunk_index" not in meta.columns:
            continue
        vals = []
        for _, r in meta.iterrows():
            vid = (f"data/{name}/videos/observation.images.{cam}/"
                   f"chunk-{int(r[k+'/chunk_index']):03d}/file-{int(r[k+'/file_index']):03d}.mp4")
            subprocess.run(["ffmpeg","-loglevel","error","-y","-ss",
                            str(float(r[k+"/from_timestamp"])+0.4),"-i",vid,"-frames:v","1","/tmp/_l.png"],check=True)
            im = cv2.imread("/tmp/_l.png")
            vals.append([im[:,:,2].mean(), im[:,:,1].mean(), im[:,:,0].mean(),
                         cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()])
        a = np.array(vals)
        ref[cam] = {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(), "n": len(a)}
    json.dump(ref, open(cache, "w"), indent=1)
    print(f"学習データの統計を計算してキャッシュした: {cache}")

# 固定カメラだけを判定に使う(手首はアーム姿勢に依存するため参考値)
FIXED = {"top"}
worst = 0.0
for cam, i in idx.items():
    if cam not in ref:
        continue
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    for _ in range(10):
        ok, fr = cap.read()
    cap.release()
    if not ok:
        print(f"!! カメラ {i} ({cam}) から取得できず"); continue
    now = np.array([fr[:,:,2].mean(), fr[:,:,1].mean(), fr[:,:,0].mean(),
                    cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean()])
    m, s = np.array(ref[cam]["mean"]), np.maximum(np.array(ref[cam]["std"]), 1e-6)
    z = (now - m) / s
    tag = "" if cam in FIXED else "  ※参考値: アーム姿勢に依存するため判定に使わない"
    print(f"\n=== {cam} (学習データ {ref[cam]['n']} 本ぶん){tag} ===")
    print(f"{'':<6}{'学習時':>9}{'いま':>9}{'差':>9}{'σ':>9}")
    for n_, mm, nn, zz in zip(["R","G","B","輝度"], m, now, z):
        print(f"{n_:<6}{mm:>9.1f}{nn:>9.1f}{nn-mm:>+9.1f}{zz:>+9.1f}")
    if cam in FIXED:
        worst = max(worst, float(np.abs(z).max()))
        signed = z[:3]
        if signed.max() < 0:
            print("  → 全チャンネルが低い = 単純に暗い。照明を足すこと。")
        elif signed.min() > 0:
            print("  → 全チャンネルが高い = 明るすぎる。光を減らすこと。")
        elif abs(signed[0] - signed[2]) > 3:
            print("  → R と B の差が大きい = 色かぶり。光源の種類(昼光/電球)が違う。")

print()
if worst < 3:
    print(f"判定: ○ 分布内 (最大 {worst:.1f} σ)。学習時と同じ見え方。")
elif worst < 6:
    print(f"判定: △ やや外れている (最大 {worst:.1f} σ)。精度が落ちる可能性。")
else:
    print(f"判定: ✗ 分布外 (最大 {worst:.1f} σ)。")
    print("      ポリシーは学習時と違う見え方の画像を受け取っている。")
    print("      照明を学習時に合わせるか、この条件でも録画して学習し直すこと。")
PY
