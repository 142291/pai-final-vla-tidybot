#!/usr/bin/env bash
# いまのカメラ映像と、録画時のフレームを並べて比較する。
#
# 使い方: scripts/05d_compare_framing.sh [データセット名]
#
# カメラや台が動いていると、ポリシーは学習時と違う見え方の画像を受け取る。
# 両方のポリシーが「毎回同じ方向に外す」ときの第一容疑者。
# 比べるのは箱・台の縁・アームの土台など動かないはずのもの。キューブは動いてよい。
#
# ■ カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
NAME="${1:-so101_cube_v2}"
OUT="framing_check"
mkdir -p "$OUT"

python3 - "$NAME" "$OUT" "$CAMERAS" <<'PY'
import sys, re, subprocess, glob, pandas as pd, cv2
from PIL import Image, ImageDraw
name, out, cams = sys.argv[1], sys.argv[2], sys.argv[3]

# CAMERAS 文字列から index を拾う
idx = {m[0]: int(m[1]) for m in re.findall(r"(\w+):\s*\{[^}]*index_or_path:\s*(\d+)", cams)}
print("カメラ:", idx)

pairs = []
for cam_name, i in idx.items():
    # いまの映像
    cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    for _ in range(10):      # ウォームアップ(最初の数枚は露出が合っていない)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"!! カメラ {i} ({cam_name}) から取得できず"); continue
    now = f"{out}/{cam_name}_now.png"; cv2.imwrite(now, frame)

    # 録画時のフレーム(エピソード0の開始)
    key = f"videos/observation.images.{cam_name}"
    meta = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(f"data/{name}/meta/episodes/**/*.parquet", recursive=True))]).sort_values("episode_index")
    r = meta.iloc[0]
    vid = (f"data/{name}/videos/observation.images.{cam_name}/"
           f"chunk-{int(r[key+'/chunk_index']):03d}/file-{int(r[key+'/file_index']):03d}.mp4")
    then = f"{out}/{cam_name}_then.png"
    subprocess.run(["ffmpeg","-loglevel","error","-y","-ss",
                    str(float(r[key+"/from_timestamp"])+0.4),"-i",vid,"-frames:v","1",then], check=True)
    pairs.append((cam_name, then, now))

# 上下に並べる(録画時 / いま)
if pairs:
    W, H = 640, 360
    canvas = Image.new("RGB", (W*2, H*len(pairs)), "black")
    d = ImageDraw.Draw(canvas)
    for j, (cam_name, then, now) in enumerate(pairs):
        canvas.paste(Image.open(then).resize((W,H)), (0, j*H))
        canvas.paste(Image.open(now).resize((W,H)),  (W, j*H))
        d.text((6, j*H+6),   f"{cam_name} 録画時", fill="yellow")
        d.text((W+6, j*H+6), f"{cam_name} いま",   fill="cyan")
    p = f"{out}/compare.png"; canvas.save(p)
    print("保存:", p)
PY
