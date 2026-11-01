#!/usr/bin/env bash
# 録画したデータセットの品質を数値で検証する。
#
# 使い方: scripts/06c_check_dataset.sh <データセット名> [カメラキー]
# 例:    scripts/06c_check_dataset.sh so101_cube_v2
#
# 出すもの:
#   - エピソード数 / フレーム数 / fps / タスク名
#   - エピソード長の分布(上限に張り付いているもの、極端に短いものを警告)
#   - 各エピソードの開始フレームから検出した赤キューブの位置と、その散らばり
#   - 照明のばらつき(均一すぎると時間帯が変わるだけで動かなくなる)
#   - 全エピソードの開始フレームを並べたコンタクトシート PNG
#
# 汎化は対象物の位置のばらつきから来る。散らばりが足りなければ撮り直すこと。
# 詳細は docs/recording-protocol.md
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?データセット名}"; CAM="${2:-observation.images.top}"
python3 - "$NAME" "$CAM" <<'PY'
import sys, glob, json, subprocess, os
import numpy as np, cv2, pandas as pd
from PIL import Image, ImageDraw

name, cam = sys.argv[1], sys.argv[2]
root = f"data/{name}"
if not os.path.isdir(root):
    sys.exit(f"{root} が無い")

info = json.load(open(f"{root}/meta/info.json"))
fps = info.get("fps", 30)
print(f"=== {name} ===")
print(f"エピソード {info.get('total_episodes')} / フレーム {info.get('total_frames')} / fps {fps}")
t = pd.read_parquet(f"{root}/meta/tasks.parquet")
print(f"タスク {len(t)} 種:")
for task in t.index:
    print(f"  - {task}")
if len(t) > 1:
    print("  ! タスク名が複数ある。同じ作業なら揃えるべき(Phase 4 で効く)")

fs = sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in fs]).sort_values("episode_index")

# --- エピソード長 ---
sec = df["length"] / fps
print(f"\n=== エピソード長(秒) ===")
print(f"中央 {sec.median():.1f} / 平均 {sec.mean():.1f} / 最短 {sec.min():.1f} / 最長 {sec.max():.1f}")
cap = df[sec > sec.max() * 0.95]
if len(cap) > 2 and sec.max() > 25:
    eps = ", ".join(f"ep{int(e)}" for e in cap.episode_index)
    print(f"  ! 上限に張り付き: {eps}  (待ち時間や失敗のやり直しが入っていないか確認)")

# --- 赤キューブの位置 ---
k = f"videos/{cam}"
if k + "/chunk_index" not in df.columns:
    sys.exit(f"カメラキー {cam} が見つからない")
tmp = f"/tmp/dscheck_{name}"
os.makedirs(tmp, exist_ok=True)
rows, tiles, lights = [], [], []
for _, r in df.iterrows():
    ep = int(r.episode_index)
    vid = f"{root}/videos/{cam}/chunk-{int(r[k+'/chunk_index']):03d}/file-{int(r[k+'/file_index']):03d}.mp4"
    out = f"{tmp}/ep{ep:03d}.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss",
                    str(float(r[k + "/from_timestamp"]) + 0.4), "-i", vid,
                    "-frames:v", "1", out], check=True)
    img = cv2.imread(out); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 80), (8, 255, 255)) | cv2.inRange(hsv, (172, 120, 80), (180, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lights.append([img[:,:,2].mean(), img[:,:,1].mean(), img[:,:,0].mean(),
                   cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()])
    n, _, st, ce = cv2.connectedComponentsWithStats(m)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        rows.append((ep, int(ce[i][0]), int(ce[i][1])))
    else:
        rows.append((ep, None, None))
    tiles.append((ep, out))

H, W = img.shape[:2]
d = pd.DataFrame(rows, columns=["ep", "x", "y"]).dropna()
print(f"\n=== 赤キューブの開始位置 (画面 {W}x{H}) ===")
miss = [r[0] for r in rows if r[1] is None]
if miss:
    print(f"  ! 検出できず: {', '.join('ep'+str(e) for e in miss)}")
if len(d):
    # 中央値から大きく外れたものは置き忘れ / リセット漏れの疑い
    dist = np.hypot(d.x - d.x.median(), d.y - d.y.median())
    odd = d[dist > max(80, dist.median() * 4)]
    for _, o in odd.iterrows():
        print(f"  ! ep{int(o.ep)} が他と大きく違う位置 ({int(o.x)},{int(o.y)}) — リセット漏れの疑い")
    keep = d.drop(odd.index)
    rx, ry = keep.x.max() - keep.x.min(), keep.y.max() - keep.y.min()
    px, py = rx / W * 100, ry / H * 100
    print(f"x: {keep.x.min():.0f}〜{keep.x.max():.0f}  幅 {rx:.0f}px = 画面の {px:.0f}%")
    print(f"y: {keep.y.min():.0f}〜{keep.y.max():.0f}  幅 {ry:.0f}px = 画面の {py:.0f}%")
    print()
    if px < 30 or py < 30:
        print("判定: ✗ 位置が固定されすぎ。このデータでは汎化しない。")
        print("      リセットのたびに赤キューブを作業範囲全体へ散らすこと。")
    elif px < 50 or py < 50:
        print("判定: △ もう少し端まで使いたい。")
    else:
        print("判定: ○ 十分に散っている。")

# --- 照明のばらつき ---
L = np.array(lights)
print(f"\n=== 照明のばらつき (俯瞰カメラ, {len(L)} 本) ===")
print(f"{'':<6}{'平均':>9}{'std':>9}{'最小':>9}{'最大':>9}")
for i, nm in enumerate(["R", "G", "B", "輝度"]):
    print(f"{nm:<6}{L[:,i].mean():>9.1f}{L[:,i].std():>9.1f}{L[:,i].min():>9.1f}{L[:,i].max():>9.1f}")
spread = L[:, 3].max() - L[:, 3].min()
sd = L[:, 3].std()
print()
if sd < 4:
    print(f"判定: ✗ 照明が均一すぎる (輝度 std {sd:.1f}, 幅 {spread:.1f})")
    print("      全部を同じ時間帯・同じ照明で撮ると、条件が変わるだけで動かなくなる。")
    print("      2026-09-29 に実際に遭遇: std 1.9 のデータで学習し、翌朝 8 時に")
    print("      最大 12.8σ の分布外となって掴めなくなった。")
    print("      時間帯・カーテン・室内灯を変えて撮り足すこと。")
elif sd < 8:
    print(f"判定: △ 照明のばらつきがやや少ない (輝度 std {sd:.1f}, 幅 {spread:.1f})")
else:
    print(f"判定: ○ 照明にばらつきがある (輝度 std {sd:.1f}, 幅 {spread:.1f})")

# --- コンタクトシート ---
tw, th = 213, 120
cols = 6; rowsn = (len(tiles) + cols - 1) // cols
canvas = Image.new("RGB", (tw * cols, th * rowsn), "black")
dr = ImageDraw.Draw(canvas)
for i, (ep, f) in enumerate(tiles):
    canvas.paste(Image.open(f).resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    dr.text(((i % cols) * tw + 4, (i // cols) * th + 4), f"ep{ep}", fill="yellow")
sheet = f"{root}/starts_grid.png"
canvas.save(sheet)
print(f"\nコンタクトシート: {sheet}")
PY
