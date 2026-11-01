#!/usr/bin/env bash
# 指定した設定でカメラを実際に開けるか試し、1 フレームずつ保存する。
# 2 台同時に開けるかも確認するので、USB 帯域の問題を切り分けられる。
# 使い方: scripts/05b_test_cameras.sh [幅 高さ fps]   省略時 640 480 30
# macOS のカメラ権限が要るので「ターミナル.app」から実行すること。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/.venv/bin/activate"
W="${1:-640}"; H="${2:-480}"; F="${3:-30}"
OUT="$ROOT/outputs/cam_test"; mkdir -p "$OUT"
python - "$W" "$H" "$F" "$OUT" <<'PY' 2>&1 | grep -viE "^objc\[|AVFFrameReceiver|AVFAudioReceiver|global cap"
import sys, cv2
W, H, F, OUT = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
MJPG = cv2.VideoWriter_fourcc(*"MJPG")
print(f"=== 個別に開く ({W}x{H} @{F:.0f}fps, MJPG) ===")
ok = []
for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"  index {i}: 開けない"); cap.release(); continue
    cap.set(cv2.CAP_PROP_FOURCC, MJPG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H); cap.set(cv2.CAP_PROP_FPS, F)
    for _ in range(5): cap.read()          # ウォームアップ
    r, fr = cap.read()
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    if r:
        cv2.imwrite(f"{OUT}/cam{i}.jpg", fr); ok.append(i)
        print(f"  index {i}: OK  実際 {aw}x{ah} @{af:.0f}fps  -> {OUT}/cam{i}.jpg")
    else:
        print(f"  index {i}: 開いたが読めない")
    cap.release()

print(f"\n=== 同時に開く(2 台同時に使えるかの本番テスト) ===")
caps = {}
for i in ok:
    c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    c.set(cv2.CAP_PROP_FOURCC, MJPG)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H); c.set(cv2.CAP_PROP_FPS, F)
    caps[i] = c
for _ in range(5):
    for c in caps.values(): c.read()
good = []
for i, c in caps.items():
    r, fr = c.read()
    print(f"  index {i}: {'同時読み取り OK' if r else '同時だと読めない(帯域不足の疑い)'}")
    if r: good.append(i)
for c in caps.values(): c.release()
print(f"\n同時に使える index: {good}")
print(f"画像を見てどれが俯瞰/手首か確認: open {OUT}")
PY
