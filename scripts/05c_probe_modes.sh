#!/usr/bin/env bash
# 各カメラが「実際に安定して出せる」解像度を調べる。
# 1 フレーム目だけ見ても分からないので、2 台同時に 100 フレーム読んで、途中でサイズが変わらないか確認する。
# 使い方: scripts/05c_probe_modes.sh
# カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/.venv/bin/activate"
python - <<'PY' 2>&1 | grep -viE "^objc\[|AVFFrameReceiver|AVFAudioReceiver|global cap"
import cv2, time, collections
CAMS = [0, 1]                                   # 0=手首 1=俯瞰
MODES = [(640,360), (640,480), (800,448), (960,540), (320,240)]
FOURCCS = ["MJPG", None]
N = 100                                          # 連続読み取り数

for fourcc in FOURCCS:
    for (W, H) in MODES:
        caps, sizes, fails = {}, {}, {}
        okopen = True
        for i in CAMS:
            c = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if not c.isOpened(): okopen = False; break
            if fourcc: c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, W); c.set(cv2.CAP_PROP_FRAME_HEIGHT, H); c.set(cv2.CAP_PROP_FPS, 30)
            caps[i] = c; sizes[i] = collections.Counter(); fails[i] = 0
        if not okopen:
            for c in caps.values(): c.release()
            print(f"{str(fourcc):5s} {W}x{H}: 開けない"); continue
        for _ in range(10):                      # ウォームアップ
            for c in caps.values(): c.read()
        t0 = time.time()
        for _ in range(N):
            for i, c in caps.items():
                r, f = c.read()
                if r and f is not None: sizes[i][(f.shape[1], f.shape[0])] += 1
                else: fails[i] += 1
        dt = time.time() - t0
        for c in caps.values(): c.release()
        parts = []
        for i in CAMS:
            got = sizes[i].most_common()
            desc = " ".join(f"{w}x{h}x{n}" for (w,h),n in got) or "読めず"
            stable = len(got) == 1 and got[0][0] == (W,H) and fails[i] == 0
            parts.append(f"cam{i}[{desc}{' 失敗'+str(fails[i]) if fails[i] else ''}]{'OK' if stable else '!!'}")
        print(f"{str(fourcc):5s} 要求 {W}x{H}: {'  '.join(parts)}   {N/dt:.0f} 周/秒")
print("\n全部 OK かつ要求どおりのサイズが出ているものを config.env に使う")
PY
