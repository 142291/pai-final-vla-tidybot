#!/usr/bin/env bash
# 俯瞰カメラを学習データの画角に合わせ込む(ライブで重ね表示＋ずれを数値表示)。
#
# 使い方: scripts/05f_align_camera.sh [データセット名]   既定 so101_cube_v2
#
# 学習データ 1 本目の最初のフレームを半透明で重ね、箱まわり(左下)の
# 位相相関で dx/dy(px)を毎フレーム表示する。|dx|,|dy| が 3px 以下になるまでカメラを動かす。
#   dx > 0 … 今の映像が学習時より右にずれている → 映像を左へ動かす向きにカメラを調整
#   dy > 0 … 今の映像が学習時より下にずれている → 映像を上へ動かす向きにカメラを調整
#   画面上の表示 "move image left 30px" などは「映像をどちらへ何 px 動かせばよいか」。
#   重ね表示(学習時の画像が半透明)の箱の輪郭がぴったり重なれば OK。
# q で終了。
# ONESHOT=1 で、ウィンドウを開かずに 1 回だけ測って数値を表示して終わる。
#
# 2026-10-17: カメラ位置が変わっていてπ0.5が上下を繰り返すだけになった。直した後も
# dx=-30, dy=+10 残っていたので作った。
#
# ■ カメラ権限の都合で「ターミナル.app」(またはカメラ許可済みのターミナル)から実行すること。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
NAME="${1:-so101_cube_v2}"
REF="outputs/cam_test/ref_top_${NAME}.jpg"
mkdir -p outputs/cam_test
if [ ! -f "$REF" ]; then
  VID=$(ls data/"$NAME"/videos/observation.images.top/*/*.mp4 | head -1)
  ffmpeg -hide_banner -loglevel error -y -i "$VID" -frames:v 1 "$REF"
fi

ONESHOT="${ONESHOT:-0}" python3 -u - "$REF" "$CAMERAS" <<'PY' 2>&1 | grep -viE "^objc\[|AVFFrameReceiver|AVFAudioReceiver"
import sys, re, os, cv2, numpy as np
oneshot = os.environ.get("ONESHOT") == "1"
ref_path, cams = sys.argv[1], sys.argv[2]
idx = int(re.search(r"top:\s*\{[^}]*index_or_path:\s*(\d+)", cams).group(1))
ref = cv2.resize(cv2.imread(ref_path), (640, 360))
cap = cv2.VideoCapture(idx)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
roi = (slice(150, 360), slice(0, 300))          # 箱がある左下。アームとキューブは動くので外す
win = cv2.createHanningWindow((300, 210), cv2.CV_32F)
def edges(img):
    # 箱まわりの輝度で比べる。照明が大きく違うと相関が下がり数値が不安定になる(10-18 朝は corr 0.1 前後)。
    # その場合は重ね画像 align_now.jpg と生画像 top_now_raw.jpg を目で確認すること
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[roi]
g_ref = edges(ref)
if not cap.isOpened():
    sys.exit(f"カメラ index {idx} を開けない(他のプロセスが使っていないか確認)")
print(f"top = index {idx}。ウィンドウが開く。q で終了。|dx|,|dy| <= 3px を目標に。")
fails = 0
while True:
    ok, frame = cap.read()
    if not ok:
        fails += 1
        if fails % 60 == 0:
            print(f"フレームが取れない({fails} 回連続)。ロボットのクライアント等がカメラを使っていないか確認")
        if fails > 300:
            sys.exit("中止: カメラからフレームが来ない")
        continue
    fails = 0
    frame = cv2.resize(frame, (640, 360))
    g = edges(frame)
    (dx, dy), resp = cv2.phaseCorrelate(g_ref, g, win)
    good = abs(dx) <= 3 and abs(dy) <= 3
    if oneshot:
        fails = getattr(sys.modules[__name__], "_n", 0) + 1; sys.modules[__name__]._n = fails
        if fails < 15:          # 露出が落ち着くまで数フレーム捨てる
            continue
        cv2.imwrite("outputs/cam_test/align_now.jpg", cv2.addWeighted(frame, 0.55, ref, 0.45, 0))
        cv2.imwrite("outputs/cam_test/top_now_raw.jpg", frame)
        print(f"dx={dx:+.1f}px dy={dy:+.1f}px corr={resp:.2f} -> {'OK' if good else 'ずれあり'}  (重ね画像: outputs/cam_test/align_now.jpg)")
        break
    view = cv2.addWeighted(frame, 0.55, ref, 0.45, 0)
    cv2.rectangle(view, (0, 150), (300, 359), (0, 255, 0) if good else (0, 0, 255), 2)
    hint = ("OK" if good else
            f"move image {'left' if dx > 0 else 'right'} {abs(dx):.0f}px, {'up' if dy > 0 else 'down'} {abs(dy):.0f}px")
    cv2.putText(view, f"dx={dx:+.1f} dy={dy:+.1f} corr={resp:.2f}  -> {hint}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if good else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("align top camera (q to quit)", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print(f"最終: dx={dx:+.1f}px dy={dy:+.1f}px corr={resp:.2f}")
        break
cap.release(); cv2.destroyAllWindows()
PY
