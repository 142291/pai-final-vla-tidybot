#!/usr/bin/env bash
# H200 で推論し、Mac のアームを動かす(π0.5 用)。
#
# 使い方: scripts/12_language_remote.sh "<指示(英語)>"
#
# なぜ遠隔か:
#   π0.5 は 4.14B あり、Mac では forward に 3〜4 秒かかって実時間制御に乗らない。
#   H200 での実測は 463ms。往復遅延 110ms + 画像転送 691ms を足しても
#   合計 1,264ms で、チャンク持続 1,667ms(50step @30Hz)に収まる。
#
# 2026-10-17 の改修(FAST=1、既定):
#   従来の robot_client は観測(生画像 1.38MB)を制御ループの中で同期送信しており、
#   送っている間ロボットに指令が出なかった(10-01 のログで 1 周 3.5〜4.3 秒)。
#   scripts/remote/fast_client.py は送信を裏スレッドに移し、画像を元の解像度のまま
#   JPEG にする。実測: 1,351KB → 140KB、送信 p50 957ms → 174ms(bench_transport.py)。
#   サーバーは JPEG を復元できる scripts/remote/fast_server.py(H200 の tmux pserver_fast, 8124)。
#   10 秒ごとに send / age / loop の遅延が表示される。送信は SEND_HZ(既定 4)回/秒まで。
#   Wi-Fi は往復 20〜286ms とぶれが大きい(10-17 実測)。可能なら有線 LAN で。
#
# 環境変数:
#   FAST   1 で改修版クライアントと 8124 番のサーバーを使う(既定 1)。0 で従来版(8123)
#   RUN    学習の出力名(既定 pi05_v3full = S1〜S3 の 144 本)。pi05_v3s1 = S1 のみ、pi05_so101_cube = 旧 v2
#   CKPT   チェックポイント(既定 020000 = pi05_v3full の最終)
#   PORT   サーバーのポート(既定 FAST=1 なら 8124、0 なら 8123)
#   THRESH 再計画のしきい値(既定 0.5)。キューがこの割合を切ると新チャンクを要求する。
#          0.5 だと 25 アクション = 0.83 秒ごとで、往復遅延 770ms とほぼ同周期。
#          古い観測にもとづく修正が入り続けて振動する疑いがあるときは 0.3 に下げる。
#   AGG    チャンクの混合方法(既定 weighted_average = 旧0.3:新0.7)。
#          conservative(旧0.7:新0.3) にすると急な方針転換が減る。
#          他に latest_only, average。
#
# ■ カメラ権限の都合で「ターミナル.app」から実行すること。
# ■ 事前に H200 で policy_server が動いていること:
#     ssh gpu-server 'tmux ls | grep pserver'
#
# ■ カメラ名について:
#   robot_client は rename_map を送れない(RemotePolicyConfig を 5 引数で構築しており
#   rename_map が空のまま、サーバー側がチェックポイントの設定を上書きしてしまう)。
#   そこでカメラ名そのものを π0.5 が期待する名前にしている。録画側には影響しない。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
TASK="${1:?指示を英語で}"
FAST="${FAST:-1}"
if [ -n "${CLIENT_PY:-}" ]; then   # 17_orchestrate.sh が 16_orchestrator.py を指定する
  PORT="${PORT:-8124}"; CLIENT=(python "$CLIENT_PY")
elif [ "$FAST" = "1" ]; then
  PORT="${PORT:-8124}"; CLIENT=(python scripts/remote/fast_client.py)
else
  PORT="${PORT:-8123}"; CLIENT=(python -m lerobot.async_inference.robot_client)
fi
RUN="${RUN:-pi05_v3full}"
CKPT="${CKPT:-020000}"
REMOTE="${REMOTE_HOST:-gpu-server}"
MODEL="/root/ai-arm/outputs/train/$RUN/checkpoints/$CKPT/pretrained_model"

# 方策の種類(POLICY、既定 pi05)。比較実験用に smolvla / act も動かせる(2026-10-25)
POLICY="${POLICY:-pi05}"
# カメラ名を方策が学習時に見た名前に差し替える(rename_map が送れないため)
# pi05: base_0_rgb / left_wrist_0_rgb、smolvla: camera1 / camera2、act: top / wrist のまま
# (macOS の sed は \b 非対応なので python で置換する)
CAMS=$(python3 -c "
import re, sys
s, pol = sys.argv[1], sys.argv[2]
names = {'pi05': ('base_0_rgb', 'left_wrist_0_rgb'), 'smolvla': ('camera1', 'camera2')}.get(pol)
if names:
    s = re.sub(r'\btop\b', names[0], s)
    s = re.sub(r'\bwrist\b', names[1], s)
print(s)" "$CAMERAS" "$POLICY")

echo "指示    : $TASK"
echo "モデル  : $REMOTE:$RUN/$CKPT ($POLICY)"
echo "カメラ  : $CAMS"
echo "再計画  : しきい値 ${THRESH:-0.5} / 混合 ${AGG:-weighted_average}"
echo "経路    : FAST=$FAST (ポート $PORT)"
echo "停止    : Ctrl+C (非常時はフォロワーの電源 OFF)"
echo

# SSH トンネル(サーバーは 127.0.0.1 でしか待ち受けていない)
CTL=/tmp/pi05_tunnel_$$
cleanup() { ssh -S "$CTL" -O exit "$REMOTE" 2>/dev/null || true; }
trap cleanup EXIT
# テザリング等で一瞬切れることがあるので 5 回まで再試行する(2026-10-17 に "Network is unreachable" で 2 回失敗)
for i in 1 2 3 4 5; do
  if ssh -o ConnectTimeout=10 -M -S "$CTL" -fN -L "$PORT:127.0.0.1:$PORT" "$REMOTE"; then break; fi
  [ "$i" = 5 ] && { echo "H200 に繋がらない。回線を確認すること"; exit 1; }
  echo "接続に失敗、3 秒後に再試行 ($i/5)"; sleep 3
done
echo "トンネル確立: localhost:$PORT -> $REMOTE:$PORT"

"${CLIENT[@]}" \
  --server_address="127.0.0.1:$PORT" \
  --policy_type="$POLICY" \
  --pretrained_name_or_path="$MODEL" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold="${THRESH:-0.5}" \
  --aggregate_fn_name="${AGG:-weighted_average}" \
  --task="$TASK" \
  --fps=30 \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false \
  --robot.cameras="$CAMS"
