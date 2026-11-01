#!/usr/bin/env bash
# データセット録画(ローカル保存のみ。Hugging Face へのアップロードはしない)。
#
# 使い方: scripts/06_record.sh <データセット名> <エピソード数> "<タスク説明(英語)>" [追加引数...]
# 例:    scripts/06_record.sh so101_cube 50 "Pick up the red cube and put it in the box"
#
# 保存先: data/<データセット名>/   (repo_id は local/<データセット名>)
# 録画中のキー: n または → 次へ / r または ← 撮り直し / q または ESC 終了
#
# 環境変数で調整:
#   EPISODE_SEC   1 エピソードの長さ(既定 30 秒)
#   RESET_SEC     エピソード間のリセット時間(既定 10 秒)
#   NO_CAM=1      カメラなしで録画(動作確認用。本番では使わない)
#   NO_DISPLAY=1  rerun の映像表示を切る(CPU 負荷が下がる。録画自体には影響しない)
#   VCODEC=...    動画コーデック(既定 h264_videotoolbox=Apple Silicon のハードウェア。他に h264, libsvtav1)
#   ENC_THREADS=N エンコードのスレッド数(既定 4)
#
# 途中から足すとき(タスクを増やす / エピソードを追加する):
#   scripts/06_record.sh so101_multi 50 "Pick up the blue cube" --resume=true
#   ※ --resume=true のとき <エピソード数> は「追加する数」
#
# カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="${1:?データセット名}"; NUM="${2:?エピソード数}"; TASK="${3:?タスク説明}"
shift 3
DATA_ROOT="$ROOT/data/$NAME"
ARGS=(
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false
  --teleop.type=so101_leader  --teleop.port="$LEADER_PORT"  --teleop.id="$LEADER_ID"  --teleop.use_degrees=false
  --dataset.repo_id="local/$NAME"
  --dataset.root="$DATA_ROOT"
  --dataset.num_episodes="$NUM"
  --dataset.single_task="$TASK"
  --dataset.episode_time_s="${EPISODE_SEC:-30}"
  --dataset.reset_time_s="${RESET_SEC:-10}"
  --dataset.push_to_hub=false
  --dataset.streaming_encoding=true
  --dataset.encoder_threads="${ENC_THREADS:-4}"
  --dataset.rgb_encoder.vcodec="${VCODEC:-h264_videotoolbox}"
  --dataset.rgb_encoder.crf=23
)
if [ "${NO_CAM:-0}" != "1" ]; then
  ARGS+=( --robot.cameras="$CAMERAS" )
  if [ "${NO_DISPLAY:-0}" = "1" ]; then ARGS+=( --display_data=false ); else ARGS+=( --display_data=true ); fi
fi
echo "録画: $NAME / $NUM エピソード / \"$TASK\""
echo "保存先: $DATA_ROOT"
echo "キー: n=次へ  r=撮り直し  q=終了"
lerobot-record "${ARGS[@]}" "$@"
