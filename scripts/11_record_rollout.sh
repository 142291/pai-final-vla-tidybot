#!/usr/bin/env bash
# ポリシーで自律動作させながら、その様子を録画する(診断用)。
#
# 使い方: scripts/11_record_rollout.sh "<指示(英語)>" [エピソード数] [秒数]
#
# 通常の実行(10_language.sh)は何も残らないので、失敗の原因を後から調べられない。
# これで録画しておけば、グリッパーがどこへ行ったか、キューブがどこにあったかを
# フレーム単位で測れる。
#
# 環境変数:
#   POLICY  使うモデル(既定は SmolVLA の最終チェックポイント)
#   NAME    データセット名(既定 diag)。rollout_ が無ければ自動で付く
#
# ■ カメラ権限の都合で「ターミナル.app」から実行すること。
# ■ エピソードの終了は n(または →)、やり直しは r、終了は q。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
TASK="${1:?指示を英語で}"; NUM="${2:-1}"; SEC="${3:-30}"
NAME="${NAME:-diag}"
# lerobot はロールアウト用データセット名に rollout_ 接頭辞を要求する
case "$NAME" in rollout_*) ;; *) NAME="rollout_$NAME" ;; esac
POLICY="${POLICY:-outputs/train/smolvla_so101_cube_v2/checkpoints/last/pretrained_model}"
DATA_ROOT="$ROOT/data/$NAME"

[ -d "$POLICY" ] || { echo "モデルが無い: $POLICY" >&2; exit 1; }

echo "指示    : $TASK"
echo "モデル  : $POLICY"
echo "保存先  : $DATA_ROOT"
echo "キー    : n=次へ  r=やり直し  q=終了"
echo

ARGS=(
  --strategy.type=episodic
  --policy.path="$POLICY"
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false
  --robot.cameras="$CAMERAS"
  --task="$TASK"
  --policy.device="${POLICY_DEVICE:-mps}"
  --fps="${FPS:-30}"
  --inference.type="${INFER:-rtc}"
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
  --dataset.repo_id="local/$NAME"
  --dataset.root="$DATA_ROOT"
  --dataset.num_episodes="$NUM"
  --dataset.single_task="$TASK"
  --dataset.episode_time_s="$SEC"
  --dataset.reset_time_s="${RESET_SEC:-10}"
  --dataset.push_to_hub=false
  --dataset.streaming_encoding=true
  --dataset.encoder_threads="${ENC_THREADS:-4}"
  --dataset.rgb_encoder.vcodec="${VCODEC:-h264_videotoolbox}"
  --dataset.rgb_encoder.crf=23
  --display_data=false
)
[ -n "${RESUME:-}" ] && ARGS+=(--resume=true)

lerobot-rollout "${ARGS[@]}"
