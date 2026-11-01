#!/usr/bin/env bash
# ACT ポリシーを学習(ローカルのデータセットを使う)。
# 使い方: scripts/07_train_act.sh <データセット名> [追加引数...]
#
# Mac(MPS)だと数時間かかる。速くするには:
#   H200 VM で:  scripts/07_train_act.sh so101_cube --policy.device=cuda
#   HF Jobs で:  scripts/07_train_act.sh so101_cube --job.target=a10g-small   (要 hf auth login)
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="${1:?データセット名}"; shift
DATA_ROOT="$ROOT/data/$NAME"
[ -d "$DATA_ROOT" ] || { echo "データセットがありません: $DATA_ROOT" >&2; exit 1; }
lerobot-train \
  --dataset.repo_id="local/$NAME" \
  --dataset.root="$DATA_ROOT" \
  --policy.type=act \
  --output_dir="$ROOT/outputs/train/act_$NAME" \
  --job_name="act_$NAME" \
  --policy.device="${TRAIN_DEVICE:-mps}" \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  "$@"
