#!/usr/bin/env bash
# 学習済みポリシーで自律動作(録画なし)。
# 使い方: scripts/08_rollout.sh <policy path or HF repo> "<タスク説明>" [秒数]
#
# 環境変数:
#   POLICY_DEVICE=cpu|mps   推論デバイス(既定 mps)
#   ROLLOUT_FPS=N           制御周波数(既定 30。学習データと揃えること)
#   N_ACTION_STEPS=N        1回の予測を何ステップ実行するか(既定は学習時の設定)
#   INFER=rtc               推論を裏スレッドに逃がす(SmolVLA 等のみ。ACT は非対応)
#                           チャンク境界でループが止まらなくなる
#   RENAME_MAP='{...}'      データセットのカメラ名をポリシー側の名前に合わせる
#                           SmolVLA は必須(top/wrist -> camera1/camera2)
#
# n_action_steps について:
#   ACT は chunk_size=100 で 100 ステップ分をまとめて予測する。既定では
#   その 100 ステップ(30Hz なら 3.3 秒)を完全に開ループで実行するので、
#   途中でカメラを見て軌道を修正できない。20 程度に下げると 0.66 秒ごとに
#   再計画して閉ループになるが、予測のたびに約 100ms ループが止まる
#   (MPS での forward 1 回分)ので、その頻度も上がる。
#
# カメラ権限の都合で「ターミナル.app」から実行すること。
set -euo pipefail
source "$(dirname "$0")/env.sh"
POLICY="${1:?policy path}"; TASK="${2:?task}"; DUR="${3:-60}"

ARGS=(
  --strategy.type=base
  --policy.path="$POLICY"
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" --robot.use_degrees=false
  --robot.cameras="$CAMERAS"
  --task="$TASK"
  --policy.device="${POLICY_DEVICE:-mps}"
  --fps="${ROLLOUT_FPS:-30}"
  --duration="$DUR"
)
[ -n "${N_ACTION_STEPS:-}" ] && ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")
[ -n "${INFER:-}" ] && ARGS+=(--inference.type="$INFER")
[ -n "${RENAME_MAP:-}" ] && ARGS+=(--rename_map="$RENAME_MAP")

lerobot-rollout "${ARGS[@]}"
