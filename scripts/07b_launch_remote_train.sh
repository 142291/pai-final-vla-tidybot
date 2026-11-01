#!/usr/bin/env bash
# H200 で学習を起動し、実際にステップが進み始めたことを確認してから戻る。
#
# 使い方: scripts/07b_launch_remote_train.sh <セッション名> <リモートのスクリプトパス>
# 例:    scripts/07b_launch_remote_train.sh pi05 '~/ai-arm/run_pi05.sh'
#
# なぜこれが要るか:
#   2026-09-29、pi05 の学習を tmux に投げて 7 時間後の回収を仕掛けたが、
#   実際には 12 分後に gated repo エラーで落ちていた。起動を確認せずに
#   長時間の段取りを組むと、丸ごと無駄になる。
#   このスクリプトは「ステップが進んだ」か「落ちた」かを必ず判定する。
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION="${1:?tmux セッション名}"; SCRIPT="${2:?リモートのスクリプトパス}"
HOST="${HOST:-gpu-server}"
LOG="~/ai-arm/logs/${SESSION}.log"
TIMEOUT="${LAUNCH_TIMEOUT:-900}"   # 起動確認の上限(秒)。モデルのダウンロードを含むので長め

ssh -o ConnectTimeout=20 "$HOST" "
  cd ~/ai-arm && mkdir -p logs && rm -f ${LOG}
  tmux kill-session -t ${SESSION} 2>/dev/null || true
  tmux new-session -d -s ${SESSION} '${SCRIPT} > ${LOG} 2>&1; sleep 86400'
" >/dev/null
echo "起動した: tmux[$SESSION]  ログ: $LOG"
echo "ステップが進むか、落ちるかを最大 $((TIMEOUT/60)) 分まで監視します..."

START=$(date +%s)
while :; do
  sleep 20
  OUT=$(ssh -o ConnectTimeout=20 "$HOST" "
    if grep -qE 'Traceback|Error:|EXIT=[1-9]|error:' ${LOG} 2>/dev/null; then echo FAILED;
    elif tail -c 3000 ${LOG} 2>/dev/null | tr '\r' '\n' | grep -qE '[0-9]+/[0-9]+ \[' ; then echo RUNNING;
    else echo WAITING; fi" 2>/dev/null || echo WAITING)
  case "$OUT" in
    RUNNING)
      echo "✓ 学習が進行中であることを確認"
      ssh -o ConnectTimeout=20 "$HOST" "tail -c 300 ${LOG} | tr '\r' '\n' | tail -1" 2>/dev/null
      exit 0 ;;
    FAILED)
      echo "✗ 落ちました。ログ末尾:"
      ssh -o ConnectTimeout=20 "$HOST" "tail -c 1200 ${LOG} | tr '\r' '\n' | grep -v '^$' | tail -12" 2>/dev/null
      exit 1 ;;
  esac
  NOW=$(date +%s)
  if [ $((NOW-START)) -gt "$TIMEOUT" ]; then
    echo "✗ $((TIMEOUT/60)) 分経ってもステップが進みません。ログ末尾:"
    ssh -o ConnectTimeout=20 "$HOST" "tail -c 800 ${LOG} | tr '\r' '\n' | grep -v '^$' | tail -8" 2>/dev/null
    exit 1
  fi
  printf "."
done
