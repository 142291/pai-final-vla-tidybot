#!/usr/bin/env bash
# 指示(日本語可)を受けて、段取り → π0.5 → 成否判定 → やり直し/人の介入 を通しで行う。
#
# 使い方: scripts/17_orchestrate.sh "片付けて"
#         HUMAN=0 scripts/17_orchestrate.sh "紫以外を箱に入れて"   # リーダーアームを使わない
#
# ■ 起動前にフォロワーをホーム姿勢に(その姿勢を「ホーム」として覚える)。
#   人が手伝うときに使うので、リーダーアームもホーム姿勢に置いておく。
# ■ カメラ権限の都合で「ターミナル.app」かアプリのターミナルから実行すること。
# ■ 段取りに Claude を使うには config.env に `export ANTHROPIC_API_KEY=...` を書く
#   (無ければキーワードのルールで段取りする)。
# ■ 操作は画面(orchestrator ウィンドウ)をクリックしてから:
#     h 人が操作する/やめる  s 成功にする  f 失敗にする  q 終了
# 仕組みと記録先は scripts/16_orchestrator.py の冒頭を参照。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if [ "${HUMAN:-1}" = "1" ]; then
  export LEADER_PORT LEADER_ID
else
  unset LEADER_PORT LEADER_ID
fi
# 既定は「止まって考える」方式(先頭 15 ステップだけ実行して止まり、止まった後の画像で次を予測)。
# 2026-10-24 の評価で通常方式は掴めず、この方式で 2/2 成功した。SYNC_K=0 で通常方式。
export SYNC_K="${SYNC_K:-15}"
export ATTEMPT_SEC="${ATTEMPT_SEC:-60}"
# ROUND=round1 のように付けると、19_autonomy.py でラウンドごとに集計できる
export ROUND="${ROUND:-}" RUN="${RUN:-pi05_v3full}" CKPT="${CKPT:-020000}"
# 送った画像・関節角・指令を記録する(失敗の仕方を後から確かめるため。events.jsonl の時刻と突き合わせる)
export TRIAL_REC_DIR="${TRIAL_REC_DIR:-outputs/runs/rec_$(date +%Y%m%d_%H%M%S)}"
export CLIENT_PY=scripts/16_orchestrator.py
exec scripts/12_language_remote.sh "${1:-片付けて}"
