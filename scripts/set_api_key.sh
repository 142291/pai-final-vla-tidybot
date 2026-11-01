#!/usr/bin/env zsh
# Claude の API キーを config.env に追記する(画面にも履歴にも残さない)。
# 使い方: scripts/set_api_key.sh を実行し、表示に従ってキーを貼り付けて Enter。
# すでに ANTHROPIC_API_KEY の行があれば置き換える。
cd "$(dirname "$0")/.."
read -s "k?Claude の API キーを貼り付けて Enter(画面には出ません): "
echo
if [[ -z "$k" ]]; then echo "空だったので何もしない"; exit 1; fi
[[ "$k" == sk-ant-api* ]] || echo "注意: Console の API キーは sk-ant-api で始まる。違うキーかもしれない"
grep -v "^export ANTHROPIC_API_KEY=" config.env > config.env.tmp && mv config.env.tmp config.env
print -r -- "export ANTHROPIC_API_KEY=$k" >> config.env
unset k
echo "config.env に追記しました"
