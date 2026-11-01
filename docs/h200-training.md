# H200 VM での学習手順

Mac(MPS)より 8.5 倍速い。本番の 10 万ステップは Mac で 22 時間、H200 で約 2.7 時間。

## 接続先

```
Host gpu-server   (~/.ssh/config に登録済み)
HostName <GPU サーバーの IP>
User root
```

GPU: NVIDIA H200 (143GB VRAM) / ドライバ 575.57.08 / Python 3.12.3 / ディスク空き 2.5TB

> [!IMPORTANT]
> **VM は使うときだけ起動する。** 停止中は SSH がタイムアウトする(2026-09-27 に遭遇)。
> FPT Cloud のコンソールで起動してから接続すること。
>
> **自動停止は 2026-09-28 に無効化した。** 元は `/etc/cron.d/auto-shutdown` →
> `/usr/local/bin/auto-shutdown.sh` が毎日 JST 0 時に `who | grep -q pts/` を見て、
> SSH セッションが無ければ `shutdown -h now` していた。
> 現在はその cron 行をコメントアウト済み(バックアップ: VM の
> `/root/auto-shutdown.cron.orig`)。
>
> **つまり VM は明示的に止めるまで課金が続く。** 使い終わったら止めること:
>
> ```bash
> ssh gpu-server 'shutdown -h now'   # または FPT コンソールで停止
> ```
>
> 自動停止を戻すとき:
>
> ```bash
> ssh gpu-server \
>   "sed -i 's|^#0 0 \* \* \*|0 0 * * *|' /etc/cron.d/auto-shutdown && cat /etc/cron.d/auto-shutdown"
> ```

## 初回セットアップ(済み・2026-09-27)

```bash
ssh gpu-server
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/ai-arm && cd ~/ai-arm
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "lerobot[training]"
# 既定の torch は CUDA 13.0 版でドライバ 12.9 と合わない。12.8 版に入れ替える
UV_HTTP_TIMEOUT=300 uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -c "import torch; print(torch.cuda.is_available())"   # True になること
```

> [!WARNING]
> `uv pip install torch` の既定は `+cu130` でドライバ 12.9 では動かない(`CUDA 利用可: False`)。
> 必ず `--index-url .../cu128` を付ける。NCCL のダウンロードがタイムアウトしやすいので `UV_HTTP_TIMEOUT=300` を付ける。

## データ転送

```bash
# Mac 側。macOS 標準の rsync は --info=progress2 非対応なので付けない
ssh gpu-server 'mkdir -p ~/ai-arm/data/so101_cube'
rsync -az data/so101_cube/ gpu-server:~/ai-arm/data/so101_cube/
```

415MB(12 エピソード)で約 7 分半。

## 学習

```bash
ssh gpu-server
export PATH="$HOME/.local/bin:$PATH"; cd ~/ai-arm && source .venv/bin/activate
lerobot-train \
  --dataset.repo_id=local/so101_cube \
  --dataset.root=$HOME/ai-arm/data/so101_cube \
  --policy.type=act \
  --output_dir=$HOME/ai-arm/outputs/train/act_so101_cube \
  --job_name=act_so101_cube \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --steps=100000 --save_freq=10000 --batch_size=8 --num_workers=4
```

> [!TIP]
> **tmux の中で回すこと。** 自動停止は無効化したので 0 時に落ちる心配はないが、
> SSH が切れたときに学習を巻き込まないために tmux は依然として必要。
> (`nohup ... &` でも切断自体には耐えるが、ログの追いかけと停止が面倒になる)

```bash
ssh gpu-server
tmux new -s train        # 既にあるなら tmux attach -t train
export PATH="$HOME/.local/bin:$PATH"; cd ~/ai-arm && source .venv/bin/activate
lerobot-train ...        # 上のコマンド
# Ctrl-b d で detach。そのまま exit で SSH を切ってよい
```

進捗の確認と後片付け:

```bash
ssh gpu-server 'tmux ls; who'          # セッションが残っているか
ssh gpu-server -t 'tmux attach -t train'   # ログを見る
ssh gpu-server 'tmux kill-session -t train' # 学習が終わったら消す
```

> [!IMPORTANT]
> **学習が終わったら VM を止めること。** 自動停止を無効化したので、放置すると
> H200 の課金が続く。`ssh gpu-server 'shutdown -h now'`。

### 実測値(12 エピソード / 6982 フレーム / ACT 52M パラメータ)

| バッチ | 速度 | 備考 |
| --- | --- | --- |
| 8 | **10.42 step/s** | 最速。VRAM 3GB しか使わない |
| 32 | 2.82 step/s | |
| 64 | 1.59 step/s | サンプル/秒は増えるが総時間では不利 |

Mac(MPS)は batch 8 で 1.23 step/s。

## モデルの回収

```bash
# Mac 側
rsync -az gpu-server:'~/ai-arm/outputs/train/act_so101_cube/checkpoints/last/pretrained_model/' \
  outputs/train/act_so101_cube/checkpoints/last/pretrained_model/
```

207MB で約 1 分 40 秒。`training_state/` は実行に不要なので転送しない(591MB → 207MB)。

## 実行

```bash
scripts/08_rollout.sh outputs/train/act_so101_cube/checkpoints/last/pretrained_model \
  "Pick up the red cube and place it in the bowl" 30
```

Mac(MPS)での推論速度は 1078 回/秒。30Hz に対して十分。

## SmolVLA の学習(2026-09-29)

ACT と SmolVLA を **H200 上で並列実行**した。合計 VRAM 23.6GB で収まる。

```bash
lerobot-train \
  --dataset.repo_id=local/so101_cube_v2 \
  --dataset.root=$HOME/ai-arm/data/so101_cube_v2 \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --output_dir=$HOME/ai-arm/outputs/train/smolvla_so101_cube_v2 \
  --job_name=smolvla_so101_cube_v2 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --steps=20000 --save_freq=2000 --batch_size=64 --num_workers=4
```

> [!WARNING]
> **`uv pip install 'lerobot[smolvla]'` を使わないこと。** torch が cu130 に
> 差し替わってドライバ 12.9 で動かなくなる(このファイル冒頭の罠と同じ)。
> 必要なのは 2 つだけなので、個別に入れる。transformers は torch に依存しない:
>
> ```bash
> UV_HTTP_TIMEOUT=300 uv pip install "transformers<5.6.0,>=5.4.0" "num2words>=0.5.14,<0.6.0"
> ```

> [!IMPORTANT]
> **`--rename_map` は学習時と実行時の両方で要る。** smolvla_base は
> `camera1/camera2/camera3` を期待し、こちらのデータは `top/wrist`。
> カメラ数の検証は「どちらかが部分集合なら通る」ので 2 台でも問題ない
> (camera3 は欠損扱いでスキップされる)。
> 実行時に渡し忘れると、rollout が `preprocessor_overrides` で保存済みの
> 設定を空で上書きしてしまい、特徴量の不一致で落ちる。

### 実測(50 エピソード / 19,475 フレーム、ACT と並列実行)

| | ステップ | 速度 | 所要 | サイズ | 最終 loss |
| --- | --- | --- | --- | --- | --- |
| ACT | 100,000 | 7.34 step/s | 3:47:07 | 208MB | 0.059 |
| SmolVLA | 20,000 | 1.33 step/s | 4:11:05 | 880MB | 0.017 |

単独なら ACT は 10.4 step/s 出る。上記は並列実行での値。

## Mac 側の推論速度(2026-09-29 実測)

`scripts/10_language.sh` の設定はこの実測にもとづく。

| | forward 1 回 | 判定 |
| --- | --- | --- |
| SmolVLA / MPS | **742ms** (733-744、ブレなし) | チャンク持続 1667ms に間に合う |
| SmolVLA / CPU | 約 7,000ms | 論外 |
| ACT / MPS | 約 100ms | 間に合う |

SmolVLA の forward はちょうど `n_action_steps=50` ごと。初回のみ 1,220ms
(MPS のカーネルコンパイル)。チャンク内の通常ティックは 5.0ms。

> [!IMPORTANT]
> **SmolVLA は `--inference.type=rtc` が事実上必須。**
> 同期実行だと 1.67 秒ごとに 0.74 秒フリーズする。RTC なら裏スレッドで
> 次チャンクを計算するので制御ループが止まらない(余裕 925ms)。
> ACT は RTC 非対応なので、こちらは `N_ACTION_STEPS` で調整する。

## π0.5 の学習(2026-09-29、未完了)

SmolVLA が実機で 0/10 だったため π0.5 へ乗り換えを試みた。論文の実測で
SO-101 の Pen Transfer 相当が SmolVLA 70% に対して π0.5 95%、失敗からの
回復率も 3.2% 対 30.8% と大きく上回る。

> [!CAUTION]
> **π0.5 は HuggingFace の gated リポジトリへの認証が必須。**
> トークナイザに `google/paligemma-3b-pt-224` を使うが、これは Google の
> ライセンス同意が要る。認証が無いと、モデル本体(16GB)を落とし切った後、
> プロセッサ構築の段階で落ちる:
>
> ```
> ValueError: Failed to instantiate processor step 'tokenizer_processor'
> Error: You are trying to access a gated repo.
> Access to model google/paligemma-3b-pt-224 is restricted.
> ```
>
> `lerobot/pi05_base` 自体にトークナイザは同梱されていない(config.json と
> model.safetensors のみ)ので、回避できない。

### 事前に必要な手順

1. https://huggingface.co/google/paligemma-3b-pt-224 を開き、ライセンスに同意する
2. https://huggingface.co/settings/tokens で read 権限のトークンを作る
3. VM に設定する:

```bash
ssh gpu-server
export PATH="$HOME/.local/bin:$PATH"; cd ~/ai-arm && source .venv/bin/activate
hf auth login            # トークンを貼る。~/.cache/huggingface/token に保存される
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print('OK')"
```

### 学習コマンド

```bash
lerobot-train \
  --dataset.repo_id=local/so101_cube_v2 \
  --dataset.root=$HOME/ai-arm/data/so101_cube_v2 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --rename_map='{"observation.images.top": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --dataset.image_transforms.enable=true \
  --output_dir=$HOME/ai-arm/outputs/train/pi05_so101_cube \
  --job_name=pi05_so101_cube \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --steps=30000 --save_freq=2000 --batch_size=64 --num_workers=4
```

`pi05_base` が期待するカメラ名は `base_0_rgb` / `left_wrist_0_rgb` /
`right_wrist_0_rgb`。3 台目は欠損扱いで通る(SmolVLA の camera3 と同じ)。
依存は `transformers` と `scipy` だけで、`lerobot[pi]` を入れる必要はない
(torch を cu130 に差し替えられる罠を避ける)。

> [!IMPORTANT]
> **長時間の段取りを組む前に `scripts/07b_launch_remote_train.sh` で
> 起動を確認すること。** 2026-09-29 はこれを怠り、12 分で落ちた学習の
> 完了を 7 時間待った。

### Mac 側の推論

π0.5 は 4.14B、fp32 で VRAM 16.84GB。**Mac では 3〜4 秒かかり実時間制御に
乗らない。** H200 実測は forward 463ms(初回 813ms)で、チャンク持続
1,667ms に対して 1,203ms の余裕がある。

遠隔推論は lerobot に同梱されている:

```
H200  python -m lerobot.async_inference.policy_server
Mac   python -m lerobot.async_inference.robot_client
```

実測した往復遅延は 110ms、上り帯域は約 2MB/s。`transport/utils.py` は
pickle で生テンソルを送るため画像圧縮が無く、生 640x360 x2 台で 1.38MB =
691ms かかる。合計 1,264ms で予算内だが余裕が薄い。クライアント側で
256x256 に縮小してから送れば 197ms になり、合計 770ms で安全域に入る。

## 遠隔推論の遅延対策と切り分け(2026-10-17)

### 原因: 観測の送信が制御ループを止めていた

lerobot の `robot_client` は観測を**制御ループと同じスレッドで同期送信**する。生画像
640x360 x 2 台 = 1.38MB を pickle で送るので、送っている間ロボットに指令が出ない。
10-01 のログでは制御ループ 1 周が 3.5〜4.3 秒で、「キューブ手前でウロウロ」の一因だった。

### 対策(`scripts/remote/`、`12_language_remote.sh` は既定で FAST=1)

| | 内容 |
| --- | --- |
| `fast_client.py` | 送信を裏スレッドへ(1 枠の郵便受け、古い観測は捨てる)。JPEG q90 に圧縮(**元の解像度のまま**。縮小すると π0.5 の前処理と幾何が変わる)。送信は `SEND_HZ`(既定 4)回/秒まで。10 秒ごとに send / age / loop を表示 |
| `fast_server.py` | JPEG を復元(約 5ms)。**同じモデルなら読み込み直さない**(標準は接続のたびに from_pretrained、π0.5 で 1.5〜2 分) |
| `bench_transport.py` | ロボット無しで送信時間を測る |

H200 では tmux `pserver_fast`(127.0.0.1:8124)で動かす。旧 `pserver`(8123)は停止済み。

### 実測

| | 対策前 | 対策後 |
| --- | --- | --- |
| 観測 1 回の大きさ | 1,351KB | 90〜140KB(JPEG 復元誤差は平均 2.2/255) |
| 制御ループ 1 周 | 3,500〜4,300ms | **2〜4ms** |
| 送信 p50(家の Wi-Fi) | 690〜957ms | 119〜220ms |
| age(観測→その観測による行動が届く)p50 / p95 | — | **0.85〜1.0 秒 / 1.0〜1.4 秒** |
| 接続から動き出すまで(2 回目以降) | 1.5〜2 分 | 約 1 秒 |

内訳は送信 ~0.15〜0.25 秒 + 推論待ち 0〜0.5 秒 + 推論 ~0.49 秒(H200 は vLLM と同居)+ 返送 ~0.06 秒。

> [!WARNING]
> **iPhone のテザリングでは使えない。** 往復 20〜286ms とぶれ、数十秒ごとに数秒止まる
> (age が最大 10.9 秒)。`ssh: Network is unreachable` も 2 回出た。家の Wi-Fi か有線で動かすこと。
> 送信を無制限(6 回/秒)にすると Wi-Fi の上りを自分で詰まらせる(送信 145ms → 350〜460ms)。

### 遅延を消しても掴めない — 切り分け

| 容疑 | 結果 |
| --- | --- |
| 遅延 | 白。age 0.9 秒前後で安定しても「近くまで行って僅かにずれる」 |
| カメラ位置 | **黒 → 修正**。俯瞰カメラが動いていて床とアームの土台を写していた(上下を繰り返すだけ)。直した後も dx=-30, dy=+10px → `05f_align_camera.sh` で dx=+3.8, dy=-3.0px まで合わせた |
| キャリブレーション | 白。EEPROM・キャッシュ・`calib/`(09-27)が一致 |
| 照明 | △。輝度 -2.4σ、青 -3.5σ(夜、暗い) |
| データ | 本命。77 本、照明の幅が狭い。データ v3(照明 3 条件 × 9 区画 × 3 色)で対処予定 |

> [!IMPORTANT]
> **カメラを使うチェック(05b / 05d / 05e / 05f)をロボット実行中に走らせない。** 同じカメラを
> 2 プロセスで開くと `frame width=480` や `latest frame is too old` が出て、ロボットに古い画像が渡る。
