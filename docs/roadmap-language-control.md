# ロードマップ: 自然言語で SO-101 を操作するデモ

最終形: 「赤いキューブを箱に入れて」と打つ(or 話す)と、SO-101 が自律で実行する。
方式は **SmolVLA の微調整**。事前学習データの大半が SO-100/101 のコミュニティデータなので、この機体が一番相性がよい。

現在地(2026-09-22): 両アーム組立・キャリブレーション・テレオペ完了。データ 0 件、学習済みモデルなし。

---

## Phase 0 — 設置を確定する(届いた当日、半日)

**ここで決めた配置は最後まで変えない。** カメラとフォロワーの位置関係が変わると、学習済みモデルは動かなくなる。

1. 山善の台にフォロワーをクランプ(奥の縁)。リーダーは別の場所にクランプ。
2. カメラアームを**同じ台**にクランプし、C270nd を俯瞰位置に。InnoMaker は手首マウント(公式 STL、32×32mm 用)ができるまで保留、または 2 台目の固定カメラとして使う。
3. USB ハブに基板 2 枚 + カメラを接続。台の重しと滑り止め、ケーブルの固定。
4. カメラ検出(**自分のターミナルから**。macOS の権限で Claude のプロセスからは不可):
   ```bash
   scripts/05_find_cameras.sh
   ```
   出た index を `config.env` の `CAMERAS` に書く。2 台なら `{ top: {...}, wrist: {...} }`。
5. 映像つきテレオペで、画角に作業範囲が収まっているか確認:
   ```bash
   scripts/04_teleop.sh --cam
   ```
   **判定基準: 画面だけ見て自分がそのタスクをこなせるか。** できないならカメラ位置を直す(いま直す。あとでは直せない)。

## Phase 1 — 最初のデータ(2〜3時間)

タスクは 1 つだけ。対象は木製キューブ、置き場所は 5 か所程度に限定。SO-101 の到達範囲は 40cm 台なので台の上に収める。

```bash
hf auth login                      # 初回のみ
hf auth whoami                     # 結果を config.env の HF_USER に
scripts/06_record.sh so101_cube_in_box 50 "Pick up the red cube and put it in the box"
```

- 1 エピソード 10〜20 秒、リセット込みで 30 秒 → 50 回で 30 分程度。
- 毎回同じ手順・同じ速さで動かす。一貫性が回数より効く。
- キー: `n`/→ 次へ、`r`/← 撮り直し、`q`/ESC 終了。

## Phase 2 — ACT で自律動作(半日)★最初のマイルストーン

言語はまだ使わない。「勝手に動く」を先に達成して、配置とデータが妥当かを検証する。

```bash
# H200 VM 上で(ローカル MPS だと数時間)
scripts/07_train_act.sh so101_cube_in_box --policy.device=cuda
# または HF Jobs
scripts/07_train_act.sh so101_cube_in_box --job.target=a10g-small
```

```bash
scripts/08_rollout.sh outputs/train/act_so101_cube_in_box/checkpoints/last/pretrained_model \
  "Pick up the red cube and put it in the box" 60
```

成功率 7 割程度が出れば Phase 3 へ。出ないなら原因はほぼデータ側(カメラ画角、実演のばらつき)。ここで直しておく。

## Phase 3 — 複数タスクのデータ(2〜3日)

**言語で使い分けるには、区別できるタスクが複数要る。** 3〜5 個、各 50 エピソード。

同じ `repo_id` に `--resume=true` で足し、`--dataset.single_task` だけ変える
(データセットは複数タスク文字列を保持できる: `LeRobotDatasetMetadata.get_task_index` / `total_tasks`)。

```bash
scripts/06_record.sh so101_multi 50 "Pick up the red cube and put it in the box"
scripts/06_record.sh so101_multi 50 "Pick up the blue cube and put it in the box" --resume=true --dataset.root=~/.cache/huggingface/lerobot/$HF_USER/so101_multi
scripts/06_record.sh so101_multi 50 "Hand me the tissue"                          --resume=true --dataset.root=...
```

守ること:

- **カメラは絶対に動かさない。** 全タスクで同じ画角。
- タスク文は英語、動詞と対象を明確に分ける。推論時もこの文字列と揃える。
- 視覚的に区別できるタスクにする(色違い、物体違い)。似すぎていると言語が効かない。
- `--resume=true` のとき `num_episodes` は**追加分**の数。`--dataset.root` の指定が必須。

## Phase 4 — SmolVLA を微調整(半日)★言語が通る

```bash
uv pip install 'lerobot[smolvla]'   # transformers, num2words が未導入(2026-09-22 時点)
```

```bash
lerobot-train \
  --dataset.repo_id=$HF_USER/so101_multi \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --output_dir=outputs/train/smolvla_so101 \
  --job_name=smolvla_so101 \
  --policy.device=cuda \
  --policy.repo_id=$HF_USER/smolvla_so101
```

推論は GPU 推奨。Mac(MPS)だと遅いので、重ければ LeRobot の非同期推論で H200 を推論サーバーにする。

```bash
lerobot-rollout --strategy.type=base --policy.path=$HF_USER/smolvla_so101 \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \
  --robot.cameras="$CAMERAS" --task="Hand me the tissue" --duration=60
```

**この時点でデモは成立する。** `--task=` に入れた文が指示そのもの。

## Phase 5 — デモの体裁(半日、任意)

- テキスト入力 → `lerobot-rollout` を起動する薄いラッパー。
- 音声にするなら Whisper を前段に。
- 日本語で指示したいなら、LLM で日本語→学習時の英語タスク文に翻訳して渡す(学習し直し不要)。
- 外部から指示を受ける、複数人で使う、実行ログを残す、が要るようになったら OpenClaw / NemoClaw を上に載せる。ロボット制御自体には不要。

---

## 落とし穴

| 失敗 | 結果 | 対策 |
| --- | --- | --- |
| カメラを動かした | 学習済みモデルが動かない | Phase 0 で確定、以後触らない |
| 推論時のタスク文が学習時と違う | 反応しない・別の動作をする | 文字列を一字一句揃える |
| タスクが似すぎ | 言語で区別できない | 色・物体・動詞を変える |
| 実演がばらつく | 50 回でも安定しない | 同じ手順・同じ速さ |
| 台が揺れる | 画角と腕の関係がずれる | 重し、滑り止め、速度を上げすぎない |

## 所要時間の目安

| Phase | 内容 | 時間 |
| --- | --- | --- |
| 0 | 設置確定 | 半日 |
| 1 | 1 タスク 50 エピソード | 2〜3 時間 |
| 2 | ACT で自律動作 | 半日 |
| 3 | 複数タスク 150〜250 エピソード | 2〜3 日 |
| 4 | SmolVLA 微調整 | 半日 |
| 5 | デモの体裁 | 半日 |

Phase 2 までで「自律で動く」、Phase 4 で「言葉で動く」。

---

## 補足: シミュレーション / デジタルツイン

### A. NVIDIA 公式 SO-101 sim-to-real ワークショップ(本命)

- リポジトリ: https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- 教材: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html (動画プレイリストあり)
- 内容: 設定・キャリブレーション → 実演収集 → **GR00T(VLA)の後段学習** → Isaac Lab で評価 → 実機デプロイ → sim2real ギャップを埋める手法。中級、6〜10 時間。
- 使うもの: Isaac Lab / Isaac GR00T / LeRobot / Cosmos。
- **終着点がこのロードマップの目標とほぼ同じ**(言語で動く VLA を SO-101 実機で)。
- **NVIDIA GPU 必須**なので Mac では動かない。FPT Cloud の H200 VM でやる。

### B. MuJoCo(軽量、Mac で動く)

`lerobot-env-so101` を導入済み。2026-09-22 に本機で動作確認:

```python
import gymnasium as gym, lerobot_env_so101
env = gym.make("lerobot_env_so101/SO101PickCube-v0", render_mode="rgb_array", image_obs=True)
```

- 観測: `pixels.front` (128×128×3)、`agent_pos` (14,)
- 行動空間: `Box(4,)` = 手先のデルタ [dx, dy, dz, gripper]
- lerobot 0.6.0+ が `lerobot_env_*` を自動検出し、env type `so101` として使える。

**重大な制約:**

- **行動空間が実機と違う。** 実機は 6 関節角、こちらは手先 4 次元。学習したポリシーはそのままでは実機に載らない。
- 作者自身が「position-only IK では top-down の把持にまだ到達できない」と明記。
- 重力下で手先が約 17mm 下がる定常偏差がある(比例制御のみのため)。

つまり**パイプラインの練習用**であって、実機用ポリシーを作る場所ではない。

他の実装: `so101-nexus`(teleop→模倣学習→RL、MuJoCo Warp で GPU 並列)、`so101-ros2-mujoco`、MuJoCo Menagerie の公式 SO-ARM100/101 モデル。

### シミュレーションが効く場面 / 効かない場面

| 効く | 効かない |
| --- | --- |
| パイプライン(録画→学習→実行)の練習。実機を壊さない | 接触の多い把持の転移。sim2real ギャップが大きい |
| カメラ配置の検討 | 模倣学習の本番データ。実演は実機が王道 |
| 強化学習(実機では試行回数が足りない) | 柔らかい物(ティッシュ等)の挙動 |
| 実機が使えない時間の開発 | |
| 合成データによる水増し | |

### 判断

デモまでの最短は**実データ**。Phase 0〜4 を実機で進めるのが速い。
ただし H200 があるなら、NVIDIA ワークショップを並行してやる価値は高い。終着点が同じで、sim2real を埋める手法がそのまま使える。
MuJoCo は「実機が使えない時間にパイプラインを触る」用途に限定して使う。

---

## 進捗ログ

### 2026-09-23 — Phase 0 完了、明日 Phase 1 へ

**できていること**

- 両アーム組み立て・配線・キャリブレーション完了。テレオペ 60Hz で安定動作(カメラなし)。
- キャリブレーションは `calib/` に git 管理の控えあり。`scripts/03g_calib_backup.sh verify` で一致確認済み。
- カメラ 2 台の画角を確定し、`docs/reference-framing/` に基準画像を保存。**以後カメラと台は動かさない。**
- 録画はローカル保存のみに設定済み(`data/<名前>/`、Hugging Face 不要)。

**解決した問題**

| 問題 | 原因 | 対策 |
| --- | --- | --- |
| キャリブレーション消失 | `~/.cache/huggingface/lerobot/` が空に | `03f` で EEPROM から復元、`calib/` に控え |
| 肩上下の補正が消える | トルク ON で EEPROM に書いていた | トルクを切ってから書く(スクリプト修正済み) |
| 手首回転が突然 1 回転 | 常用位置がエンコーダの切れ目の 11 度手前 | 実使用の向きで両アーム 0 点を取り直し、切れ目を 180 度先へ |

**明日やること**

1. **カメラの帯域問題を解決する。** 2 台を同じサンワサプライのハブに載せているため、
   640×480 で 2 分ほど動いたあと両方が 640×360 に落ちてテレオペが停止する。
   現在 `config.env` は 640×360 に変更済み(未検証)。
   - まず **カメラ 1 台を Mac に直挿し**して分ける。挿し替えたら `scripts/05b_test_cameras.sh` で番号を再確認
     (番号が入れ替わったら `config.env` の `index_or_path` を直す)。**カメラ本体の向きは動かさない。**
   - 直らなければ `scripts/05c_probe_modes.sh` で実際に安定するモードを調べる。
   - それでも駄目なら USB3.0 の給電ハブ(ROSONWAY B081H153FY ¥4,499)を買う。
2. **カメラつきテレオペを 5 分以上連続で回して安定を確認。**
3. **Phase 1: 50 エピソード録画。**
   ```bash
   scripts/06_record.sh so101_cube 50 "Pick up the red cube and put it in the box"
   ```
   タスク文は実際の動作に合わせて決め、**推論時も同じ文字列を使う**ので記録しておくこと。

**再開時の確認コマンド**

```bash
scripts/00b_scan_bus.sh both          # モーター 1〜6 が両アームで見えるか
scripts/03g_calib_backup.sh verify    # キャリブレーションが一致しているか
scripts/05b_test_cameras.sh           # カメラ番号と画角(ターミナル.app から)
```


---

## 2026-09-28 — Phase 2 の経路が通った(初の自律動作)

**12 エピソード / 2000 ステップの ACT で、ロボットがキューブの方へ自律的に向かった。** 掴むところまでは至らず。

| 段階 | 結果 |
| --- | --- |
| 録画 | 12 エピソード、6982 フレーム、30Hz |
| 転送 | 415MB を H200 へ、7 分半 |
| 学習 | H200 で 2000 ステップ 3 分 29 秒、損失 10.28 → 2.23 |
| モデル回収 | 207MB、1 分 40 秒 |
| 実機で自律動作 | 30 秒完走。**キューブの方向へ移動**、掴みは失敗 |

### 残る課題

**1. 実行時の制御ループが 8〜11Hz しか出ていない(要修正)**

学習データは 30Hz なので、10Hz で実行すると動作が 3 倍に引き伸ばされ**カクカクする**。
ポリシー側は無実で、前処理・転送・推論の合計は 1.5ms(655Hz 相当)しかない。
残り約 100ms は `robot.get_observation()`(モーター読み取り + カメラ)で消えている。

```bash
scripts/09b_profile_obs.sh    # ターミナル.app から。内訳を測る
```

**2. 掴めない(データ量の問題)**

対象へ近づく大まかな動きは少量のデータでも学べるが、掴む瞬間の精密な位置合わせには実演数が要る。
50 エピソード + 10 万ステップ(H200 で約 2.7 時間)で改善が見込める。

### 次の手順

```bash
# 1. 残り 38 エピソードを録る(合計 50 本)
NO_DISPLAY=1 scripts/06_record.sh so101_cube 38 "Pick up the red cube and place it in the bowl" --resume=true

# 2. H200 へ転送して本番の学習
rsync -az data/so101_cube/ gpu-server:~/ai-arm/data/so101_cube/
# VM 側で nohup lerobot-train ... --steps=100000 &  (docs/h200-training.md 参照)

# 3. モデルを回収して実行
rsync -az gpu-server:'~/ai-arm/outputs/train/act_so101_cube/checkpoints/last/pretrained_model/' \
  outputs/train/act_so101_cube/checkpoints/last/pretrained_model/
scripts/08_rollout.sh outputs/train/act_so101_cube/checkpoints/last/pretrained_model \
  "Pick up the red cube and place it in the bowl" 30
```
