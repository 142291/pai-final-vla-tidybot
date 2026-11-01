"""「言えば片付け、失敗から学ぶ」の本体: 指示 → 段取り → π0.5 → 成否判定 → やり直し/人の介入 → 記録。

使い方(カメラ権限のあるターミナルから。H200 の policy server が動いていること):
    scripts/17_orchestrate.sh "片付けて"
    scripts/17_orchestrate.sh "紫以外を箱に入れて"

流れ
  1. 学習データの開始姿勢(HOME_POSE)をホームとし、起動したらまずそこへゆっくり動かす。
  2. 俯瞰画像と指示から planner.py が色の列を作る(Claude、使えなければキーワード)。
  3. 色ごとに π0.5 を学習時と同じ英語の指示文で動かし、俯瞰カメラで対象が箱に入ったかを判定する。
       成功 = 対象の色が箱の領域で SUCCESS_HOLD 秒止まっている(運んでいる途中は動くので除ける)。
       失敗 = ATTEMPT_SEC 秒たっても成功しない、または f キー。
     どちらでもホームへ滑らかに戻してから次へ。失敗なら MAX_ATTEMPTS 回まで繰り返す。
  4. それでも駄目なら声で人を呼ぶ。人はリーダーアームで手伝う(h で開始/終了)。
     動作中にいつでも h を押せば、その場で人の操作に切り替わる(介入)。
  5. 人が操作した区間は data/flywheel_human に、自分で成功した試行は data/flywheel_auto に、
     学習データと同じ形式(カメラ名 top / wrist)で 1 回 = 1 エピソードとして保存する。

画面(HUD)の操作: h = 人が操作する/やめる、s = 成功にする、f = 失敗にする(手伝いを頼まれたときは「飛ばす」)、q = 終了
記録: outputs/runs/<日時>/ に events.jsonl(出来事)と hud.mp4(画面の録画、デモ動画の素材)。

2026-10-24: 単発の成功率は 40% だが、判定とやり直しで「完了率」を上げ、人の介入から学び直す
(フライホイール)ことを作品の軸にしたため作った。
2026-10-28: キューブ以外の物(消しゴム)も扱えるようにした。まだ教わっていない物(段取りの new)は
MAX_ATTEMPTS_NEW 回(既定 1)だけ試し、駄目ならすぐ人を呼んで持ち方を見せてもらう(その動きを保存して学び直す)。キューブ以外は色検出が使えないので、箱に入ったかを
Claude に俯瞰画像で判定させる(VlmJudge。2 回続けて「入っている」で成功)。
"""


import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from importlib import import_module

import cv2
import draccus
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "remote"))
from fast_client import SYNC_K, SYNC_SETTLE, FastRobotClient  # noqa: E402
from planner import JA, make_plan  # noqa: E402

from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

audit = import_module("06d_audit_dataset")
objects = import_module("objects")
explain_mod = import_module("explain")
EXPLAIN = os.environ.get("EXPLAIN", "1") == "1"   # 失敗したら、何が起きたかを Claude に判定させて読み上げる

ATTEMPT_SEC = float(os.environ.get("ATTEMPT_SEC", "35"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
MAX_ATTEMPTS_NEW = int(os.environ.get("MAX_ATTEMPTS_NEW", "1"))      # まだ教わっていない物は 1 回試したら人を呼ぶ
ATTEMPT_SEC_NEW = float(os.environ.get("ATTEMPT_SEC_NEW", "30"))
SUCCESS_HOLD = float(os.environ.get("SUCCESS_HOLD", "1.0"))
GRIP_OPEN = float(os.environ.get("GRIP_OPEN", "12"))       # これより大きければ「開いている」(ホームは約 1 = 閉)
HOME_SEC = float(os.environ.get("HOME_SEC", "2.5"))
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
HOME_POSE = [-7.8, -98.3, 99.5, -99.1, 4.8, 0.8]   # 学習データ 189 本の開始姿勢の中央値
HUMAN_MAX_STEP = float(os.environ.get("HUMAN_MAX_STEP", "4"))  # 人の操作に切り替えた直後の飛びを抑える(1 周あたりの最大変化)
REC_HUMAN = os.environ.get("REC_HUMAN", "flywheel_human")
REC_AUTO = os.environ.get("REC_AUTO", "flywheel_auto")      # 空文字なら自動成功は保存しない
CAM_TOP, CAM_WRIST = "base_0_rgb", "left_wrist_0_rgb"     # 推論用の名前。データセットでは top / wrist
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
VOICE = os.environ.get("VOICE", "Kyoko")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")   # キューブ以外の物が箱に入ったかの判定
JUDGE_EVERY = float(os.environ.get("JUDGE_EVERY", "2.5"))          # 判定の間隔(秒)


_speech = []                      # 読み上げ待ちの文(順に 1 つずつ話す)
_speech_cv = threading.Condition()
_speaking = [False]


def _speaker():
    while True:
        with _speech_cv:
            while not _speech:
                _speech_cv.wait()
            text = _speech.pop(0)
            _speaking[0] = True
        try:
            subprocess.run(["say", "-v", VOICE, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        with _speech_cv:
            _speaking[0] = False
            _speech_cv.notify_all()


threading.Thread(target=_speaker, daemon=True).start()


def say(text: str):
    """読み上げ(呼んだ側は待たない)。前の文を話し終えてから次を話す。

    2026-10-31: 「赤、入りました」と「終わりました」が同時に流れて重なった(say を待たずに次々起動していた)ため、
    1 本の列に並べて順に話すようにした。
    """
    if text:
        with _speech_cv:
            _speech.append(text)
            _speech_cv.notify_all()


def say_wait(timeout=15.0):
    """読み上げ待ちの文をすべて話し終えるまで待つ(終了前に最後の一言が切れないように)。"""
    t0 = time.time()
    with _speech_cv:
        while (_speech or _speaking[0]) and time.time() - t0 < timeout:
            _speech_cv.wait(timeout=0.2)


def vlm_in_box(top_bgr, obj: str) -> bool:
    """俯瞰画像で、物が箱の底に置かれているか(グリッパで持っていないか)を Claude に判定させる。"""
    import base64

    import anthropic
    from pydantic import BaseModel, Field

    class InBox(BaseModel):
        in_box: bool = Field(description="対象が箱(白い底の黒い枠のトレー)の中に置かれていて、グリッパに持たれていなければ true")
        reason: str = Field(description="判断の理由(日本語、短く)")

    ok, jpg = cv2.imencode(".jpg", top_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    resp = anthropic.Anthropic().beta.messages.parse(
        model=JUDGE_MODEL, max_tokens=1000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpg.tobytes()).decode()}},
            {"type": "text", "text": f"卓上ロボットの俯瞰カメラの画像です。{objects.JA[obj]}は箱の中に置かれていますか? "
                                     "ロボットのグリッパが持ったまま箱の上にある場合は、まだ置いていないので false です。"},
        ]}],
        output_format=InBox,
    )
    if resp.parsed_output is None:
        raise RuntimeError(f"判定が返らなかった(stop_reason={resp.stop_reason})")
    return resp.parsed_output.in_box


class VlmJudge:
    """試行のあいだ裏で JUDGE_EVERY 秒ごとに vlm_in_box を呼ぶ。直近 2 回が続けて「入っている」なら positive。"""

    def __init__(self, orch, obj):
        self.orch, self.obj, self.verdicts, self.active = orch, obj, [], True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.active and not self.orch.quit:
            t0 = time.time()
            obs = self.orch.latest()
            try:
                v = vlm_in_box(cv2.cvtColor(obs[CAM_TOP], cv2.COLOR_RGB2BGR), self.obj)
                self.verdicts.append((time.time(), v))
                self.orch.log("judge", obj=self.obj, in_box=v, sec=round(time.time() - t0, 1))
            except Exception as e:   # 通信の失敗などは「判定なし」として続ける(f/s キーで決められる)
                self.orch.log("judge_error", obj=self.obj, error=f"{type(e).__name__}: {e}")
            time.sleep(max(0.0, JUDGE_EVERY - (time.time() - t0)))

    def positive(self) -> bool:
        v = self.verdicts[-2:]
        return len(v) == 2 and all(x for _, x in v) and time.time() - v[-1][0] < 3 * JUDGE_EVERY + 6

    def stop(self):
        self.active = False


class Recorder:
    """学習データと同じ形式で 1 区間 = 1 エピソードを保存する。"""

    def __init__(self, name: str, robot_type: str):
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ref = json.load(open(os.path.join(ROOT, "data", "so101_cube_v3_train", "meta", "info.json")))["features"]
        feats = {k: v for k, v in ref.items() if k in ("action", "observation.state", "observation.images.top", "observation.images.wrist")}
        for v in feats.values():
            v["shape"] = tuple(v["shape"])
        root = os.path.join(ROOT, "data", name)
        # 録画(06_record.sh)と同じ符号化にする: h264(VideoToolbox)、crf 23、逐次符号化
        enc = dict(rgb_encoder=RGBEncoderConfig(vcodec=os.environ.get("VCODEC", "h264_videotoolbox"), crf=23),
                   streaming_encoding=True, encoder_threads=4)
        if os.path.exists(os.path.join(root, "meta", "info.json")):   # 前回の続きに追記する
            self.ds = LeRobotDataset.resume(f"local/{name}", root=root, **enc)
        else:
            self.ds = LeRobotDataset.create(f"local/{name}", 30, feats, root=root, robot_type=robot_type,
                                            use_videos=True, **enc)
        self.names = feats["action"]["names"]
        self.n = 0

    def add(self, obs: dict, action: dict, task: str):
        self.ds.add_frame({
            "observation.state": np.array([obs[k] for k in self.names], dtype=np.float32),
            "observation.images.top": obs[CAM_TOP],
            "observation.images.wrist": obs[CAM_WRIST],
            "action": np.array([action[k] for k in self.names], dtype=np.float32),
            "task": task,
        })
        self.n += 1

    def save(self, keep: bool):
        if self.n == 0:
            return
        if keep and self.n >= 30:   # 1 秒未満は誤操作とみなして捨てる
            self.ds.save_episode()
        else:
            self.ds.clear_episode_buffer()
        self.n = 0

    def close(self):
        if hasattr(self.ds, "finalize"):
            self.ds.finalize()


class Orchestrator(FastRobotClient):
    def __init__(self, config: RobotClientConfig, instruction: str):
        super().__init__(config)
        self.instruction = instruction
        self.mode = "idle"            # idle / policy / script / human
        self.task = objects.task("red")
        self.judge = None             # キューブ以外の試行中の VlmJudge
        self.epoch = 0.0              # この時刻より前の観測にもとづく行動チャンクは捨てる(色の切り替え・介入の後)
        self.script = deque()
        self.obs = None
        self.obs_lock = threading.Lock()
        self.home = None
        self.box = None
        self.keys = deque()
        self.quit = False
        self.last_sent = None
        self.last_age = None
        self.status = {"plan": [], "state": {}, "now": "", "attempt": "", "msg": ""}
        self.rec = None               # 今記録している Recorder
        self._human_synced = False    # 人の操作に切り替えた後、リーダーとフォロワーの姿勢が揃ったか
        self.rec_task = ""
        self.leader = None
        if os.environ.get("LEADER_PORT"):
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

            self.leader = SO101Leader(SO101LeaderConfig(port=os.environ["LEADER_PORT"], id=os.environ.get("LEADER_ID", "leader"),
                                                         use_degrees=False))
            # 2026-10-27: 接続時の書き込みでグリッパ(id 6)が 1 回だけ「Input voltage error」を返して落ちた。
            # 一時的なエラーなので、つなぎ直して 3 回まで試す
            for i in range(3):
                try:
                    self.leader.connect()
                    break
                except RuntimeError as e:
                    self.logger.warning(f"[orch] リーダーの接続に失敗({i + 1} 回目): {e}")
                    try:
                        self.leader.bus.disconnect(disable_torque=False)
                    except Exception:
                        pass
                    if i == 2:
                        raise
                    time.sleep(1.0)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(ROOT, "outputs", "runs", stamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self.events = open(os.path.join(self.run_dir, "events.jsonl"), "a")
        self.video = None
        self.font = ImageFont.truetype(FONT, 22)
        self.font_big = ImageFont.truetype(FONT, 30)
        self.font_small = ImageFont.truetype(FONT, 15)
        self.rec_human = Recorder(REC_HUMAN, "so_follower") if self.leader else None
        self.rec_auto = Recorder(REC_AUTO, "so_follower") if REC_AUTO else None

    # --- 記録 ---------------------------------------------------------------
    def log(self, event: str, **kw):
        row = {"t": time.time(), "event": event, **kw}
        self.events.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.events.flush()
        self.logger.info(f"[orch] {event} {kw}")

    # --- 推論結果の受け取り: 古い観測にもとづくチャンクは捨てる -----------------------
    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        if not incoming_actions or self.mode != "policy" or incoming_actions[0].get_timestamp() < self.epoch:
            return
        self.last_age = time.time() - incoming_actions[0].get_timestamp()
        return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def _clear_queue(self):
        with self.action_queue_lock:
            while not self.action_queue.empty():
                self.action_queue.get_nowait()

    def latest(self):
        with self.obs_lock:
            return self.obs

    def state(self, obs=None) -> dict:
        obs = obs or self.latest()
        return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

    # --- 制御ループ(メインスレッド、30Hz) --------------------------------------
    def control_loop(self, task: str = "", verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting (orchestrator)")
        while self.running and not self.quit:
            t0 = time.perf_counter()
            try:
                obs = self.robot.get_observation()
            except (TimeoutError, RuntimeError, ConnectionError) as e:
                # 2026-10-25: 手首カメラの 1 枚が 1.6 秒遅れただけで制御ループが落ち、通し動作が終わった。
                # 1 周飛ばして続ける(ロボットは直前の姿勢を保つ)。続くようなら記録に残す
                self._obs_errors = getattr(self, "_obs_errors", 0) + 1
                if self._obs_errors % 30 == 1:
                    self.logger.warning(f"[orch] 観測の取得に失敗({self._obs_errors} 回目): {e}")
                time.sleep(0.05)
                continue
            with self.obs_lock:
                self.obs = obs
            sent = None
            if self.mode == "policy":
                if self.actions_available():
                    sent = self.control_loop_action(verbose)
                if SYNC_K > 0:    # 止まって考える方式(fast_client.py の SYNC_K を参照)
                    if not self.actions_available():
                        self._sync_step(self.task, obs)
                elif self._ready_to_send_observation():
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    tobs = TimedObservation(timestamp=time.time(), observation={**obs, "task": self.task},
                                            timestep=max(latest_action, 0))
                    with self.action_queue_lock:
                        tobs.must_go = self.must_go.is_set() and self.action_queue.empty()
                    self.send_observation(tobs)
                    if tobs.must_go:
                        self.must_go.clear()
            elif self.mode == "script":
                if self.script:
                    sent = self.robot.send_action(self.script.popleft())
            elif self.mode == "human" and self.leader is not None:
                target = self.leader.get_action()
                cur = self.last_sent or self.state(obs)
                sent = self.robot.send_action({k: cur[k] + float(np.clip(target[k] - cur[k], -HUMAN_MAX_STEP, HUMAN_MAX_STEP))
                                               for k in target})
                # 切り替えた直後の「追いつき」(フォロワーがリーダーへ急いで寄る動き)は記録しない。
                # 2026-10-26: 人の手本 11 本のうち 8 本の先頭にこの急な動きが入っていた(22_trim_episodes.py で後から除いた)
                if not self._human_synced:
                    st = self.state(obs)
                    self._human_synced = max(abs(target[k] - st[k]) for k in JOINTS[:5]) < 5
            if sent:
                self.last_sent = {k: float(v) for k, v in sent.items()}
            if self.rec is not None and sent and (self.mode != "human" or self._human_synced):
                self.rec.add(obs, sent, self.rec_task)
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - t0)))
        return None, None

    # --- 画面 ----------------------------------------------------------------
    def text_panel(self):
        s = self.status
        # ✗ と ✋ はヒラギノに無く ☒ になっていた(2026-10-27、録画で確認)。フォントにある字を使う
        marks = {"done": "✓", "fail": "×", "doing": "▶", "todo": "・", "human": "人"}
        plan = "  ".join(f"{marks.get(s['state'].get(c, 'todo'), '・')}{JA[c]}" for c in s["plan"]) or "(段取り中)"
        mode = {"policy": "π0.5 実行中", "script": "ホームへ", "human": "人が操作中", "idle": "待機"}[self.mode]
        age = f"  遅延 {self.last_age * 1000:.0f}ms" if self.last_age and self.mode == "policy" else ""
        lines = (f"指示: {self.instruction}", f"段取り: {plan}", f"{s['now']} {s['attempt']}  [{mode}]{age}", s["msg"])
        if lines != getattr(self, "_panel_key", None):
            img = Image.new("RGB", (1280, 150))
            d = ImageDraw.Draw(img)
            d.text((14, 8), lines[0], font=self.font_big, fill=(255, 255, 255))
            d.text((14, 50), lines[1], font=self.font_big, fill=(255, 230, 120))
            d.text((14, 92), lines[2], font=self.font, fill=(180, 220, 255))
            d.text((14, 120), lines[3], font=self.font, fill=(255, 180, 180))
            d.text((1020, 122), "h 介入 s 成功 f 失敗 q 終了", font=self.font_small, fill=(150, 150, 150))
            self._panel = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._panel_key = lines
        return self._panel

    def draw_hud(self, obs):
        top = cv2.cvtColor(obs[CAM_TOP], cv2.COLOR_RGB2BGR)
        wrist = cv2.cvtColor(obs[CAM_WRIST], cv2.COLOR_RGB2BGR)
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            cv2.rectangle(top, (x0, y0), (x1, y1), (255, 255, 255), 1)
        frame = np.vstack([np.hstack([cv2.resize(top, (640, 360)), cv2.resize(wrist, (640, 360))]), self.text_panel()])
        if self.video is None:
            self.video = cv2.VideoWriter(os.path.join(self.run_dir, "hud.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                         (frame.shape[1], frame.shape[0]))
        self.video.write(frame)
        cv2.imshow("orchestrator", frame)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            self.keys.append(chr(k))
            if chr(k) == "q":      # 段取り中・ホームへ戻る途中でも止められるように
                self.quit = True

    def pop_key(self):
        return self.keys.popleft() if self.keys else None

    # --- 動作の部品 -------------------------------------------------------------
    def go_home(self, sec=None):
        cur = self.last_sent or self.state()
        n = max(2, int((sec or HOME_SEC) * 30))
        self.script = deque({k: cur[k] + (self.home[k] - cur[k]) * (1 - np.cos(np.pi * i / (n - 1))) / 2 for k in self.home}
                            for i in range(n))
        self.mode = "script"
        while self.script and not self.quit:
            time.sleep(0.05)
        time.sleep(0.4)

    def go_home_blocking(self):
        cur = self.last_sent or self.state(self.robot.get_observation())
        n = max(2, int(HOME_SEC * 30))
        for i in range(n):
            a = (1 - np.cos(np.pi * i / (n - 1))) / 2
            self.robot.send_action({k: cur[k] + (self.home[k] - cur[k]) * a for k in self.home})
            time.sleep(1 / 30)

    def target_pos_in_box(self, color):
        """対象が箱の領域内に見えていればその位置、そうでなければ None。

        キューブ以外は位置が取れないので、Claude の判定(VlmJudge)が続けて「入っている」なら (0, 0) を返す
        (位置が動かないので、そのまま SUCCESS_HOLD 秒の判定に乗る)。
        """
        if not objects.OBJECTS[color]["cube"]:
            return (0, 0) if self.judge is not None and self.judge.positive() else None
        obs = self.latest()
        p = audit.find_cube(cv2.cvtColor(obs[CAM_TOP], cv2.COLOR_RGB2BGR), color)
        return p if audit.inside(p, self.box) else None

    def target_in_box(self, color) -> bool:
        if not objects.OBJECTS[color]["cube"]:
            try:
                return vlm_in_box(cv2.cvtColor(self.latest()[CAM_TOP], cv2.COLOR_RGB2BGR), color)
            except Exception as e:
                self.log("judge_error", obj=color, error=f"{type(e).__name__}: {e}")
                return False
        return self.target_pos_in_box(color) is not None

    def attempt(self, color, limit=None) -> str:
        """π0.5 で 1 回試す。'success' / 'fail' / 'human'(介入を求められた) / 'quit' を返す。"""
        limit = limit or ATTEMPT_SEC
        self.task = objects.task(color)
        if not objects.OBJECTS[color]["cube"]:
            self.judge = VlmJudge(self, color)
        self._clear_queue()
        self.epoch = time.time()
        self.must_go.set()
        self._sync_waiting, self._sync_stop_t, self._sync_sent_t = True, self.epoch - SYNC_SETTLE, None
        if self.rec_auto:
            self.rec, self.rec_task = self.rec_auto, self.task
        self.mode = "policy"
        t0, held = time.time(), None
        self.att_frames, t_frame = [], 0.0     # 失敗の説明用に 2 秒ごとの画像を残す
        # 2026-10-31: 前の試行の終わり(説明を読み上げている間など)に押した f が残っていて、次の試行が 0.1 秒で
        # 失敗扱いになった。試行の始めに、それまでに押されたキーを捨てる
        self.keys.clear()
        result = "fail"
        while not self.quit:
            k = self.pop_key()
            if k in ("q",):
                result = "quit"
                break
            if k == "h" and self.leader is not None:
                result = "human"
                break
            if k == "s":
                result = "success"
                break
            if k == "f":
                break
            # 成功 = 箱の領域内で SUCCESS_HOLD 秒止まっている(運んでいる途中は動くので除ける)。
            # 2026-10-25: 以前は「グリッパが開いている」も条件にしていたが、放した後すぐ閉じる回があり、
            # 紫が箱に入っても成功と判定できなかった。
            p = self.target_pos_in_box(color)
            if p is not None:
                if held is None or np.hypot(p[0] - held[1][0], p[1] - held[1][1]) > 8:
                    held = (time.time(), p)          # 入った・まだ動いている → 数え直す
                elif time.time() - held[0] >= SUCCESS_HOLD:
                    result = "success"
                    break
            else:
                held = None
            if time.time() - t0 > limit:
                break
            self.status["attempt"] = self.status["attempt"].split(" ")[0] + f" {time.time() - t0:.0f}s"
            if time.time() - t_frame >= 2.0:
                obs = self.latest()
                self.att_frames.append((obs[CAM_TOP].copy(), obs[CAM_WRIST].copy()))
                t_frame = time.time()
            time.sleep(0.1)
        self.mode = "script"      # その場で止める(スクリプトが空なので指令を出さず、今の姿勢を保つ)
        self.script.clear()
        self._clear_queue()
        if self.judge is not None:
            self.judge.stop()
            self.judge = None
        if self.rec is self.rec_auto and self.rec is not None:
            self.rec = None
            self.rec_auto.save(keep=(result == "success"))
        self.log("attempt", color=color, result=result, sec=round(time.time() - t0, 1))
        return result

    def explain_failure(self, color) -> str:
        """直前の試行の画像から失敗の様子を Claude に説明させ、読み上げる一言を返す(使えなければ空文字)。"""
        fr = getattr(self, "att_frames", [])
        if not EXPLAIN or len(fr) < 3:
            return ""
        idx = sorted({min(len(fr) - 1, int(len(fr) * f)) for f in (0.0, 0.4, 0.6, 0.8, 0.98)})
        frames = [fr[i] for i in idx]
        self.status["msg"] = "何が起きたかを確認中..."
        try:
            t0 = time.time()
            e = explain_mod.explain(frames, color, explain_mod.target_from_frame(frames[0][0], color))
            self.log("explain", color=color, kind=e.kind, say=e.say, reason=e.reason, sec=round(time.time() - t0, 1))
            self.status["msg"] = e.say
            return e.say.rstrip("。") + "。"
        except Exception as ex:
            self.log("explain_error", color=color, error=f"{type(ex).__name__}: {ex}")
            return ""

    def human_help(self, color, already_started=False, new=False) -> str:
        """リーダーアームで人が手伝う。h で開始・終了。終了時に判定する。"""
        if self.leader is None:
            say("手伝ってください。終わったら、エスキーを押してください。")
            self.status["msg"] = "手で対象を箱に入れたら s、諦めるなら f"
            while not self.quit:
                k = self.pop_key()
                if k in ("s", "f", "q"):
                    return {"s": "human_success", "f": "human_fail", "q": "quit"}[k]
                time.sleep(0.1)
            return "quit"
        if not already_started:
            said, self.last_explain = getattr(self, "last_explain", ""), ""
            say(said + (f"{JA[color]}は初めてです。リーダーアームで、持ち方を見せてください。" if new
                        else "うまく掴めません。リーダーアームで手伝ってください。"))
            self.status["msg"] = "リーダーをフォロワーと同じ姿勢にして h で操作開始(手で入れたら s、飛ばすなら f)"
            while not self.quit:
                k = self.pop_key()
                if k == "h":
                    break
                if k in ("s", "f", "q"):   # s = 手で箱に入れた、f = 手伝わずに次の色へ、q = 終了
                    r = {"s": "human_success", "f": "skipped", "q": "quit"}[k]
                    self.status["msg"] = ""
                    self.log("human_skip", color=color, key=k)
                    return r
                time.sleep(0.05)
        self.task = objects.task(color)
        self.rec, self.rec_task = self.rec_human, self.task
        self.keys.clear()          # 押しっぱなし・二度押しで、始めた直後に終わらないように
        self._human_synced = False
        self.mode = "human"
        self.status["msg"] = "人が操作中(箱に入れたら h)"
        self.log("human_start", color=color, new=new)
        t0 = time.time()
        while not self.quit:
            k = self.pop_key()
            # 2026-10-25: h が 2 回続けて届き、手伝いが 0 秒で終わった。始めて 2 秒以内の h は無視する
            if k == "q" or (k == "h" and time.time() - t0 > 2.0):
                break
            time.sleep(0.05)
        self.mode = "script"
        self.script.clear()
        ok = self.target_in_box(color)
        self.rec = None
        self.rec_human.save(keep=True)   # 人の操作は成否にかかわらず残す(失敗分は後で目視で選ぶ)
        self.status["msg"] = ""
        self.log("human_end", color=color, in_box=ok, sec=round(time.time() - t0, 1))
        return "human_success" if ok else "human_fail"

    # --- 全体の流れ(別スレッド) ---------------------------------------------------
    def brain(self):
        while self.latest() is None:
            time.sleep(0.1)
        time.sleep(0.5)
        obs = self.latest()
        # ホームは学習データの開始姿勢(中央値)に固定し、起動したらまずそこへゆっくり動かす。
        # 2026-10-26: 起動時の姿勢をホームにしていたため、アームがずれた姿勢(肩 -68、手首 +43)のまま起動すると、
        # やり直しのたびにその姿勢へ戻り、学習に無い姿勢から始めて失敗し続けた。
        start = self.state(obs)
        self.home = dict(zip(JOINTS, HOME_POSE))
        far = {k: round(start[k] - self.home[k]) for k in JOINTS[:5] if abs(start[k] - self.home[k]) > 8}
        if far:
            self.log("home_correct", start=start, diff=far)
            self.status["msg"] = "ホーム姿勢へ移動中"
        self.go_home(sec=max(HOME_SEC, 4.0) if far else HOME_SEC)
        time.sleep(0.5)
        obs = self.latest()
        top = cv2.cvtColor(obs[CAM_TOP], cv2.COLOR_RGB2BGR)
        self.box = audit.find_box(top)
        self.log("start", instruction=self.instruction, home=self.home, box=self.box,
                 run=os.environ.get("RUN", "pi05_v3full"), ckpt=os.environ.get("CKPT", "020000"),
                 sync_k=SYNC_K, round=os.environ.get("ROUND", ""))
        self.status["now"] = "段取りを考え中..."
        plan = make_plan(self.instruction, top)
        self.status["plan"] = list(plan.steps)
        self.status["msg"] = plan.reply
        self.log("plan", steps=plan.steps, new=plan.new, unknown=plan.unknown, reply=plan.reply, reason=plan.reason,
                 source=plan.source)
        say(plan.reply)
        summary = []
        for color in plan.steps:
            self.status["state"][color] = "doing"
            outcome, tries = None, 0
            new = color in plan.new
            max_tries = MAX_ATTEMPTS_NEW if new else MAX_ATTEMPTS
            while tries < max_tries and outcome is None and not self.quit:
                tries += 1
                self.status["now"] = f"{JA[color]}を箱へ" + ("(初めて)" if new else "")
                self.status["attempt"] = f"{tries}/{max_tries}回目"
                r = self.attempt(color, limit=ATTEMPT_SEC_NEW if new else None)
                if r == "quit":
                    self.quit = True
                elif r == "human":                       # 動作中に人が介入した
                    outcome = self.human_help(color, already_started=True, new=new)
                    self.go_home()
                elif r == "success":
                    self.go_home()
                    outcome = "auto_success"
                    say(f"{JA[color]}、入りました。")
                else:
                    self.go_home()
                    self.last_explain = self.explain_failure(color)
                    if tries < max_tries:
                        say(self.last_explain + "もう一度やってみます。")
            if self.quit:
                break
            if outcome is None:                          # 決められた回数やっても入らなかった
                outcome = self.human_help(color, new=new)
                if outcome == "quit":
                    self.quit = True
                    break
                self.go_home()
            self.status["state"][color] = {"auto_success": "done", "human_success": "human"}.get(outcome, "fail")
            summary.append({"color": color, "outcome": outcome, "attempts": tries, "new": new})
            self.log("step", color=color, outcome=outcome, attempts=tries, new=new)
        auto = sum(s["outcome"] == "auto_success" for s in summary)
        done = sum(s["outcome"] in ("auto_success", "human_success") for s in summary)
        self.log("finish", summary=summary, auto=auto, done=done, total=len(plan.steps))
        self.status["now"] = f"完了 {done}/{len(plan.steps)}(自力 {auto})"
        self.status["attempt"] = ""
        if not self.quit:
            if plan.steps:      # 断った(段取りが空の)ときは「終わりました」と言わない
                say("終わりました。" if done == len(plan.steps) else "一部、片付けられませんでした。")
            say_wait()
            time.sleep(1)
        self.quit = True

    def shutdown(self):
        for r in (self.rec_human, self.rec_auto):
            if r is not None:
                r.close()
        if self.video is not None:
            self.video.release()
        cv2.destroyAllWindows()
        if self.leader is not None:
            self.leader.disconnect()


@draccus.wrap()
def main(cfg: RobotClientConfig):
    client = Orchestrator(cfg, instruction=cfg.task)
    if not client.start():
        return
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    threading.Thread(target=client.brain, daemon=True).start()
    loop = threading.Thread(target=client.control_loop, daemon=True)
    loop.start()
    try:
        while loop.is_alive() and not client.quit:
            obs = client.latest()
            if obs is not None:
                client.draw_hud(obs)
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        # 途中終了(q / Ctrl+C)でも、トルクを切る前にホームへ戻す。制御ループを止めてから直接送る
        client.quit = True
        loop.join(timeout=2)
        if client.home is not None and client.running and client.mode != "idle":
            try:
                client.go_home_blocking()
            except Exception as e:  # 終了処理は止めない
                print(f"ホームへ戻せなかった: {e}")
        client.quit = True
        client.shutdown()
        client.stop()
        print(f"記録: {client.run_dir}")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
