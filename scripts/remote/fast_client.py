"""制御ループを止めずに観測を送る robot_client(Mac 側で起動する)。

lerobot.async_inference.robot_client との違い:
  1. 観測の送信を裏スレッドに移す。制御ループは観測を「郵便受け」に置くだけで戻る。
     郵便受けは 1 枠で、送信中に新しい観測が来たら古い方を捨てる(古い観測を送っても意味がない)。
  2. 画像を元の解像度のまま JPEG にする(fast_common.py)。
  3. 遅延を計測して 10 秒ごとに表示する:
       send  … 観測 1 回のアップロード時間
       age   … 観測を撮ってからその観測にもとづく行動チャンクが届くまでの時間(Mac の時計だけで測る)
       loop  … 制御ループ 1 周の時間(30Hz なら 33ms 付近が正常)

使い方(引数は lerobot.async_inference.robot_client と同じ。JPEG 品質は環境変数 JPEG_QUALITY、既定 90):
    python scripts/remote/fast_client.py --server_address=127.0.0.1:8124 ...
"""

import os
import sys
import threading
import time
from queue import Queue
from dataclasses import asdict
from pprint import pformat

import draccus
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_common import encode_images  # noqa: E402

from lerobot.async_inference.configs import RobotClientConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation, visualize_action_queue_size  # noqa: E402
from lerobot.async_inference.robot_client import RobotClient  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "90"))
# 観測の送信は最大 SEND_HZ 回/秒に抑える。サーバーの推論は約 2 回/秒(1 回 ~0.5 秒)で、
# それ以上送っても最新の 1 枚以外は捨てられるだけ。2026-10-17 の実走では 6 回/秒送っており、
# Wi-Fi の上りを自分で詰まらせて送信時間が 145ms → 350〜460ms に伸びていた。
SEND_HZ = float(os.environ.get("SEND_HZ", "4"))
REPORT_SEC = 10.0
# 止まって考える方式(SYNC_K > 0 で有効): チャンクの先頭 SYNC_K 個だけ実行して止まり、止まった後に撮った
# 観測で次のチャンクを作らせる。遅延で「予測の後半(外れやすい)」を実行してしまう問題を避ける代わりに、
# 推論の待ち時間だけ動きが途切れる。
# 2026-10-24: 学習に使っていない回での誤差は予測の 0〜15 ステップ目が 3.3、25〜50 ステップ目が 11〜13。
# 遅延 0.9〜1.3 秒の通常方式では後半を実行しており、「近くまで行くが掴めない」が続いたため。
SYNC_K = int(os.environ.get("SYNC_K", "0"))
SYNC_SETTLE = float(os.environ.get("SYNC_SETTLE", "0.15"))   # 止まってから観測を撮るまでの待ち(揺れが収まるのを待つ)
# 設定すると、送った観測(JPEG のまま)と関節角・実行した行動をこのフォルダに保存する(14_eval.py が試行ごとに設定)。
# 2026-10-19: 「左右だと少しずれた所を掴み続ける」のずれの向き・大きさを後から画像で確かめるため。
REC_DIR = os.environ.get("TRIAL_REC_DIR", "")


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.send_ms, self.age_ms, self.loop_ms, self.kb, self.dropped = [], [], [], [], 0

    def add(self, name, value):
        with self.lock:
            getattr(self, name).append(value)

    def summary(self) -> str:
        with self.lock:
            def p(xs, q):
                return f"{np.percentile(xs, q):.0f}" if xs else "-"
            s = (
                f"send p50/p95 {p(self.send_ms, 50)}/{p(self.send_ms, 95)}ms ({p(self.kb, 50)}KB) | "
                f"age p50/p95 {p(self.age_ms, 50)}/{p(self.age_ms, 95)}ms | "
                f"loop p50/p95 {p(self.loop_ms, 50)}/{p(self.loop_ms, 95)}ms | "
                f"送信 {len(self.send_ms)} 回, 間引き {self.dropped} 回"
            )
            self.reset()
            return s


class FastRobotClient(RobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        self._mailbox = None
        self._mailbox_cv = threading.Condition()
        self._stats = _Stats()
        self._last_report = time.perf_counter()
        self._rec = None
        if REC_DIR:
            os.makedirs(REC_DIR, exist_ok=True)
            self._rec = open(os.path.join(REC_DIR, "log.jsonl"), "a")
            self._rec_lock = threading.Lock()
        self._sync_waiting = True     # SYNC_K: 次のチャンク待ち(止まっている)
        self._sync_stop_t = 0.0
        self._sync_sent_t = None
        self._sender = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender.start()

    def _record_obs(self, obs, encoded):
        step = obs.get_timestep()
        state = {}
        for k, v in encoded.items():
            if isinstance(v, dict):
                with open(os.path.join(REC_DIR, f"{step:05d}_{k}.jpg"), "wb") as fh:
                    fh.write(v["__jpeg__"])
            elif isinstance(v, (int, float, np.number)) or (isinstance(v, np.ndarray) and v.size == 1):
                state[k] = float(v)   # task(文字列)などは保存しない
        self._record("obs", {"step": step, "obs_t": obs.get_timestamp(), "state": state})

    def _record(self, kind, payload):
        import json

        with self._rec_lock:
            self._rec.write(json.dumps({"kind": kind, "t": time.time(), **payload}) + "\n")
            self._rec.flush()

    # --- 送信: 制御ループからは郵便受けに置くだけ ---------------------------------
    def send_observation(self, obs: TimedObservation) -> bool:
        with self._mailbox_cv:
            if self._mailbox is not None:
                self._stats.dropped += 1
                # must_go(キューが空なので必ず推論してほしい)は新しい観測に引き継ぐ
                obs.must_go = obs.must_go or self._mailbox.must_go
            self._mailbox = obs
            self._mailbox_cv.notify()
        return True

    def _sender_loop(self):
        while True:
            with self._mailbox_cv:
                while self._mailbox is None:
                    self._mailbox_cv.wait(timeout=0.5)
                    if not self.running and self._mailbox is None:
                        return
                obs, self._mailbox = self._mailbox, None
            encoded, nbytes = encode_images(obs.get_observation(), JPEG_QUALITY)
            obs = TimedObservation(
                timestamp=obs.get_timestamp(), observation=encoded, timestep=obs.get_timestep(), must_go=obs.must_go
            )
            t0 = time.perf_counter()
            if self.running:
                super().send_observation(obs)
            elapsed = time.perf_counter() - t0
            if self._rec is not None:
                try:
                    self._record_obs(obs, encoded)
                except Exception as e:  # 記録の失敗で送信が止まるとロボットが止まるので、記録だけ諦める
                    self.logger.warning(f"[fast] 記録に失敗: {e}")
            self._stats.add("send_ms", elapsed * 1000)
            self._stats.add("kb", nbytes / 1024)
            # 待っている間に郵便受けには最新の観測が溜まるので、次に送るのは常にその時点の最新
            if SEND_HZ > 0:
                time.sleep(max(0.0, 1.0 / SEND_HZ - elapsed))

    # --- 計測: 行動チャンクが届いた時点で観測からの経過時間を記録 ---------------------
    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        if incoming_actions:
            self._stats.add("age_ms", (time.time() - incoming_actions[0].get_timestamp()) * 1000)
        if SYNC_K > 0:
            return self._sync_accept(incoming_actions)
        return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def _sync_accept(self, incoming_actions):
        """止まった後に撮った観測から作られたチャンクだけを受け取り、先頭 SYNC_K 個で置き換える(混ぜない)。"""
        if not incoming_actions or not self._sync_waiting:
            return
        if incoming_actions[0].get_timestamp() < self._sync_stop_t + SYNC_SETTLE - 1e-3:
            return   # 動いている間の観測にもとづく古いチャンク
        q = Queue()
        for a in incoming_actions[:SYNC_K]:
            q.put(a)
        with self.action_queue_lock:
            self.action_queue = q
        self._sync_waiting = False

    def _sync_step(self, task: str, raw=None):
        """SYNC_K: キューが尽きたら止まり、揺れが収まってから観測を 1 枚送る(3 秒返事が無ければ送り直す)。

        raw: その周回で既に読んだ観測(16_orchestrator が渡す。カメラを 2 回読まないため)。
        """
        now = time.time()
        if not self._sync_waiting:
            self._sync_waiting, self._sync_stop_t, self._sync_sent_t = True, now, None
        if now < self._sync_stop_t + SYNC_SETTLE:
            return None
        if self._sync_sent_t is not None and now - self._sync_sent_t < 3.0:
            return None
        raw = dict(raw) if raw is not None else self.robot.get_observation()   # 渡された dict は画面表示でも使うので複製
        raw["task"] = task
        with self.latest_action_lock:
            latest_action = self.latest_action
        self.send_observation(TimedObservation(timestamp=time.time(), observation=raw,
                                               timestep=max(latest_action, 0), must_go=True))
        self._sync_sent_t = now
        return raw

    def control_loop(self, task: str, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting (fast client)")
        _performed_action = _captured_observation = None
        while self.running:
            t0 = time.perf_counter()
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
                if self._rec is not None and isinstance(_performed_action, dict):
                    self._record("act", {"action": {k: float(v) for k, v in _performed_action.items()}})
            if SYNC_K > 0:
                if not self.actions_available():
                    _captured_observation = self._sync_step(task) or _captured_observation
            elif self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)
            self._stats.add("loop_ms", (time.perf_counter() - t0) * 1000)
            if time.perf_counter() - self._last_report > REPORT_SEC:
                self.logger.info(f"[fast] {self._stats.summary()}")
                self._last_report = time.perf_counter()
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - t0)))
        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    import logging

    logging.info(pformat(asdict(cfg)))
    client = FastRobotClient(cfg)
    if client.start():
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            receiver.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
