"""JPEG 圧縮した観測を受け取れる policy_server(H200 側で起動する)。

使い方(引数は lerobot.async_inference.policy_server と同じ):
    python scripts/remote/fast_server.py --host=127.0.0.1 --port=8124

圧縮していない従来クライアントの観測もそのまま受け付ける。
"""

import logging
import os
import pickle  # nosec
import sys
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_common import decode_images  # noqa: E402

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore  # noqa: E402


PI05_STEPS = int(os.environ.get("PI05_STEPS", "0"))  # π0.5 の flow matching 反復回数(0 なら学習時の値=10)


class FastPolicyServer(PolicyServer):
    """2026-10-19: 反復回数を 10→3 にしても予測精度は変わらず(学習に使っていない回で誤差 3.2→3.1)、
    推論は 357→218ms。遅延が短いほど予測の前半(当たる部分)を実行できるので、PI05_STEPS=3 で起動する。"""
    _loaded_key = None

    def _apply_steps(self):
        if PI05_STEPS and hasattr(getattr(self, "policy", None), "config") and hasattr(self.policy.config, "num_inference_steps"):
            self.policy.config.num_inference_steps = PI05_STEPS
            self.logger.info(f"[fast] num_inference_steps = {PI05_STEPS}")

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """同じモデルなら読み込み直さない。

        標準の実装はクライアントが繋ぐたびに from_pretrained し直すので、π0.5 では
        1 回 1.5〜2 分かかる(2026-10-17 実測)。評価で何十回も走らせると致命的なので、
        種類・パス・デバイスが同じなら状態だけ更新して使い回す。
        """
        specs = pickle.loads(request.data)  # nosec
        key = (specs.policy_type, specs.pretrained_name_or_path, specs.device)
        if self.running and key == self._loaded_key and getattr(self, "policy", None) is not None:
            self.lerobot_features = specs.lerobot_features
            self.actions_per_chunk = specs.actions_per_chunk
            self.last_processed_obs = None
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            self._apply_steps()
            self.logger.info(f"[fast] 読み込み済みのモデルを再利用: {specs.pretrained_name_or_path}")
            return services_pb2.Empty()
        result = super().SendPolicyInstructions(request, context)
        if getattr(self, "policy", None) is not None:
            self._loaded_key = key
            self._apply_steps()
        return result

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        t0 = time.perf_counter()
        obs.observation = decode_images(obs.get_observation())
        decode_ms = (time.perf_counter() - t0) * 1000
        if obs.get_timestep() % 30 == 0:
            self.logger.info(f"[fast] obs #{obs.get_timestep()} JPEG decode {decode_ms:.1f}ms")
        return super()._enqueue_observation(obs)


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = FastPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"FastPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
