"""観測 1 回のアップロード時間を「生配列」と「JPEG」で比べる(ロボット・カメラ不要)。

docs/reference-framing の基準画像を 640x360 の 2 台分の観測に仕立て、
本物の gRPC 経路(SendObservations)で fast_server に送る。サーバー側はキューに積むだけなので、
測れるのは純粋な転送時間。

使い方(先に SSH トンネル: ssh -fN -L 8124:127.0.0.1:8124 <H200>):
    python scripts/remote/bench_transport.py [--n 10] [--port 8124]
"""

import argparse
import os
import pickle  # nosec
import sys
import time

import cv2
import grpc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_common import encode_images  # noqa: E402

from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore  # noqa: E402
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load_frame(name):
    bgr = cv2.imread(os.path.join(ROOT, "docs", "reference-framing", f"{name}_reference.jpg"))
    return np.ascontiguousarray(cv2.resize(bgr, (640, 360))[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--port", type=int, default=8124)
    args = ap.parse_args()

    obs = {f"{m}.pos": 0.0 for m in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}
    obs["base_0_rgb"] = load_frame("top")
    obs["left_wrist_0_rgb"] = load_frame("wrist")
    obs["task"] = "bench"

    stub = services_pb2_grpc.AsyncInferenceStub(
        grpc.insecure_channel(f"127.0.0.1:{args.port}", grpc_channel_options())
    )
    stub.Ready(services_pb2.Empty())

    for label, quality in [("raw", None), ("jpeg q90", 90), ("jpeg q80", 80)]:
        times, size = [], 0
        for i in range(args.n):
            payload = obs if quality is None else encode_images(obs, quality)[0]
            data = pickle.dumps(TimedObservation(timestamp=time.time(), observation=payload, timestep=i))
            size = len(data)
            t0 = time.perf_counter()
            stub.SendObservations(send_bytes_in_chunks(data, services_pb2.Observation, silent=True))
            times.append((time.perf_counter() - t0) * 1000)
        print(f"{label:9s} {size/1024:7.0f} KB | send p50 {np.percentile(times, 50):6.0f}ms  p95 {np.percentile(times, 95):6.0f}ms")


if __name__ == "__main__":
    main()
