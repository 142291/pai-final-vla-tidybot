"""遠隔推論の観測を JPEG で圧縮/復元する共通処理(fast_client.py / fast_server.py から使う)。

lerobot の robot_client は観測を pickle の生配列で送る。640x360 x 2 台で 1.38MB あり、
上り 2MB/s では送るだけで 0.7 秒以上かかる。しかも送信は制御ループと同じスレッドで
同期的に行われるため、その間ロボットに指令が出ない(2026-10-01 のログで 1 周 3.5〜4.3 秒)。

画像は**元の解像度のまま** JPEG にする。π0.5 の前処理は学習時と同じく 640x360 から
アスペクト比を保って 224 に縮小するので、ここで縮小すると学習時と幾何が変わってしまう。
"""

from __future__ import annotations

import cv2
import numpy as np

JPEG_KEY = "__jpeg__"


def encode_images(observation: dict, quality: int) -> tuple[dict, int]:
    """観測 dict の画像(HxWx3 uint8)を JPEG に置き換えた新しい dict と、画像の合計バイト数を返す。"""
    out, total = {}, 0
    for k, v in observation.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3 and v.dtype == np.uint8:
            # lerobot のカメラは RGB を返す。cv2 は BGR 前提なので入れ替えてから符号化する
            ok, buf = cv2.imencode(".jpg", v[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {k}")
            data = buf.tobytes()
            out[k] = {JPEG_KEY: data}
            total += len(data)
        else:
            out[k] = v
    return out, total


def decode_images(observation: dict) -> dict:
    """encode_images の逆。JPEG でない値はそのまま返す(従来のクライアントとも互換)。"""
    out = {}
    for k, v in observation.items():
        if isinstance(v, dict) and JPEG_KEY in v:
            bgr = cv2.imdecode(np.frombuffer(v[JPEG_KEY], dtype=np.uint8), cv2.IMREAD_COLOR)
            out[k] = np.ascontiguousarray(bgr[:, :, ::-1])
        else:
            out[k] = v
    return out
