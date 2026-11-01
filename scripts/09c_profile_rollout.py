"""ロールアウト制御ループの区間別プロファイラ。

lerobot-rollout と同じ引数をそのまま受け取り、制御ループの各区間に
ストップウォッチを仕込んで終了時に集計を出す。
制御ループが目標 30Hz に対して 8-11Hz しか出ない原因の切り分け用。
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

# ラベル -> 所要時間(ms)のリスト
STATS: dict[str, list[float]] = defaultdict(list)
# get_observation 開始時刻の列。隣接差分がループ周期になる
LOOP_MARKS: list[float] = []


def _timed(label, fn, mark_loop=False):
    """fn を包んで所要時間を STATS[label] に積む。"""

    def wrapper(*args, **kwargs):
        if mark_loop:
            LOOP_MARKS.append(time.perf_counter())
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            STATS[label].append((time.perf_counter() - t0) * 1000.0)

    return wrapper


def _install():
    """lerobot 内部のループ構成要素を差し替える。"""
    from lerobot.rollout import robot_wrapper as rw
    from lerobot.rollout.inference import rtc, sync
    from lerobot.rollout.strategies import base, core
    from lerobot.utils import robot_utils

    # 1. モーター+カメラの読み取り(ループの先頭。ここでループ周期も記録)
    rw.ThreadSafeRobot.get_observation = _timed(
        "1 get_observation", rw.ThreadSafeRobot.get_observation, mark_loop=True
    )
    # 2. 観測前処理 + エンジンへの通知
    core.RolloutStrategy._process_observation_and_notify = _timed(
        "2 obs_processor+notify", core.RolloutStrategy._process_observation_and_notify
    )
    # 3. 推論本体
    sync.SyncInferenceEngine.get_action = _timed(
        "3 engine.get_action", sync.SyncInferenceEngine.get_action
    )
    rtc.RTCInferenceEngine.get_action = _timed(
        "3 engine.get_action", rtc.RTCInferenceEngine.get_action
    )
    # 4. フォロワーへの書き込み(内部で Present_Position の sync_read もしている)
    rw.ThreadSafeRobot.send_action = _timed("4 send_action", rw.ThreadSafeRobot.send_action)
    # 5. 目標周期までの待機。ループが速すぎる場合のみ呼ばれる
    base.precise_sleep = _timed("5 precise_sleep", robot_utils.precise_sleep)
    # A. 3+4+補間+action_processor の合計。1〜5 の和との差が「未計測の隙間」
    base.send_next_action = _timed("A send_next_action 合計", core.send_next_action)


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def _report():
    print("\n" + "=" * 68)
    print("制御ループ区間別プロファイル (ms)")
    print("=" * 68)
    print(f"{'区間':<26}{'回数':>6}{'平均':>9}{'中央':>9}{'p95':>9}{'最大':>9}")
    print("-" * 68)
    for label in sorted(STATS):
        xs = STATS[label]
        print(
            f"{label:<26}{len(xs):>6}{sum(xs) / len(xs):>9.1f}"
            f"{_pct(xs, 0.5):>9.1f}{_pct(xs, 0.95):>9.1f}{max(xs):>9.1f}"
        )
    print("-" * 68)

    # 1〜5 の中央値を足して、実測ループ周期と比べる
    parts = ["1 get_observation", "2 obs_processor+notify", "A send_next_action 合計", "5 precise_sleep"]
    total = sum(_pct(STATS[p], 0.5) for p in parts if STATS.get(p))
    print(f"{'上記 1,2,A,5 の中央値合計':<26}{'':>6}{'':>9}{total:>9.1f}")

    if len(LOOP_MARKS) > 2:
        periods = [
            (LOOP_MARKS[i + 1] - LOOP_MARKS[i]) * 1000.0 for i in range(len(LOOP_MARKS) - 1)
        ]
        # 最初の数回は起動処理を含むので捨てる
        periods = periods[3:] or periods
        med = _pct(periods, 0.5)
        print(f"{'実測ループ周期':<26}{len(periods):>6}"
              f"{sum(periods) / len(periods):>9.1f}{med:>9.1f}"
              f"{_pct(periods, 0.95):>9.1f}{max(periods):>9.1f}")
        if med > 0:
            print(f"\n実測 {1000.0 / med:.1f} Hz")
        gap = med - total
        print(f"未計測の隙間: 中央値で {gap:.1f} ms")
        if gap > 5:
            print("→ 計測した区間の外に原因がある(GC / スレッド競合 / 計測漏れ)")
        else:
            print("→ 上の表で最も大きい区間が原因")
    print("=" * 68)


def main():
    _install()
    from lerobot.scripts.lerobot_rollout import main as rollout_main

    try:
        rollout_main()
    finally:
        _report()


if __name__ == "__main__":
    sys.exit(main())
