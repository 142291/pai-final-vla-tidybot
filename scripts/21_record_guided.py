"""ラウンド 1 の追加録画を、1 本ずつ「どこに何を置くか」を画面と声で案内しながら撮る。

使い方(カメラ権限のあるターミナルから。リーダーとフォロワーをホーム姿勢にしてから):
    .venv/bin/python scripts/21_record_guided.py            # 赤 15 本 → 黄 15 本 → 紫 15 本
    .venv/bin/python scripts/21_record_guided.py yellow     # 黄から(途中で止めても、撮り終えた本は飛ばして続きから)

画面(record ウィンドウ)をクリックしてからキーで操作する:
    スペース … 待機中: 録画を始める / 録画中: 成功したので保存して次へ(キューブを箱で放した直後に押す)
    r       … 録画中: この本を捨てて撮り直す
    q       … 終了(撮り終えた本は保存済み)

1 本の流れ:
    1. 画面の俯瞰映像に、置き場所が丸で出る。声でも「3 本目。赤を真ん中中央へ。紫を箱に入れておく」と言う。
    2. キューブを丸の位置に置き、指示があればほかの色を箱に入れる。アームはホーム姿勢に戻す。
    3. スペースで録画開始 → リーダーで掴んで箱に入れる(真上から下ろして掴む)→ 放したらスペース。
    4. 次の本の案内が出る。
フォロワーは録画していない間もリーダーに追従する(戻すのもリーダーで行う)。
保存先は data/so101_cube_v3_r1(学習データと同じ形式・同じ符号化)。進み具合は outputs/record_r1/progress.json。

新しい物を教える(OBJ=eraser など。置き場所は 25_eraser_guide.py、30 本):
    OBJ=eraser .venv/bin/python scripts/21_record_guided.py
    保存先は data/so101_eraser、進み具合は outputs/record_eraser/progress.json。

2026-10-25: lerobot-record では今何本目で何をすればよいか分からないという声を受けて作った。
置き場所の決め方は 20_round1_guide.py を参照。
2026-10-28: 消しゴムを教える録画に使うため、OBJ で物を切り替えられるようにした。
"""

import json
import os
import subprocess
import sys
import time
from importlib import import_module

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
objects = import_module("objects")
OBJ = os.environ.get("OBJ", "")
if OBJ:
    guide = import_module("25_eraser_guide")
    NAME = os.environ.get("REC_NAME", f"so101_{OBJ}")
    PROGRESS = os.path.join(ROOT, "outputs", f"record_{OBJ}", "progress.json")
    COLORS = [OBJ]
else:
    guide = import_module("20_round1_guide")
    NAME = os.environ.get("REC_NAME", "so101_cube_v3_r1")
    PROGRESS = os.path.join(ROOT, "outputs", "record_r1", "progress.json")
    COLORS = ["red", "yellow", "purple"]
MAX_SEC = float(os.environ.get("MAX_SEC", "40"))
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
JA = objects.JA
DRAW = {k: v["draw"] for k, v in objects.OBJECTS.items()}


def say(text):
    try:
        subprocess.Popen(["say", "-v", "Kyoko", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, "config.env")):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.replace("export ", "", 1).split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")
    return env


def connect():
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    env = load_env()
    cams = {k: OpenCVCameraConfig(index_or_path=v["index_or_path"], width=v["width"], height=v["height"],
                                  fps=v["fps"], fourcc=v.get("fourcc"))
            for k, v in yaml.safe_load(env["CAMERAS"]).items()}
    leader = SO101Leader(SO101LeaderConfig(port=env["LEADER_PORT"], id=env["LEADER_ID"], use_degrees=False))
    leader.connect()   # 録画(06_record)と同じく、先にリーダーをつなぐ
    robot = SO101Follower(SO101FollowerConfig(port=env["FOLLOWER_PORT"], id=env["FOLLOWER_ID"], use_degrees=False,
                                              cameras=cams))
    robot.connect()
    return robot, leader


def open_dataset():
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = os.path.join(ROOT, "data", NAME)
    enc = dict(rgb_encoder=RGBEncoderConfig(vcodec=os.environ.get("VCODEC", "h264_videotoolbox"), crf=23),
               streaming_encoding=True, encoder_threads=4)
    if os.path.exists(os.path.join(root, "meta", "info.json")):
        return LeRobotDataset.resume(f"local/{NAME}", root=root, **enc)
    ref = json.load(open(os.path.join(ROOT, "data", "so101_cube_v3_train", "meta", "info.json")))["features"]
    feats = {k: dict(v, shape=tuple(v["shape"])) for k, v in ref.items()
             if k in ("action", "observation.state", "observation.images.top", "observation.images.wrist")}
    return LeRobotDataset.create(f"local/{NAME}", 30, feats, root=root, robot_type="so_follower", use_videos=True, **enc)


def load_progress():
    try:
        return json.load(open(PROGRESS))
    except (OSError, ValueError):
        return {c: [] for c in COLORS}


def save_progress(p):
    os.makedirs(os.path.dirname(PROGRESS), exist_ok=True)
    json.dump(p, open(PROGRESS, "w"), ensure_ascii=False, indent=1)


class Screen:
    def __init__(self):
        self.big = ImageFont.truetype(FONT, 30)
        self.mid = ImageFont.truetype(FONT, 22)
        self.small = ImageFont.truetype(FONT, 16)
        self._key, self._panel = None, None

    def panel(self, lines):
        if lines != self._key:
            img = Image.new("RGB", (1280, 170))
            d = ImageDraw.Draw(img)
            d.text((14, 6), lines[0], font=self.big, fill=(255, 255, 255))
            d.text((14, 48), lines[1], font=self.big, fill=(255, 230, 120))
            d.text((14, 92), lines[2], font=self.mid, fill=(180, 220, 255))
            d.text((14, 128), lines[3], font=self.mid, fill=(255, 170, 170))
            d.text((930, 146), "スペース 開始/保存   r 撮り直し   q 終了", font=self.small, fill=(150, 150, 150))
            self._panel = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._key = lines
        return self._panel

    def show(self, obs, target, color, lines):
        top = cv2.cvtColor(obs["top"], cv2.COLOR_RGB2BGR)
        wrist = cv2.cvtColor(obs["wrist"], cv2.COLOR_RGB2BGR)
        x, y = target
        r = 14 + int(4 * (1 + np.sin(time.time() * 6)))   # 点滅して目立たせる
        cv2.circle(top, (x, y), r, DRAW[color], 3)
        cv2.circle(top, (x, y), 3, (255, 255, 255), -1)
        frame = np.vstack([np.hstack([cv2.resize(top, (640, 360)), cv2.resize(wrist, (640, 360))]), self.panel(lines)])
        cv2.imshow("record", frame)
        return cv2.waitKey(1) & 0xFF


def main(start_color=None):
    progress = load_progress()
    order = COLORS[COLORS.index(start_color):] if start_color else COLORS
    todo = [(c, row) for c in order for row in guide.plan(c) if row[0] not in progress.get(c, [])]
    per = {c: len(guide.plan(c)) for c in COLORS}
    total = sum(per.values())
    if not todo:
        print("全部撮り終えています(outputs/record_r1/progress.json)")
        return
    print(f"残り {len(todo)} 本。フォロワーとリーダーをホーム姿勢にしておくこと。")
    robot, leader = connect()
    ds = open_dataset()
    names = ds.features["action"]["names"]
    scr = Screen()
    i = 0
    state = "wait"
    t_rec = None
    announced = None
    tick = 0
    try:
        while i < len(todo):
            t0 = time.perf_counter()
            color, (n, x, y, place, box) = todo[i]
            if announced != (color, n):
                where = place if OBJ else place.split('(')[0]   # 新しい物は向き(横・縦・斜め)も読み上げる
                say(f"{JA[color]}、{n}本目。{JA[color]}を{where}へ。{box}。")
                announced = (color, n)
            obs = robot.get_observation()
            act = leader.get_action()
            sent = robot.send_action(act)
            if state == "rec":
                ds.add_frame({
                    "observation.state": np.array([obs[k] for k in names], dtype=np.float32),
                    "observation.images.top": obs["top"],
                    "observation.images.wrist": obs["wrist"],
                    "action": np.array([sent[k] for k in names], dtype=np.float32),
                    "task": objects.task(color),
                })
            done = sum(len(v) for v in progress.values())
            head = f"{JA[color]} {n}/{per[color]} 本目(全体 {done + 1}/{total})  置き場所: {place}"
            if state == "wait":
                lines = (head, f"箱: {box}", f"{JA[color]}を丸の位置に置き、アームをホームへ → スペースで録画開始",
                         "真上まで運んでから、まっすぐ下ろして掴む")
            else:
                el = time.time() - t_rec
                hz = len(ds.writer.episode_buffer["timestamp"]) / max(el, 1e-3) if hasattr(ds, "writer") and getattr(ds.writer, "episode_buffer", None) else 0
                lines = (head, f"箱: {box}", f"● 録画中 {el:4.1f} 秒({hz:.0f}Hz)  箱の中で放したらスペース(撮り直しは r)",
                         "真上まで運んでから、まっすぐ下ろして掴む")
                if el > MAX_SEC:
                    ds.clear_episode_buffer()
                    state = "wait"
                    say("長すぎるので撮り直します。")
            # 画面は 2 周に 1 回だけ描く(lerobot-record は表示が重く 3.6〜14Hz まで落ちていた。録画は 30Hz を保ちたい)
            tick += 1
            key = scr.show(obs, (x, y), color, lines) if tick % 2 == 0 else 255
            if key in (ord(" "), 13):
                if state == "wait":
                    state, t_rec = "rec", time.time()
                    say("録画開始")
                elif time.time() - t_rec > 1.5:    # 押した直後の二度押しで保存しないように
                    ds.save_episode()
                    progress.setdefault(color, []).append(n)
                    save_progress(progress)
                    state = "wait"
                    i += 1
                    say("保存しました")
            elif key == ord("r") and state == "rec":
                ds.clear_episode_buffer()
                state = "wait"
                announced = None
                say("撮り直します")
            elif key == ord("q"):
                if state == "rec":
                    ds.clear_episode_buffer()
                print("q が押されたので終了します")
                break
            time.sleep(max(0, 1 / 30 - (time.perf_counter() - t0)))
    except Exception as e:
        print(f"エラーで終了: {type(e).__name__}: {e}")
        raise
    finally:
        ds.finalize()
        cv2.destroyAllWindows()
        robot.disconnect()
        leader.disconnect()
        print(f"保存: data/{NAME}  進み具合: {json.dumps({c: len(v) for c, v in progress.items()}, ensure_ascii=False)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
