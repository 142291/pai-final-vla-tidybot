"""実機評価を決まった手順で回し、結果を記録・集計する。

使い方(カメラ権限のあるターミナルから):
    .venv/bin/python scripts/14_eval.py sheet                 # 配置図 outputs/eval/positions.jpg を作る
    .venv/bin/python scripts/14_eval.py run <RUN> <CKPT>      # 例: run pi05_v3full 020000
    .venv/bin/python scripts/14_eval.py report                # これまでの結果を集計

手順(run):
  - 3 色 × 5 か所 = 15 試行。順序は固定シードで並べ替える(どのモデルも同じ順序・同じ配置)。
  - 各試行で「対象の色を P_k、残り 2 色を決まった位置」に置くよう表示する → Enter でロボットが動く。
  - TRIAL_SEC 秒(既定 40)たつか、動作中に結果(s/g/w/f/x)を打って Enter すると止まる。結果:
        s = 成功(対象を箱に入れた)
        g = 掴んだが箱に入らない
        w = 違う色を掴んだ / 入れた
        f = 掴めない
        x = 試行を無効にしてやり直す(置き間違い・通信切れなど)
  - 結果は outputs/eval/results.csv に追記。成功率は Wilson の 95% 信頼区間つきで出す。

配置(俯瞰カメラ 640x360 の座標。positions.jpg に番号を描く):
  P1 右上 / P2 右中 / P3 手前中央 / P4 左上(箱の上) / P5 中央

新しい物(OBJ=eraser など)の評価:
    OBJ=eraser SYNC_K=15 TRIAL_SEC=60 .venv/bin/python scripts/14_eval.py run <RUN> <CKPT>
  - 置き場所は 25_eraser_guide.py の eval_trials()(上の 5 か所 × 向き 2 通り = 10 試行。教える録画では使わない場所)。
  - 条件名は <RUN>+eraser になる(キューブの結果とは別に集計される)。教わる前のモデルでも同じ 10 試行を回し、前後を比べる。
"""

import csv
import datetime as dt
import math
import os
import random
import signal
import subprocess
import sys

import cv2

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "outputs", "eval")
CSV = os.path.join(OUT, "results.csv")
POS = {1: (520, 70), 2: (540, 230), 3: (360, 300), 4: (150, 90), 5: (390, 170)}
JA_COLOR = {"red": "赤", "yellow": "黄", "purple": "紫"}


def place_name(x, y):
    """俯瞰画像の座標を「左奥」「真ん中手前」のような呼び名にする(3×3 に区切る)。

    左右・奥手前は、俯瞰画像の向きに合わせる: 左 = 箱のある側、奥 = アームの付け根側、手前 = その反対。
    2026-10-25: P1〜P5 の番号では置く場所が分かりにくいという声を受けて変えた。
    """
    col = "左" if x < 213 else ("真ん中" if x < 427 else "右")
    row = "奥" if y < 120 else ("中央" if y < 240 else "手前")
    return col + row


def place(k):
    return place_name(*POS[k])
COLORS = ["red", "yellow", "purple"]
TASK = "Pick up the {} cube and put it in the box"
TRIAL_SEC = int(os.environ.get("TRIAL_SEC", "40"))
SEED = 20261019
OBJ = os.environ.get("OBJ", "")   # 例: eraser(新しい物の評価)
DRAW = {"red": (0, 0, 255), "yellow": (0, 220, 255), "purple": (220, 0, 200), "eraser": (255, 255, 255)}
JA_COLOR["eraser"] = "消しゴム"


def trials():
    """(色, 対象の位置, {残り 2 色: 位置}) の 15 試行。残りは対象の次の 2 か所に置く。"""
    out = []
    for c in COLORS:
        others = [o for o in COLORS if o != c]
        for k in POS:
            rest = {others[0]: (k % 5) + 1, others[1]: ((k + 1) % 5) + 1}
            out.append((c, k, rest))
    random.Random(SEED).shuffle(out)
    return out


def sheet():
    ref = os.path.join(ROOT, "outputs", "cam_test", "ref_top_so101_cube_v3_s1.jpg")
    img = cv2.resize(cv2.imread(ref), (640, 360))
    for k, (x, y) in POS.items():
        cv2.circle(img, (x, y), 16, (255, 255, 255), 2)
        cv2.putText(img, f"P{k}", (x - 14, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        print(f"P{k} = {place(k)}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "positions.jpg")
    cv2.imwrite(path, img)
    print(path)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def run_trial(run, ckpt, task, rec_dir=""):
    """ロボットを動かし、TRIAL_SEC 秒たつか結果が入力されたら止める。(結果 or None, 経過秒) を返す。

    動いている最中に s / g / w / f / x を打って Enter すると、その場で止めてその結果を使う
    (成功した瞬間に s を打てば待たずに次へ進める。経過秒は「成功までの時間」として記録する)。
    """
    import select
    import time

    env = dict(os.environ, RUN=run, CKPT=ckpt, TRIAL_REC_DIR=rec_dir)
    p = subprocess.Popen(["scripts/12_language_remote.sh", task], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"  動作中(最大 {TRIAL_SEC} 秒)。結果が決まったら s/g/w/f/x + Enter で即終了", flush=True)
    t0 = time.time()
    early = None
    try:
        while p.poll() is None and time.time() - t0 < TRIAL_SEC:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                key = sys.stdin.readline().strip().lower()
                if key in ("s", "g", "w", "f", "x"):
                    early = key
                    break
    except KeyboardInterrupt:
        print("  (Ctrl+C で停止)")
    elapsed = time.time() - t0
    stop(p)
    return early, elapsed


def kill_leftovers():
    """前回 Ctrl+C で止めたときに取り残されたクライアントを止める。

    2026-10-24: 評価を Ctrl+C で止めると、子のクライアント(別セッションで起動)が残ってアームとカメラを
    握ったままになり、次の評価で「何も動かない」状態になった。さらに残ったクライアントのログを
    warmup が「準備完了」と誤認していた。
    """
    import time

    out = subprocess.run(["pgrep", "-f", "remote/fast_client.py|16_orchestrator.py|12_language_remote.sh"],
                         capture_output=True, text=True).stdout.split()
    pids = [int(x) for x in out if int(x) != os.getpid()]
    if not pids:
        return
    print(f"取り残されたクライアントを停止します: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop(p):
    if p is not None and p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)   # クライアントに Ctrl+C を送る → トルクを切って終了
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def warmup(model_run, ckpt, limit=300):
    """モデルを H200 に読み込ませる(初回は 2 分前後かかる)。制御ループが始まったらすぐ止める。

    2026-10-19: これが無いと 1 試行目が読み込み中に 40 秒で打ち切られ、ロボットが動かないまま終わった。
    """
    import glob
    import time

    print("モデルを読み込み中(初回は約 2 分)。アームはホーム姿勢、キューブは作業範囲の外に置いておくこと...", flush=True)
    t0 = time.time()
    env = dict(os.environ, RUN=model_run, CKPT=ckpt)
    ready = False
    # 2026-10-19: 接続直後にモーター 4 番の書き込みが 1 回だけ失敗して落ちた(読み直すと 6 個とも正常)。
    # すぐ終わった場合は一時的な通信エラーとみなして 3 回まで起動し直す。
    for attempt in range(3):
        t1 = time.time()
        p = subprocess.Popen(["scripts/12_language_remote.sh", TASK.format("red")], cwd=ROOT, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            while time.time() - t0 < limit and p.poll() is None:
                time.sleep(2)
                # ファイル名の数字は作成時刻(秒)。この起動より前に作られたログ(取り残されたクライアント)は見ない
                logs = [l for l in glob.glob(os.path.join(ROOT, "logs", "robot_client_*.log"))
                        if int(os.path.basename(l)[13:-4]) >= int(t1) - 1]
                if any("Control loop thread starting" in open(l, errors="ignore").read() for l in logs):
                    ready = True
                    break
        finally:   # Ctrl+C で止めてもクライアントを残さない
            stop(p)
        if ready or time.time() - t0 >= limit:
            break
        print(f"  クライアントが途中で終了(ロボットとの通信エラーの可能性)。起動し直す ({attempt + 1}/3)", flush=True)
    print(f"  {'準備完了' if ready else '準備できず(回線・カメラ・H200 を確認)'}({time.time() - t0:.0f} 秒)")
    if not ready:
        sys.exit(1)


def place_preview(color, k, rest, i, n):
    return preview([(color, POS[k], True)] + [(o, POS[p], False) for o, p in rest.items()], i, n)


def preview(marks, i, n, note=""):
    """置き場所を俯瞰カメラの映像に丸で示し、スペース/Enter を待つ(評価ごとに同じ位置へ置くため)。

    marks は (物, (x, y), 対象か) の並び。note は画面の下に出す一言(向きなど、英数字のみ)。

    2026-10-26: 置き場所を言葉で伝えるだけだと回ごとにずれ、モデルの差と置き場所の差が混ざった
    (ラウンド 1 の評価が悪く見えたが、同じ夜に前のモデルも 1/5 だった)。録画(21_record_guided)と同じく
    映像の上に丸を出して、毎回同じ位置に置けるようにする。カメラは試行の前だけ開き、閉じてから試行に入る。
    PREVIEW=0 で無効(従来どおり文字の案内だけ)。
    """
    import time

    import numpy as np
    import yaml

    if os.environ.get("PREVIEW", "1") != "1":
        return True
    env = {}
    for line in open(os.path.join(ROOT, "config.env")):
        if "=" in line and not line.strip().startswith("#"):
            a, b = line.strip().replace("export ", "", 1).split("=", 1)
            env[a.strip()] = b.strip().strip("'\"")
    idx = yaml.safe_load(env["CAMERAS"])["top"]["index_or_path"]
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    draw = DRAW
    ok = True
    t_open = time.time()
    try:
        while True:
            got, frame = cap.read()
            if not got:
                time.sleep(0.05)
                continue
            frame = cv2.resize(frame, (640, 360))
            r = 13 + int(4 * (1 + np.sin(time.time() * 6)))
            for c, (x, y), target in marks:
                cv2.circle(frame, (x, y), r if target else 12, draw[c], 3 if target else 2)
                cv2.putText(frame, ("TARGET " if target else "") + c, (x - 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw[c], 2)
            cv2.putText(frame, f"[{i}/{n}] place objects on circles, arm at home -> SPACE  {note}", (8, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("place", frame)
            key = cv2.waitKey(30) & 0xFF
            # 2026-10-26: 開いた直後に、前に押したキー(結果入力の Enter など)が届いて、置く前に始まってしまった。
            # 開いてから 2 秒は無視し、開始はスペースだけにする
            if time.time() - t_open < 2.0:
                continue
            if key in (32, ord("x")):
                break
            if key == ord("q"):
                ok = False
                break
    finally:
        cap.release()
        cv2.destroyWindow("place")
        cv2.waitKey(1)
    return ok


def specs():
    """試行の並び。各要素は (対象, 指示文, 画面の丸, 案内の文, 記録する位置, 記録するほかの物, 画面の一言)。"""
    out = []
    if OBJ:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from importlib import import_module
        objects = import_module("objects")
        orient_en = {"横向き": "horizontal", "縦向き": "vertical", "斜め": "diagonal"}
        for n, x, y, orient, cubes in import_module("25_eraser_guide").eval_trials():
            marks = [(OBJ, (x, y), True)] + [(c, p, False) for c, p in cubes.items() if p != "box"]
            inbox = all(p == "box" for p in cubes.values())
            ctx = "キューブ 3 個は箱の中" if inbox else "、".join(f"{JA_COLOR[c]}を{place_name(*p)}" for c, p in cubes.items())
            others = "all cubes in box" if inbox else ", ".join(f"{c}@{p[0]},{p[1]}" for c, p in cubes.items())
            out.append((OBJ, objects.task(OBJ), marks, f"{JA_COLOR[OBJ]}を【{place_name(x, y)}】に{orient}で。{ctx}",
                        n, f"{orient_en[orient]}; {others}", f"{OBJ} {orient_en[orient]}"))
        return out
    for c, k, rest in trials():
        marks = [(c, POS[k], True)] + [(o, POS[p], False) for o, p in rest.items()]
        say = "、".join(f"{JA_COLOR[o]}を{place(p)}" for o, p in rest.items())
        others = ", ".join(f"{o}→P{p}" for o, p in rest.items())   # 記録用(results.csv の形式は変えない)
        out.append((c, TASK.format(c), marks, f"{JA_COLOR[c]}を【{place(k)}】に。ほかは {say}", k, others, ""))
    return out


def run(model_run, ckpt):
    # EVAL_TAG で条件名を付けて記録する(モデルは同じで動かし方だけ違う比較用。例: EVAL_TAG=steps3)
    label = model_run + (f"+{OBJ}" if OBJ else "") + (f"+{os.environ['EVAL_TAG']}" if os.environ.get("EVAL_TAG") else "")
    os.makedirs(OUT, exist_ok=True)
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "run", "ckpt", "trial", "color", "pos", "others", "result", "note"])
    ts = specs()
    kill_leftovers()
    if os.environ.get("SKIP_WARMUP") != "1":
        warmup(model_run, ckpt)
    print(f"モデル {model_run}/{ckpt}  条件 {label}  {len(ts)} 試行、1 試行 {TRIAL_SEC} 秒。配置図: outputs/eval/positions.jpg")
    # 途中で止めても続きから再開する(同じ run/ckpt で記録済みの試行は飛ばす)
    done = 0
    if not new:
        done = sum(1 for r in csv.DictReader(open(CSV)) if (r["run"], r["ckpt"]) == (label, ckpt))
    if done:
        print(f"記録済み {done} 試行 → {done + 1} 試行目から再開(最初からやり直すなら results.csv の該当行を消す)")
    # ONLY="2,3" のように指定すると、その試行だけをやり直す(置く前に始まった回などを撮り直すとき。
    # 先に results.csv から該当の行を消しておく)
    only = [int(x) - 1 for x in os.environ.get("ONLY", "").split(",") if x.strip()]
    order = only if only else list(range(done, len(ts)))
    for i in order:
        c, task, marks, guide_text, k, others, note_en = ts[i]
        print(f"\n[{i+1}/{len(ts)}] 対象: {guide_text}。"
              f"「place」画面の丸の位置に置き、アームをホーム姿勢にしてから、画面をクリックしてスペース(q で中断)")
        if not preview(marks, i + 1, len(ts), note_en):
            break
        if os.environ.get("PREVIEW", "1") != "1":
            input("  準備できたら Enter ")
        # 送った画像・関節角・実行した行動を試行ごとに保存する(失敗の仕方を後から確かめるため)
        rec = os.path.join(OUT, "rec", f"{label}_{ckpt}_t{i+1:02d}_{dt.datetime.now():%H%M%S}")
        while True:   # x(やり直し)なら同じ試行をもう一度
            r, elapsed = run_trial(model_run, ckpt, task, rec)
            while r not in ("s", "g", "w", "f", "x"):
                r = input("  結果 s=成功 g=掴んだが入らない w=違う物 f=掴めない x=やり直し: ").strip().lower()
            if r != "x":
                break
            if not preview(marks, i + 1, len(ts), note_en):
                f.close()
                return
            rec = os.path.join(OUT, "rec", f"{label}_{ckpt}_t{i+1:02d}_{dt.datetime.now():%H%M%S}")
        note = "" if r == "s" else input("  メモ(任意): ").strip()   # 成功ならメモは聞かずに次へ
        note = f"t={elapsed:.1f}s" + (f" {note}" if note else "")
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), label, ckpt, i + 1, c, k, others, r, note])
        f.flush()
    f.close()
    report()


def report():
    if not os.path.exists(CSV):
        print("結果なし")
        return
    rows = list(csv.DictReader(open(CSV)))
    models = sorted({(r["run"], r["ckpt"]) for r in rows})
    for m in models:
        rs = [r for r in rows if (r["run"], r["ckpt"]) == m]
        k = sum(r["result"] == "s" for r in rs)
        p, lo, hi = wilson(k, len(rs))
        by = " ".join(f"{c}:{sum(r['result']=='s' for r in rs if r['color']==c)}/{sum(r['color']==c for r in rs)}" for c in COLORS)
        kinds = {x: sum(r["result"] == x for r in rs) for x in "sgwf"}
        print(f"{m[0]}/{m[1]}: 成功 {k}/{len(rs)} = {p:.0%} [95%CI {lo:.0%}–{hi:.0%}]  色別 {by}  内訳 {kinds}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "sheet":
        sheet()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        report()
