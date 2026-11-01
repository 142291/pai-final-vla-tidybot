"""提出用のデモ動画(3 分以内)を outputs/video/final.mp4 に書き出す。

使い方(matplotlib と Pillow が入った Python と ffmpeg で動かす):
    python3 scripts/27_final_video.py            # 全部作る
    python3 scripts/27_final_video.py 3 7        # 3 番目と 7 番目の場面だけ作り直してつなぐ

構成は「VLA はなぜ掴めないのか」(docs/submission/report と同じ筋)。
  つかみ(紫以外を片付けて)→ 題 → 同じ置き場所で非同期と止まって考えるを並べる → 切り分け(予測の先ほど外れる)→ 解決(止まって考える)
  → データと対照実験 → 言葉で頼む(片付けて)→ 断る(緑)→ 失敗の説明と人への引き継ぎ(赤だけ)→ まとめ

素材:
  outputs/video/phone/*.mp4   スマホで横から撮った映像(位置情報などのメタデータは消してある)
  outputs/eval/rec/<試行>/     評価で推論に使った画像(謎の場面)
  docs/submission/figures/    説明資料の図
スマホの映像の時刻は、ロボットの声の位置から events.jsonl の時刻に合わせた(SYNC の値、秒)。
倍速の区間には画面左上に「n×」を出す。運営のジングルは JINGLE_OPEN / JINGLE_END に動画のパスを渡すと前後に付く。
"""

import glob
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PHONE = os.path.join(ROOT, "outputs", "video", "phone")
REC = os.path.join(ROOT, "outputs", "eval", "rec")
FIG = os.path.join(ROOT, "docs", "submission", "figures")
OUT = os.path.join(ROOT, "outputs", "video")
WORK = os.path.join(OUT, "work")
W, H, FPS = 1920, 1080, 30

FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "Hiragino Kaku Gothic Pro", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
})

# スマホの映像の時刻 = events.jsonl の時刻(開始からの秒)+ SYNC
SYNC = {"A_except_purple.mp4": 1.3, "B1_tidy_all.mp4": 2.2, "D_red_only_human.mp4": 15.3}


# ---------------------------------------------------------------- 文字の画像

def font(path, size):
    return ImageFont.truetype(path, size)


def caption_png(path, ja, en, who=None):
    """画面下の字幕(日本語+英語)。who を渡すと「ロボット」などの話し手を色付きで前に付ける。"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fj, fe, fw = font(FONT_B, 44), font(FONT_R, 30), font(FONT_B, 44)
    lines = ja.split("\n")
    pad, gap = 22, 10
    hj = 58 * len(lines)
    he = 40 * len(en.split("\n")) if en else 0
    top = H - 60 - hj - he - pad * 2 - (gap if en else 0)
    d.rectangle([0, top, W, H - 60 + 0], fill=(0, 0, 0, 165))
    y = top + pad
    for i, line in enumerate(lines):
        x = 80
        if who and i == 0:
            d.text((x, y), who, font=fw, fill=(255, 200, 90, 255))
            x += int(d.textlength(who, font=fw)) + 18
        d.text((x, y), line, font=fj, fill=(255, 255, 255, 255))
        y += 58
    if en:
        y += gap
        for line in en.split("\n"):
            d.text((80, y), line, font=fe, fill=(205, 205, 200, 255))
            y += 40
    im.save(path)


def tag_png(path, text, sub=""):
    """画面左上の小さな札(倍速の表示、素材の説明)。"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f, fs = font(FONT_B, 40), font(FONT_R, 26)
    w = max(d.textlength(text, font=f), d.textlength(sub, font=fs) if sub else 0)
    h = 64 + (38 if sub else 0)
    d.rounded_rectangle([40, 40, 40 + w + 44, 40 + h], radius=12, fill=(0, 0, 0, 170))
    d.text((62, 50), text, font=f, fill=(255, 210, 80, 255))
    if sub:
        d.text((62, 102), sub, font=fs, fill=(220, 220, 215, 255))
    im.save(path)


def card_png(path, title, sub, body=""):
    im = Image.new("RGB", (W, H), "#111111")
    d = ImageDraw.Draw(im)
    d.text((140, 330), title, font=font(FONT_B, 92 if len(title) <= 16 else 68), fill="#f0efec")
    d.text((144, 470), sub, font=font(FONT_R, 40), fill="#7fb2ee")
    y = 580
    for line in body.split("\n") if body else []:
        d.text((144, y), line, font=font(FONT_R, 38), fill="#c3c2b7")
        y += 62
    im.save(path)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / 2 / n) / den
    h = z * ((p * (1 - p) / n + z * z / 4 / n / n) ** 0.5) / den
    return c - h, c + h


def bars_png(path, title, bars, note=""):
    """成功率の棒グラフ(95% 信頼区間つき)を動画の画面全体の大きさで描く。字幕の分、下を空ける。"""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    fig.text(0.06, 0.90, title, fontsize=34, color=INK, va="center")
    if note:
        fig.text(0.06, 0.84, note, fontsize=20, color=INK2, va="center")
    ax = fig.add_axes([0.10, 0.30, 0.82, 0.48], facecolor=SURFACE)
    for x, (label, k, n, col) in enumerate(bars):
        p = k / n
        lo, hi = wilson(k, n)
        ax.bar(x, p * 100, width=0.56, color=col, zorder=2)
        ax.errorbar(x, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], color=INK2, capsize=8, lw=2, zorder=3)
        ax.text(x, 4, f"{k}/{n}", ha="center", va="bottom", fontsize=22, color="white", zorder=4)
        ax.text(x, hi * 100 + 3, f"{p * 100:.0f}%", ha="center", va="bottom", fontsize=30, color=INK, zorder=4)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=21)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", color="#e6e5e1", lw=1, zorder=0)
    ax.set_ylabel("成功率(%) 縦線は 95% 信頼区間", fontsize=18)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def image_card_png(path, img, title):
    """説明資料の図を、見出し付きで画面に収める(下は字幕のために空ける)。"""
    im = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    d.text((115, 50), title, font=font(FONT_B, 46), fill=INK)
    fig = Image.open(img).convert("RGB")
    box_w, box_h = W - 160, 720
    s = min(box_w / fig.width, box_h / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    im.paste(fig, ((W - fig.width) // 2, 130))
    im.save(path)


# ---------------------------------------------------------------- 場面

def ph(src, a, b, speed=1.0):
    return {"src": src, "a": a, "b": b, "speed": speed}


def S(name, **kw):
    kw["name"] = name
    return kw


# 字幕は (場面の中での開始秒, 終了秒, 日本語, 英語, 話し手)
SCENES = [
    S("hook", clip=ph("A_except_purple.mp4", 6.0, 21.6), caps=[
        (0, 1.3, "指示「紫以外を片付けて」", "Instruction: \"Tidy up everything except purple\"", None),
        (1.3, 6.2, "紫以外の、赤と黄色のキューブを箱に入れますね。", "\"I'll put the red and yellow cubes in the box.\"", "ロボット"),
        (6.2, 15.6, "実機・1 倍速。自分で集めた手本で π0.5 を学習させた", "Real robot, real time. π0.5 fine-tuned on my own demonstrations", None),
    ]),
    S("title", card=("VLA はなぜ掴めないのか", "Why couldn't my VLA grasp? — tracking down the causes on a real SO-101",
                     "原因を一つずつ突き止めて直し、言葉で頼める片付けロボットにした記録"), dur=4),
    S("compare", pair=(ph("E_async.mp4", 8.0, 27.0), ph("E_sync.mp4", 8.4, 27.4)),
      labels=(("標準の実行(非同期)", "Standard async execution"), ("止まって考える実行", "Stop-and-think execution")), caps=[
        (0, 6.5, "同じモデル・同じ置き場所。右は実行の仕方だけを変えた",
         "Same model, same placement — the right side changes only how actions are executed", None),
        (6.5, 13, "左はキューブのすぐ横まで行って、止まったまま掴めない",
         "Left: it reaches the cube, then stalls without grasping", None),
        (13, 19, "右は 1 回目で箱へ。左は 60 秒たっても掴めなかった。なぜか?",
         "Right: in the box on the first try. Left: still nothing after 60 s. Why?", None),
    ]),
    S("diagnose", image=("fig_horizon.png", "切り分け: 予測の何ステップ先ほど外れるか"), dur=13, caps=[
        (0, 6.5, "人の操作と比べると、予測は先のステップほど外れる(誤差 3.3 → 13.1)",
         "Against human teleoperation, predictions drift further out (error 3.3 → 13.1)", None),
        (6.5, 13, "推論の往復に約 1 秒。その間に、外れやすい予測の後半を実行していた",
         "With a ~1 s inference round trip, the robot was executing the least accurate tail of each chunk", None),
    ]),
    S("fix", bars=("解決: 止まって考える実行", "同じモデル・決めた同じ 15 か所", [
        ("標準(非同期)\n10/31", 5, 15, MUTED), ("止まって考える\n10/28 と 10/31", 26, 30, BLUE)]), dur=10, caps=[
        (0, 5, "先頭の 15 ステップ(0.5 秒)だけ動かして止まり、最新の画像で予測し直す",
         "Stop-and-think: run only the first 15 steps (0.5 s), stop, and re-plan from a fresh image", None),
        (5, 10, "決めた 15 か所でも 33% → 87%(Fisher の正確検定 p < 0.001)",
         "Changing only how it executes: 33% → 87% (Fisher's exact test, p < 0.001)", None),
    ]),
    S("data", bars=("データの穴と、対照実験", "決めた同じ 15 か所での成功率", [
        ("R0\n150 本", 8, 15, BLUE), ("R1\n+箱に入った\n状態 45 本", 10, 15, BLUE), ("R2\n+人の手助け", 12, 15, BLUE),
        ("R3\n+人の手助け", 12, 15, BLUE), ("対照\nR1 のまま\n学習を延長", 14, 15, ORANGE)]), dur=12, caps=[
        (0, 6, "箱にキューブが入った状態から始まる手本が 0 本だった → 45 本足すと、止まらずに動くように",
         "No demos started with a cube already in the box — adding 45 of them got it moving again", None),
        (6, 12, "人の手助けを学習に戻す効果は確認できず。伸びの正体は学習不足の解消だった",
         "Feeding human assists back showed no measurable gain — the gain came from training longer (control: 93%)", None),
    ]),
    S("tidy", clip=ph("B1_tidy_all.mp4", 8.0, 26.0), caps=[
        (0, 0.4, "指示「片付けて」", "Instruction: \"Tidy up\"", None),
        (0.4, 4.5, "紫、黄、赤のキューブを順に箱へ片付けます。", "\"I'll put the purple, yellow and red cubes in the box, in that order.\"", "ロボット"),
        (4.5, 14, "段取りは Claude が俯瞰の画像を見て決める。動かすのは π0.5", "Claude plans from the overhead image; π0.5 drives the arm", None),
        (14, 18, "俯瞰カメラで判定: 箱の中で 1 秒止まったら成功", "The overhead camera checks: cube resting in the box for 1 s = success", None),
    ]),
    S("tidy_fast", clip=ph("B1_tidy_all.mp4", 26.0, 79.0, 4.0), tag=("4× 速送り", "4× speed"), caps=[
        (0, 13.25, "残りの 2 色も 1 回目で成功", "The other two cubes also go in on the first try", None),
    ]),
    S("refuse", optional="C_green.mp4", clip=ph("C_green.mp4", 0.0, 11.8), caps=[
        (0, 6.0, "指示「緑のを入れて」(机にあるのは赤・黄・紫だけ)", "Instruction: \"Put the green one in\" (only red, yellow and purple on the table)", None),
        (6.0, 10.0, "すみません、緑のものは見当たらないので入れられません。", "\"Sorry, I don't see anything green, so I can't put it in.\"", "ロボット"),
        (10.0, 11.8, "机の上を見て、できないことは断る", "It checks the table and declines what it can't do", None),
    ]),
    S("explain", clip=ph("D_red_only_human.mp4", 72.0, 80.0), caps=[
        (0, 8, "「赤だけ入れて」。苦手な置き場所(真ん中)では、すぐ横まで行って掴めない",
         "\"Put only the red one in.\" At a hard placement it gets right next to the cube and misses", None),
    ]),
    S("explain2", clip=ph("D_red_only_human.mp4", 85.8, 94.0), caps=[
        (0, 8.2, "赤のすぐ横まで行けましたが、掴めませんでした。もう一度やってみます。",
         "\"I got right next to the red cube but couldn't grasp it. I'll try again.\"", "ロボット"),
    ], tag=("失敗の理由は Claude が試行中の画像から判断", "Claude explains the failure from frames of the attempt")),
    S("call", clip=ph("D_red_only_human.mp4", 219.6, 229.8), caps=[
        (0, 1.6, "(2 回目・3 回目も同じ失敗。省略)", "(Attempts 2 and 3 fail the same way — skipped)", None),
        (1.6, 10.2, "…掴めませんでした。うまく掴めません。リーダーアームで手伝ってください。",
         "\"...couldn't grasp it. I can't get it. Please help me with the leader arm.\"", "ロボット"),
    ]),
    S("assist", clip=ph("D_red_only_human.mp4", 229.8, 279.5, 4.0), tag=("4× 速送り", "4× speed"), caps=[
        (0, 12.5, "人がリーダーアームで引き取る。その操作は学習データとして保存する", "A human takes over with the leader arm; the assist is saved as training data", None),
    ]),
    S("done", clip=ph("D_red_only_human.mp4", 279.5, 284.5), caps=[
        (0, 2.6, "箱に入ったことをカメラで確かめて、次へ", "The camera confirms the cube is in the box", None),
        (2.6, 5, "終わりました。", "\"Done.\"", "ロボット"),
    ]),
    S("end", card=("コード・評価の記録・説明資料は GitHub で公開", "Code, evaluation logs and the report are on GitHub",
                   "限界: 1 色 20〜45 秒と遅い / 学習に無い物は扱えない / 評価は 1 台・1 人・各 15 試行\n"
                   "Limits: slow (20–45 s per cube), only taught objects, one robot and 15 trials per condition"), dur=6),
]


# ---------------------------------------------------------------- 書き出し

ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" ".join(cmd)[:2000])
        print(r.stderr[-3000:])
        raise SystemExit(1)


def overlays(sc, i):
    """字幕と札の PNG を作り、(パス, 開始, 終了) の並びで返す。"""
    out = []
    for j, (t0, t1, ja, en, who) in enumerate(sc.get("caps", [])):
        p = os.path.join(WORK, f"s{i:02d}_cap{j}.png")
        caption_png(p, ja, en, who)
        out.append((p, t0, t1))
    if sc.get("tag"):
        p = os.path.join(WORK, f"s{i:02d}_tag.png")
        tag_png(p, *sc["tag"])
        out.append((p, 0, 999))
    return out


def render_with_overlays(base_inputs, base_filter, dur, ovs, out, audio_map):
    """base_filter は [base] を出す filter_complex。そこに PNG を時間指定で重ねる。"""
    cmd = ["ffmpeg", "-v", "error", "-y"] + base_inputs
    n0 = sum(1 for x in base_inputs if x == "-i")
    for p, _, _ in ovs:
        cmd += ["-loop", "1", "-t", f"{dur:.3f}", "-i", p]
    fc = base_filter
    last = "base"
    for k, (p, t0, t1) in enumerate(ovs):
        fc += f";[{last}][{n0 + k}:v]overlay=0:0:enable='between(t,{t0},{t1})'[o{k}]"
        last = f"o{k}"
    cmd += ["-filter_complex", fc, "-map", f"[{last}]"] + audio_map + ["-t", f"{dur:.3f}"] + ENC + [out]
    run(cmd)


def silent():
    return ["-f", "lavfi", "-t", "60", "-i", "anullsrc=r=48000:cl=stereo"]


def render(sc, i):
    out = os.path.join(WORK, f"s{i:02d}.mp4")
    ovs = overlays(sc, i)
    if "clip" in sc:
        c = sc["clip"]
        dur = (c["b"] - c["a"]) / c["speed"]
        sp = c["speed"]
        vf = (f"[0:v]trim=start={c['a']}:end={c['b']},setpts=(PTS-STARTPTS)/{sp},fps={FPS},"
              f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1[base]")
        if sp == 1:
            af = f"[0:a]atrim=start={c['a']}:end={c['b']},asetpts=PTS-STARTPTS[a]"
        else:   # 速送りの区間は音を小さく、速さに合わせる
            tempo = ",".join(["atempo=2.0"] * int(round(__import__("math").log2(sp))))
            af = f"[0:a]atrim=start={c['a']}:end={c['b']},asetpts=PTS-STARTPTS,{tempo},volume=0.35[a]"
        render_with_overlays(["-i", os.path.join(PHONE, c["src"])], vf + ";" + af, dur, ovs, out, ["-map", "[a]"])
    elif "pair" in sc:
        (l, r), dur = sc["pair"], sc["pair"][0]["b"] - sc["pair"][0]["a"]
        lab = os.path.join(WORK, f"s{i:02d}_labels.png")
        im = Image.new("RGBA", (W, H), (17, 17, 17, 255))
        d = ImageDraw.Draw(im)
        for k, (ja, en) in enumerate(sc["labels"]):
            x0 = 8 + k * 960
            d.text((x0 + 24, 150), ja, font=font(FONT_B, 40), fill=(255, 210, 80, 255) if k else (220, 220, 215, 255))
            d.text((x0 + 24, 202), en, font=font(FONT_R, 26), fill=(200, 200, 195, 255))
        im.save(lab)
        vf = (f"[2:v]format=rgba[bg];"
              f"[0:v]trim=start={l['a']}:end={l['b']},setpts=PTS-STARTPTS,fps={FPS},scale=944:-2,tpad=stop_mode=clone:stop_duration=5[L];"
              f"[1:v]trim=start={r['a']}:end={r['b']},setpts=PTS-STARTPTS,fps={FPS},scale=944:-2,tpad=stop_mode=clone:stop_duration=5[R];"
              f"[bg][L]overlay=8:250[t1];[t1][R]overlay=968:250,format=yuv420p,setsar=1[base];"
              f"[1:a]atrim=start={r['a']}:end={r['b']},asetpts=PTS-STARTPTS,apad[a]")
        render_with_overlays(["-i", os.path.join(PHONE, l["src"]), "-i", os.path.join(PHONE, r["src"]),
                              "-loop", "1", "-t", f"{dur:.3f}", "-i", lab], vf, dur, ovs, out, ["-map", "[a]"])
    elif "frames" in sc:
        pat, t0, t1 = sc["frames"]
        d = sorted(glob.glob(os.path.join(REC, pat)))[0]
        tops = sorted(glob.glob(os.path.join(d, "*_base_0_rgb.jpg")))
        mt = [os.path.getmtime(p) for p in tops]
        lst = os.path.join(WORK, f"s{i:02d}_frames.txt")
        dur = t1 - t0
        side = os.path.join(WORK, f"s{i:02d}_side")
        os.makedirs(side, exist_ok=True)
        with open(lst, "w") as f:
            acc = 0.0
            for k, p in enumerate(tops):
                if mt[k] - mt[0] < t0:
                    continue
                nxt = (mt[k + 1] - mt[k]) if k + 1 < len(tops) else 1.0
                pair = Image.new("RGB", (W, H), "#111111")
                a = Image.open(p).resize((940, 529))
                b = Image.open(p.replace("_base_0_rgb", "_left_wrist_0_rgb")).resize((940, 529))
                pair.paste(a, (13, 205))
                pair.paste(b, (967, 205))
                dd = ImageDraw.Draw(pair)
                dd.text((20, 160), "俯瞰カメラ / overhead", font=font(FONT_R, 28), fill="#c3c2b7")
                dd.text((974, 160), "手首カメラ / wrist", font=font(FONT_R, 28), fill="#c3c2b7")
                fp = os.path.join(side, f"{k:05d}.jpg")
                pair.save(fp, quality=92)
                f.write(f"file '{fp}'\nduration {nxt:.3f}\n")
                acc += nxt
                if acc >= dur:
                    break
            f.write(f"file '{fp}'\n")
        vf = f"[0:v]fps={FPS},format=yuv420p,setsar=1[base]"
        render_with_overlays(["-f", "concat", "-safe", "0", "-i", lst] + silent(), vf, dur, ovs, out, ["-map", "1:a"])
    else:
        png = os.path.join(WORK, f"s{i:02d}_bg.png")
        if "card" in sc:
            card_png(png, *sc["card"])
        elif "bars" in sc:
            bars_png(png, sc["bars"][0], sc["bars"][2], sc["bars"][1])
        elif "image" in sc:
            image_card_png(png, os.path.join(FIG, sc["image"][0]), sc["image"][1])
        dur = sc["dur"]
        vf = f"[0:v]fps={FPS},format=yuv420p,setsar=1[base]"
        render_with_overlays(["-loop", "1", "-t", str(dur), "-i", png] + silent(), vf, dur, ovs, out, ["-map", "1:a"])
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    only = {int(x) for x in sys.argv[1:]}
    parts = []
    jo, je = os.environ.get("JINGLE_OPEN"), os.environ.get("JINGLE_END")
    for i, sc in enumerate(SCENES):
        if "optional" in sc:
            p = os.path.join(PHONE, sc["optional"])
            if not os.path.exists(p):
                print(f"[{i}] {sc['name']}: 素材 {sc['optional']} が無いので飛ばす")
                continue
        out = os.path.join(WORK, f"s{i:02d}.mp4")
        if not only or i in only or not os.path.exists(out):
            print(f"[{i}] {sc['name']} を作る")
            render(sc, i)
        parts.append(out)
    if jo:
        parts.insert(0, normalize(jo, "jingle_open"))
    if je:
        parts.append(normalize(je, "jingle_end"))
    lst = os.path.join(WORK, "concat.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    final = os.path.join(OUT, "final.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
         "-map_metadata", "-1", "-movflags", "+faststart", final])
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", final],
                             capture_output=True, text=True).stdout)
    print(f"保存: {final}({d:.1f} 秒)")


def normalize(src, name):
    out = os.path.join(WORK, f"{name}.mp4")
    # 運営のジングルは音が無いことがあるので、無音を足してほかの場面と同じ形式にそろえる
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip()
    amap = ["-map", "0:a:0"] if has_audio else ["-map", "1:a"]
    run(["ffmpeg", "-v", "error", "-y", "-i", src] + silent() + ["-map", "0:v:0"] + amap + ["-vf",
         f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}",
         "-shortest"] + ENC + [out])
    return out


if __name__ == "__main__":
    main()
