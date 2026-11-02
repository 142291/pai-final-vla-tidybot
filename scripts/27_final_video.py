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
    lines = ja.split("\n")
    # 横幅に収まらない字幕は、収まるまで文字を小さくする(はみ出して切れるのを防ぐ)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    sj = 44
    while sj > 30 and max(probe.textlength(l, font=font(FONT_B, sj)) for l in lines) + (
            probe.textlength(who, font=font(FONT_B, sj)) + 18 if who else 0) > W - 160:
        sj -= 2
    se = 30
    while en and se > 22 and max(probe.textlength(l, font=font(FONT_R, se)) for l in en.split("\n")) > W - 160:
        se -= 1
    fj, fe, fw = font(FONT_B, sj), font(FONT_R, se), font(FONT_B, sj)
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


def fit(text, path, size, maxw, low=24):
    """横幅 maxw に収まるまで文字を小さくしたフォントを返す。"""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while size > low and max(probe.textlength(l, font=font(path, size)) for l in text.split("\n")) > maxw:
        size -= 2
    return font(path, size)


# スライドは「段」の並びとして作る。段 0 がタイトル(結論)、そのあとリード文の 1 行ずつ、図の順に現れる。
DARK = "#111111"


def head(d, title, en, x, y, maxw, ts=64, ink="#f0efec", sub="#7fb2ee"):
    """タイトル(結論)と英訳を描き、描き終えた y を返す。"""
    ft = fit(title, FONT_B, ts, maxw)
    for line in title.split("\n"):
        d.text((x, y), line, font=ft, fill=ink)
        y += int(ft.size * 1.35)
    if en:
        d.text((x + 2, y + 4), en, font=fit(en, FONT_R, 30, maxw), fill=sub)
        y += 58
    return y


def lead_stages(im, lead, x, y, maxw, body="#c3c2b7", fs=34):
    """リード文を 1 行ずつ足した段の画像を返す(im は書き換える)。描き終えた y も返す。"""
    out = []
    if not lead:
        return out, y
    fl = fit(lead, FONT_R, fs, maxw)
    d = ImageDraw.Draw(im)
    y += 14
    for line in lead.split("\n"):
        d.text((x + 2, y), line, font=fl, fill=body)
        y += int(fl.size * 1.6)
        out.append(im.copy())
    return out, y


def card_stages(title, en, lead=""):
    im = Image.new("RGB", (W, H), DARK)
    y = head(ImageDraw.Draw(im), title, en, 140, 280, W - 280, ts=76)
    st = [im.copy()]
    more, _ = lead_stages(im, lead, 140, y, W - 280)
    return st + more


def why_stages(title, en, items):
    im = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(im)
    y = head(d, title, en, 140, 150, W - 280, ts=60) + 70
    st = [im.copy()]
    for k, (ja, en2) in enumerate(items):
        d.text((140, y), "●", font=font(FONT_B, 36), fill="#eb6834" if k == len(items) - 1 else "#7fb2ee")
        d.text((200, y - 4), ja, font=fit(ja, FONT_B, 42, W - 340), fill="#f0efec")
        d.text((200, y + 60), en2, font=font(FONT_R, 28), fill="#a9a8a0")
        y += 170
        st.append(im.copy())
    return st


def flow_stages(title, en, boxes, color="#7fb2ee", fill="#1b2430", lead=""):
    """左から右へ並ぶ箱の流れ図。箱が 1 つずつ現れる。"""
    im = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(im)
    y = head(d, title, en, 120, 110, W - 240, ts=56)
    st = [im.copy()]
    more, y = lead_stages(im, lead, 120, y, W - 240)
    st += more
    n, bw, gap, bh = len(boxes), 330, 56, 250
    y0 = max(380, y + 60)
    x0 = (W - (n * bw + (n - 1) * gap)) // 2
    for k, (hd, body) in enumerate(boxes):
        x = x0 + k * (bw + gap)
        if k > 0:
            ax = x - gap + 8
            d.line([ax, y0 + bh // 2, ax + gap - 24, y0 + bh // 2], fill="#f0efec", width=5)
            d.polygon([(ax + gap - 24, y0 + bh // 2 - 12), (ax + gap - 12, y0 + bh // 2), (ax + gap - 24, y0 + bh // 2 + 12)], fill="#f0efec")
        d.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=18, outline=color, width=4, fill=fill)
        d.text((x + 24, y0 + 24), hd, font=fit(hd, FONT_B, 38, bw - 40), fill="#f0efec")
        yy = y0 + 90
        for line in body.split("\n"):
            d.text((x + 24, yy), line, font=fit(line, FONT_R, 28, bw - 40), fill="#c3c2b7")
            yy += 42
        st.append(im.copy())
    return st


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / 2 / n) / den
    h = z * ((p * (1 - p) / n + z * z / 4 / n / n) ** 0.5) / den
    return c - h, c + h


def fig_image(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=SURFACE, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def bars_frame(title, en, lead_lines, bars, g):
    """棒グラフのスライド 1 枚。lead_lines は見せる行、g は棒の伸び(None なら図を出さない、1 で完成)。"""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    fig.text(0.06, 0.925, title, fontsize=36, color=INK, va="center", weight="bold")
    fig.text(0.061, 0.872, en, fontsize=19, color=BLUE, va="center")
    if lead_lines:
        fig.text(0.061, 0.835, "\n".join(lead_lines), fontsize=21, color=INK2, va="top", linespacing=1.6)
    if g is not None:
        ax = fig.add_axes([0.10, 0.18, 0.82, 0.44], facecolor=SURFACE)
        for x, (label, k, n, col) in enumerate(bars):
            p = k / n
            lo, hi = wilson(k, n)
            ax.bar(x, p * 100 * g, width=0.56, color=col, zorder=2)
            if g >= 1:
                ax.errorbar(x, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], color=INK2, capsize=8, lw=2, zorder=3)
                ax.text(x, 4, f"{k}/{n}", ha="center", va="bottom", fontsize=22, color="white", zorder=4)
                ax.text(x, hi * 100 + 3, f"{p * 100:.0f}%", ha="center", va="bottom", fontsize=30, color=INK, zorder=4)
        ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=21)
        ax.set_xlim(-0.6, len(bars) - 0.4)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(axis="y", color="#e6e5e1", lw=1, zorder=0)
        ax.set_ylabel("成功率(%) 縦線は 95% 信頼区間", fontsize=18)
    return fig_image(fig)


def bars_stages(title, en, lead, bars):
    """段(タイトル、リード文の各行、空の図)と、棒が伸びるコマの並びを返す。"""
    lines = lead.split("\n") if lead else []
    st = [bars_frame(title, en, lines[:k], bars, None) for k in range(len(lines) + 1)]
    grow = [bars_frame(title, en, lines, bars, 1 - (1 - f / 18) ** 3) for f in range(19)]
    return st, grow


def image_stages(img, title, en, lead):
    """説明資料の図のスライド。タイトル → リード文 → 図の順に現れる。"""
    im = Image.new("RGB", (W, H), SURFACE)
    y = head(ImageDraw.Draw(im), title, en, 115, 40, W - 230, ts=50, ink=INK, sub=BLUE)
    st = [im.copy()]
    more, y = lead_stages(im, lead, 115, y, W - 230, body=INK2)
    st += more
    fig = Image.open(img).convert("RGB")
    box_w, box_h = W - 160, H - y - 30
    s = min(box_w / fig.width, box_h / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    im.paste(fig, ((W - fig.width) // 2, y + 10))
    st.append(im.copy())
    return st


def slide_frames(stages, grow, dur, dirpath, step=0.75, fade=0.35):
    """段を順に重ねる(各段は前の段から fade 秒で切り替わる)。concat 用の (画像, 秒) の並びを返す。"""
    os.makedirs(dirpath, exist_ok=True)
    nf = int(round(fade * FPS))
    seq = []
    prev = Image.new("RGB", (W, H), stages[0].getpixel((5, 5)))
    t = 0.0
    # 最後の段(と棒の伸び)が終わってから 2.5 秒以上読めるように、段の間隔を詰める
    room = dur - 2.5 - 0.4 - fade - (0.4 + 19 / FPS if grow else 0)
    step = min(step, room / max(1, len(stages) - 1))
    keys = [(0.0 if k == 0 else 0.4 + step * k, im) for k, im in enumerate(stages)]
    for tk, im in keys:
        if tk > t:
            seq.append((prev, tk - t))
            t = tk
        for j in range(1, nf + 1):
            seq.append((Image.blend(prev, im, j / nf), 1 / FPS))
        t += nf / FPS
        prev = im
    if grow:
        tg = t + 0.4
        seq.append((prev, tg - t))
        t = tg
        for im in grow:
            seq.append((im, 1 / FPS))
        t += len(grow) / FPS
        prev = grow[-1]
    seq.append((prev, max(0.1, dur - t)))
    lst = os.path.join(dirpath, "list.txt")
    with open(lst, "w") as f:
        fp = None
        for k, (im, d) in enumerate(seq):
            fp = os.path.join(dirpath, f"{k:04d}.jpg")
            im.save(fp, quality=93)
            f.write(f"file '{fp}'\nduration {d:.4f}\n")
        f.write(f"file '{fp}'\n")
    return lst


# ---------------------------------------------------------------- 場面

def ph(src, a, b, speed=1.0):
    return {"src": src, "a": a, "b": b, "speed": speed}


def S(name, **kw):
    kw["name"] = name
    return kw


# 字幕は (場面の中での開始秒, 終了秒, 日本語, 英語, 話し手)
SCENES = [
    S("compare", pair=(ph("E_async.mp4", 8.0, 27.0), ph("E_sync.mp4", 8.4, 27.4)), crop="900:560:950:380", pair_y=200,
      counter=1.8, chapter="実行の仕方だけを変えて比べる",
      labels=(("非同期実行(LeRobot の標準)", "Asynchronous execution (LeRobot default)"), ("同期実行(止まって考える)", "Synchronous execution: stop and re-think")), caps=[
        (0, 5.5, "同じモデル・同じ置き場所。違うのは、アクションチャンクの実行の仕方だけ",
         "Same fine-tuned π0.5, same placement — only the way actions are executed differs", None),
        (5.5, 12.5, "左はキューブのすぐ横まで行って、止まったまま掴めない",
         "Left: it reaches the cube, then stalls without grasping", None),
        (12.5, 19, "右は 1 回目で箱へ。左は 60 秒たっても掴めなかった。なぜか?",
         "Right: in the box on the first try. Left: still nothing after 60 s. Why?", None),
    ]),
    S("title", card=("VLA が掴めない原因を切り分け、\n言葉で頼める片付けロボットにした",
                     "Why couldn't my VLA grasp? Tracking down the causes on a real SO-101",
                     "自前データで π0.5 を学習。原因は「アクションチャンクの実行の仕方」と「デモに無い状況」だった。"), dur=5),
    S("why", why=("クラウドの GPU で VLA を動かすと約 1 秒遅れ、実機では掴み損ねた",
                  "Running the VLA on a cloud GPU added ~1 s of lag, and the real arm missed the grasp", [
        ("観測から行動が届くまで約 1 秒(推論遅延)。そのあいだもアームは動いている。", "~1 s round trip from observation to action, while the arm keeps moving"),
        ("学習は順調でも、実機では「すぐ横まで行って掴めない」が起きる。", "Training looks fine, yet the real robot hovers and misses"),
        ("原因を切り分けて直す。それでも失敗したら理由を説明して人に頼む。", "Find and fix the cause; when it still fails, explain why and ask a human"),
    ]), dur=6),
    S("system", flow=("言葉で頼むと、Claude が段取りを決め、π0.5 が動かす",
                        "Ask in words: Claude plans the order, π0.5 moves the arm", [
        ("言葉で頼む", "「紫以外を\n片付けて」"), ("段取り", "Claude が俯瞰の\n画像を見て色の\n順番を決める"),
        ("動かす", "π0.5(自前データで\n学習)を H200 で\n推論、同期実行"), ("判定", "俯瞰カメラで\n箱の中に 1 秒\n止まったら成功")]), dur=6),
    S("diagnose", image=("fig_horizon.png", "アクションチャンクは先ほど外れ、遅延のあいだにその後半を実行していた",
                         "The action chunk drifts further out, and the ~1 s lag meant executing its least accurate tail",
                         "人の操作と比べた予測の誤差は、0〜5 ステップ目 3.3 → 35〜50 ステップ目 13.1 と、先ほど大きい。\n"
                         "推論遅延が約 1 秒あるので、非同期実行では届いた時点で前半を飛ばし、外れやすい後半を実行していた(最有力の原因)。"), dur=10),
    S("fix", bars=("同期実行に変えると、成功率は 33% → 80%",
                   "Synchronous execution: 33% → 80% with the same model (after the additional training shown next)",
                   "アクションチャンクの先頭 15 ステップ(0.5 秒)だけ動かして止まり、最新の画像で推論し直す(止まって考える)。\n"
                   "講義(基礎編 第 7 回)の通り同期実行は止まるが、遅延が約 1 秒あるこの構成では、非同期実行より確実に掴めた。\n"
                   "比べたのは、この後で説明する追加学習を終えたモデル。同じモデル・同じ晩・決めた 15 配置で、実行の仕方だけを変えた。", [
        ("非同期実行", 5, 15, MUTED), ("同期実行(止まって考える)", 12, 15, BLUE)]), dur=10),
    S("data", bars=("足りない状況のデモを 45 本足すと、1 個入った後も止まらなくなった",
                    "Adding 45 demos for missing situations: no more freezing after the first cube",
                    "最初のデモ 144 本には、箱にキューブが入った状態から始まるものが 1 本も無く、1 個目を入れると止まっていた。\n"
                    "その状態と、置き場所のすき間のデモを 45 本足して追加学習。止まった観測を再生すると、動こうとする量は 1.6 → 120.8。\n"
                    "ここからの成功率はすべて同期実行。53% → 67%(学習の step も増えているので、デモだけの効果とは言い切れない)。", [
        ("最初のモデル\n(デモ 144 本)", 8, 15, MUTED), ("+ 箱に入った状態・\nすき間のデモ 45 本", 10, 15, BLUE)]), dur=10),
    S("hook", clip=ph("A_except_purple.mp4", 6.0, 17.0), chapter="言葉で頼む", caps=[
        (0, 1.3, "指示「紫以外を片付けて」", "Instruction: \"Tidy up everything except purple\"", None),
        (1.3, 6.2, "紫以外の、赤と黄色のキューブを箱に入れますね。", "\"I'll put the red and yellow cubes in the box.\"", "ロボット"),
        (6.2, 11, "段取りは Claude が俯瞰の画像を見て決め、π0.5 が動かす(1 倍速)", "Claude plans from the overhead image; π0.5 moves the arm (real time)", None),
    ]),
    S("tidy_fast", clip=ph("B1_tidy_all.mp4", 8.0, 79.0, 4.0), tag=("4× 速送り", "4× speed"), chapter="言葉で頼む", caps=[
        (0, 17.75, "「片付けて」で 3 色とも 1 回目で成功した回(撮影した回の中から選んだもの)",
         "\"Tidy up\": all three cubes on the first try (a selected take)", None),
    ]),
    S("refuse", optional="C_green.mp4", clip=ph("C_green.mp4", 0.0, 11.8), chapter="言葉で頼む", caps=[
        (0, 6.0, "指示「緑のを入れて」(机にあるのは赤・黄・紫だけ)", "Instruction: \"Put the green one in\" (only red, yellow and purple on the table)", None),
        (6.0, 10.0, "すみません、緑のものは見当たらないので入れられません。", "\"Sorry, I don't see anything green, so I can't put it in.\"", "ロボット"),
        (10.0, 11.8, "机の上を見て、できないことは断る", "It checks the table and declines what it can't do", None),
    ]),
    S("failflow", flow=("工夫: 失敗したら、理由を説明してやり直し、だめなら人を呼ぶ",
                          "When it fails: Claude explains why, retries up to 3 times, then calls a human", [
        ("失敗", "箱に入らない\n(掴めない など)"), ("理由を説明", "Claude が試行中の\n画像を見て\n言葉で説明する"),
        ("やり直す", "同じ指示で\nもう一度\n(3 回まで)"), ("人を呼ぶ", "リーダーアームで\n引き取る(介入)\n→ 学習データに保存")],
        "#eb6834", "#2a1d18",
        "人に頼むときも、何がうまくいかなかったかを言葉で伝える。人の操作は、次の学習に使うデータになる。"), dur=5),
    S("explain", clip=ph("D_red_only_human.mp4", 72.0, 80.0), chapter="失敗したら人を呼ぶ", caps=[
        (0, 8, "「赤だけ入れて」。苦手な置き場所(真ん中)では、すぐ横まで行って掴めない",
         "\"Put only the red one in.\" At a hard placement it gets right next to the cube and misses", None),
    ]),
    S("explain2", clip=ph("D_red_only_human.mp4", 85.8, 94.0), chapter="失敗したら人を呼ぶ", caps=[
        (0, 8.2, "赤のすぐ横まで行けましたが、掴めませんでした。もう一度やってみます。",
         "\"I got right next to the red cube but couldn't grasp it. I'll try again.\"", "ロボット"),
    ], tag=("失敗の理由は Claude が試行中の画像から判断", "Claude explains the failure from frames of the attempt")),
    S("call", clip=ph("D_red_only_human.mp4", 219.6, 229.8), chapter="失敗したら人を呼ぶ", caps=[
        (0, 1.6, "(2 回目・3 回目も同じ失敗。省略)", "(Attempts 2 and 3 fail the same way — skipped)", None),
        (1.6, 10.2, "…掴めませんでした。うまく掴めません。リーダーアームで手伝ってください。",
         "\"...couldn't grasp it. I can't get it. Please help me with the leader arm.\"", "ロボット"),
    ]),
    S("assist", clip=ph("D_red_only_human.mp4", 229.8, 279.5, 4.0), tag=("4× 速送り", "4× speed"), chapter="失敗したら人を呼ぶ", caps=[
        (0, 12.5, "人がリーダーアームで引き取る(介入)。その操作は、そのまま学習データとして保存する",
         "A human takes over with the leader arm; the assist is saved as training data", None),
    ]),
    S("done", clip=ph("D_red_only_human.mp4", 279.5, 284.5), chapter="失敗したら人を呼ぶ", caps=[
        (0, 2.6, "箱に入ったことをカメラで確かめて、次へ", "The camera confirms the cube is in the box", None),
        (2.6, 5, "終わりました。", "\"Done.\"", "ロボット"),
    ]),
    S("relearn", bars=("人の介入を学習に足すと、67% → 80%",
                       "Adding human-intervention data: 67% → 80%",
                       "ロボットが失敗した場面だけ人がリーダーアームで引き取り(介入)、その操作を学習データに足した。\n"
                       "どちらも同じモデルから、同じだけ(+4,000 step)追加学習。違いは介入のデータを入れたかだけ。\n"
                       "ただし差は 15 試行中 2 回で、言い切るには試行を増やす必要がある。", [
        ("介入のデータなし", 10, 15, MUTED), ("介入のデータあり", 12, 15, BLUE)]), dur=10),
    S("end", card=("github.com/142291/pai-final-vla-tidybot", "コード・試行ごとの評価記録・説明資料(PDF)を公開 / Code, per-trial logs and the report",
                   "同期実行に変えるだけで 33% → 80%(同じモデル・同じ晩・同じ 15 配置)\n"
                   "人の介入を学習に足すと 67% → 80%(同じだけ学習した介入なしは 67%)\n"
                   "通し動作「片付けて」: 自律化率 15/24、残りは人を呼んで 24/24\n"
                   "限界: 1 色 20〜45 秒と遅い / 学習に無い物は未検証 / 1 台・評価者 1 人・各 15 試行"), dur=7),
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
    if sc.get("chapter"):
        p = os.path.join(WORK, f"s{i:02d}_chapter.png")
        chapter_png(p, sc["chapter"])
        out.append((p, 0, 999))
    if sc.get("counter"):   # 左右の画面の左下に、指示からの経過秒を出す
        t_start, dur = sc["counter"], sc["pair"][0]["b"] - sc["pair"][0]["a"]
        k = 0
        while k < dur:
            sec = int(t_start + k + 1e-6)
            nxt = min(dur, (sec + 1) - t_start)
            p = os.path.join(WORK, f"s{i:02d}_cnt{sec:03d}.png")
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            for x0 in (8, 968):
                txt = f"指示から {sec} 秒"
                d.rounded_rectangle([x0 + 14, 760, x0 + 30 + d.textlength(txt, font=font(FONT_B, 40)), 822], radius=10, fill=(0, 0, 0, 190))
                d.text((x0 + 22, 766), txt, font=font(FONT_B, 40), fill=(255, 255, 255, 255))
            im.save(p)
            out.append((p, round(k, 3), round(nxt, 3)))
            k = nxt
    return out


def chapter_png(path, text):
    """画面右上の章の札(いまどの話かを示す)。"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(FONT_B, 32)
    w = d.textlength(text, font=f)
    d.rounded_rectangle([W - 60 - w - 36, 40, W - 60, 96], radius=12, fill=(42, 120, 214, 220))
    d.text((W - 60 - w - 18, 48), text, font=f, fill=(255, 255, 255, 255))
    im.save(path)


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
        crop = f"crop={sc['crop']}," if sc.get("crop") else ""
        lab = os.path.join(WORK, f"s{i:02d}_labels.png")
        im = Image.new("RGBA", (W, H), (17, 17, 17, 255))
        d = ImageDraw.Draw(im)
        for k, (ja, en) in enumerate(sc["labels"]):
            x0 = 8 + k * 960
            d.text((x0 + 24, sc.get("pair_y", 250) - 100), ja, font=font(FONT_B, 40), fill=(255, 210, 80, 255) if k else (220, 220, 215, 255))
            d.text((x0 + 24, sc.get("pair_y", 250) - 48), en, font=font(FONT_R, 26), fill=(200, 200, 195, 255))
        im.save(lab)
        vf = (f"[2:v]format=rgba[bg];"
              f"[0:v]trim=start={l['a']}:end={l['b']},setpts=PTS-STARTPTS,fps={FPS},{crop}scale=944:-2,tpad=stop_mode=clone:stop_duration=5[L];"
              f"[1:v]trim=start={r['a']}:end={r['b']},setpts=PTS-STARTPTS,fps={FPS},{crop}scale=944:-2,tpad=stop_mode=clone:stop_duration=5[R];"
              f"[bg][L]overlay=8:{sc.get('pair_y', 250)}[t1];[t1][R]overlay=968:{sc.get('pair_y', 250)},format=yuv420p,setsar=1[base];"
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
        grow = None
        if "card" in sc:
            st = card_stages(*sc["card"])
        elif "bars" in sc:
            st, grow = bars_stages(*sc["bars"])
        elif "why" in sc:
            st = why_stages(*sc["why"])
        elif "flow" in sc:
            st = flow_stages(*sc["flow"])
        elif "image" in sc:
            st = image_stages(os.path.join(FIG, sc["image"][0]), *sc["image"][1:])
        (grow[-1] if grow else st[-1]).save(os.path.join(WORK, f"s{i:02d}_bg.png"))
        dur = sc["dur"]
        lst = slide_frames(st, grow, dur, os.path.join(WORK, f"s{i:02d}_anim"))
        vf = f"[0:v]fps={FPS},format=yuv420p,setsar=1[base]"
        render_with_overlays(["-f", "concat", "-safe", "0", "-i", lst] + silent(), vf, dur, ovs, out, ["-map", "1:a"])
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
    # 本編だけをつないで BGM を重ねる(運営のジングルには重ねない)
    lst = os.path.join(WORK, "body.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    body = os.path.join(WORK, "body.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", body])
    parts = [os.path.join(WORK, "body_bgm.mp4")]
    add_bgm(body, parts[0])
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


def make_bgm(path, dur, sr=48000):
    """BGM をその場で合成する(外部の音源を使わないので権利の心配が無い)。
    ゆったりした 4 和音の繰り返し: 柔らかい和音(パッド)+ 8 分音符の分散和音 + 低音。"""
    import wave

    import numpy as np
    bpm = 84
    beat = 60 / bpm
    bar = beat * 4
    chords = [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 67]]   # Cmaj7 Am7 Fmaj7 G
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2))
    k = 0
    while k * bar < dur:
        ch = chords[k % 4]
        a, b = int(k * bar * sr), min(n, int((k + 1) * bar * sr))
        tt = t[a:b] - k * bar
        env = np.minimum(1, tt / 0.6) * np.minimum(1, (bar - tt) / 0.6)
        pad = sum(np.sin(2 * np.pi * hz(m) * tt) + 0.3 * np.sin(2 * np.pi * hz(m) * 2.003 * tt) for m in ch) / len(ch)
        out[a:b, 0] += 0.22 * env * pad
        out[a:b, 1] += 0.22 * env * pad
        bass = np.sin(2 * np.pi * hz(ch[0] - 12) * tt) * env
        out[a:b] += 0.18 * bass[:, None]
        for j in range(8):   # 分散和音(8 分音符)、左右に少し振る
            s0 = a + int(j * beat / 2 * sr)
            if s0 >= n:
                break
            ln = min(n - s0, int(1.2 * sr))
            tp = np.arange(ln) / sr
            m = ch[[0, 2, 1, 3, 2, 1, 3, 2][j]] + 12
            note = np.sin(2 * np.pi * hz(m) * tp) * np.exp(-tp * 4.0) * 0.16
            pan = 0.5 + 0.3 * np.sin(j)
            out[s0:s0 + ln, 0] += note * (1 - pan)
            out[s0:s0 + ln, 1] += note * pan
        k += 1
    fade = np.minimum(1, np.minimum(t / 1.5, (dur - t) / 2.5))[:, None]
    out = out * fade
    out = out / np.abs(out).max() * 0.8
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype("<i2").tobytes())


def add_bgm(src, out, vol=0.22):
    """本編に BGM を重ねる。ロボットの声などの元の音がある間は BGM を自動で下げる(sidechaincompress)。"""
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                             capture_output=True, text=True).stdout)
    wav = os.path.join(WORK, "bgm.wav")
    make_bgm(wav, d)
    fc = (f"[0:a]asplit=2[orig][key];[1:a]volume={vol}[bg];"
          f"[bg][key]sidechaincompress=threshold=0.03:ratio=6:attack=30:release=600[duck];"
          f"[orig][duck]amix=inputs=2:duration=first:normalize=0[a]")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", wav, "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", out])


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
