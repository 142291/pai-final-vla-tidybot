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


def why_png(path, title, items):
    """「なぜ取り組むか」の 1 枚。items は (日本語, 英語) の並び。"""
    im = Image.new("RGB", (W, H), "#111111")
    d = ImageDraw.Draw(im)
    d.text((140, 170), title, font=font(FONT_B, 64), fill="#f0efec")
    y = 330
    for k, (ja, en) in enumerate(items):
        d.text((140, y), "●", font=font(FONT_B, 36), fill="#eb6834" if k == len(items) - 1 else "#7fb2ee")
        d.text((200, y - 4), ja, font=font(FONT_B, 42), fill="#f0efec")
        d.text((200, y + 60), en, font=font(FONT_R, 28), fill="#a9a8a0")
        y += 170
    im.save(path)


def system_png(path, title, boxes, branch):
    """しくみの全体図。boxes は左から右へ並ぶ (見出し, 説明) の箱、branch は判定の下に出る失敗時の流れ。"""
    im = Image.new("RGB", (W, H), "#111111")
    d = ImageDraw.Draw(im)
    d.text((120, 110), title, font=font(FONT_B, 56), fill="#f0efec")
    n, bw, gap, y0, bh = len(boxes), 300, 60, 330, 220
    x0 = (W - (n * bw + (n - 1) * gap)) // 2
    centers = []
    for k, (head, body) in enumerate(boxes):
        x = x0 + k * (bw + gap)
        d.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=18, outline="#7fb2ee", width=4, fill="#1b2430")
        d.text((x + 24, y0 + 24), head, font=font(FONT_B, 36), fill="#f0efec")
        yy = y0 + 84
        for line in body.split("\n"):
            d.text((x + 24, yy), line, font=font(FONT_R, 26), fill="#c3c2b7")
            yy += 38
        if k < n - 1:
            ax = x + bw + 8
            d.line([ax, y0 + bh // 2, ax + gap - 16, y0 + bh // 2], fill="#f0efec", width=5)
            d.polygon([(ax + gap - 16, y0 + bh // 2 - 12), (ax + gap - 4, y0 + bh // 2), (ax + gap - 16, y0 + bh // 2 + 12)], fill="#f0efec")
        centers.append(x + bw // 2)
    # 判定(最後の箱)の下に、失敗したときの流れ
    cx = centers[-1]
    d.line([cx, y0 + bh + 8, cx, y0 + bh + 70], fill="#eb6834", width=5)
    d.text((cx + 16, y0 + bh + 18), "失敗", font=font(FONT_B, 30), fill="#eb6834")
    bx1 = x0 + (n - 4) * (bw + gap)
    d.rounded_rectangle([bx1, y0 + bh + 80, x0 + n * bw + (n - 1) * gap, y0 + bh + 200], radius=18, outline="#eb6834", width=4, fill="#2a1d18")
    d.text((bx1 + 28, y0 + bh + 100), branch[0], font=font(FONT_B, 34), fill="#f0efec")
    d.text((bx1 + 28, y0 + bh + 150), branch[1], font=font(FONT_R, 26), fill="#c3c2b7")
    im.save(path)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / 2 / n) / den
    h = z * ((p * (1 - p) / n + z * z / 4 / n / n) ** 0.5) / den
    return c - h, c + h


def bars_png(path, title, bars, note="", big=""):
    """成功率の棒グラフ(95% 信頼区間つき)を動画の画面全体の大きさで描く。字幕の分、下を空ける。"""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    fig.text(0.06, 0.90, title, fontsize=34, color=INK, va="center")
    if note:
        fig.text(0.06, 0.84, note, fontsize=20, color=INK2, va="center")
    if big:
        fig.text(0.86, 0.875, big, fontsize=58, color=BLUE, ha="right", va="center", weight="bold")
    ax = fig.add_axes([0.10, 0.38, 0.82, 0.40], facecolor=SURFACE)
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
    S("compare", pair=(ph("E_async.mp4", 8.0, 27.0), ph("E_sync.mp4", 8.4, 27.4)), crop="900:560:950:380", pair_y=200,
      counter=1.8, chapter="謎: 掴めない",
      labels=(("標準の実行(非同期)", "Standard async execution"), ("止まって考える実行", "Stop-and-think execution")), caps=[
        (0, 5.5, "同じモデル・同じ置き場所。違うのは、行動の実行の仕方だけ",
         "Same fine-tuned π0.5, same placement — only the way actions are executed differs", None),
        (5.5, 12.5, "左はキューブのすぐ横まで行って、止まったまま掴めない",
         "Left: it reaches the cube, then stalls without grasping", None),
        (12.5, 19, "右は 1 回目で箱へ。左は 60 秒たっても掴めなかった。なぜか?",
         "Right: in the box on the first try. Left: still nothing after 60 s. Why?", None),
    ]),
    S("title", card=("VLA はなぜ掴めないのか", "Why couldn't my VLA grasp? — tracking down the causes on a real SO-101",
                     "自前データで π0.5 を学習させ、原因を一つずつ切り分けて、言葉で頼める片付けロボットにした"), dur=4),
    S("why", why=("なぜ取り組むか", [
        ("クラウドの GPU で VLA を動かすと、観測から行動まで約 1 秒遅れる", "Running a VLA on a cloud GPU adds ~1 s between observation and action"),
        ("学習は順調でも、実機では「すぐ上まで行って掴めない」が起きる", "Training looks fine, yet the real robot hovers and misses"),
        ("原因を切り分けて直し、失敗したら理由を説明して人に頼めるようにする", "Find and fix the cause; when it still fails, explain why and ask a human"),
    ]), dur=6),
    S("system", system=("しくみ", [
        ("言葉で頼む", "「紫以外を\n片付けて」"), ("段取り", "Claude が俯瞰の\n画像を見て色の\n順番を決める"),
        ("動かす", "π0.5(自前データで\n学習)を H200 で\n推論、止まって考える"), ("判定", "俯瞰カメラで\n箱の中に 1 秒\n止まったら成功")],
        ("Claude が失敗の理由を説明 → やり直し(3 回まで)→ 人を呼ぶ", "人はリーダーアームで引き取る(操作は学習データとして保存)")), dur=6),
    S("diagnose", image=("fig_horizon.png", "原因: 予測の何ステップ先ほど外れるか"), dur=12, chapter="原因", caps=[
        (0, 6, "人の操作と比べると、予測(action chunk)は先のステップほど外れる(誤差 3.3 → 13.1)",
         "Against human teleoperation, the action chunk drifts further out (error 3.3 → 13.1)", None),
        (6, 12, "推論の往復に約 1 秒。その間に、外れやすい予測の後半を実行していた(最有力の原因)",
         "With a ~1 s round trip, the robot executed the least accurate tail of each chunk (most likely cause)", None),
    ]),
    S("fix", bars=("解決: 止まって考える実行", "同じモデル・同じ晩・決めた 15 配置(5 か所 × 3 色)", [
        ("標準の実行(非同期)", 5, 15, MUTED), ("止まって考える実行", 12, 15, BLUE)]),
      big="33% → 80%", dur=10, chapter="解決", caps=[
        (0, 5, "先頭の 15 ステップ(0.5 秒)だけ動かして止まり、最新の画像で予測し直す",
         "Run only the first 15 steps (0.5 s), stop, and re-predict from a fresh image", None),
        (5, 10, "変えたのは実行の仕方だけ。15 配置中 5 → 12 で成功",
         "Only the execution changed: 5 → 12 of 15 placements succeed", None),
    ]),
    S("data", bars=("データの穴: 1 個入ると、次が動かない", "箱に入った状態から始まる手本: 144 本中 0 本", [
        ("R0\n手本 144 本", 8, 15, BLUE), ("R1\n+箱に入った状態・\nすき間 45 本", 10, 15, BLUE)]),
      big="53% → 67%", dur=10, chapter="データの穴", caps=[
        (0, 5, "1 個目が箱に入ると、2 個目に取りかからず止まった。その状態の手本が 1 本も無かった",
         "After the first cube was in the box, it froze: no demo ever started from that state", None),
        (5, 10, "その状態の手本を 45 本足すと、止まらずに動くようになった",
         "Adding 45 such demos got it moving again", None),
    ]),
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
    S("explain", clip=ph("D_red_only_human.mp4", 72.0, 80.0), chapter="失敗したら", caps=[
        (0, 8, "「赤だけ入れて」。苦手な置き場所(真ん中)では、すぐ横まで行って掴めない",
         "\"Put only the red one in.\" At a hard placement it gets right next to the cube and misses", None),
    ]),
    S("explain2", clip=ph("D_red_only_human.mp4", 85.8, 94.0), chapter="失敗したら", caps=[
        (0, 8.2, "赤のすぐ横まで行けましたが、掴めませんでした。もう一度やってみます。",
         "\"I got right next to the red cube but couldn't grasp it. I'll try again.\"", "ロボット"),
    ], tag=("失敗の理由は Claude が試行中の画像から判断", "Claude explains the failure from frames of the attempt")),
    S("call", clip=ph("D_red_only_human.mp4", 219.6, 229.8), chapter="失敗したら", caps=[
        (0, 1.6, "(2 回目・3 回目も同じ失敗。省略)", "(Attempts 2 and 3 fail the same way — skipped)", None),
        (1.6, 10.2, "…掴めませんでした。うまく掴めません。リーダーアームで手伝ってください。",
         "\"...couldn't grasp it. I can't get it. Please help me with the leader arm.\"", "ロボット"),
    ]),
    S("assist", clip=ph("D_red_only_human.mp4", 229.8, 279.5, 4.0), tag=("4× 速送り", "4× speed"), chapter="失敗したら", caps=[
        (0, 12.5, "人がリーダーアームで引き取る。その操作は、そのまま学習データとして保存する",
         "A human takes over with the leader arm; the assist is saved as training data", None),
    ]),
    S("done", clip=ph("D_red_only_human.mp4", 279.5, 284.5), chapter="失敗したら", caps=[
        (0, 2.6, "箱に入ったことをカメラで確かめて、次へ", "The camera confirms the cube is in the box", None),
        (2.6, 5, "終わりました。", "\"Done.\"", "ロボット"),
    ]),
    S("relearn", bars=("学び直し: 人の手助けを学習に戻す", "同じ 15 配置。R1 から同じだけ(4,000 step)学習を足して比べた", [
        ("R1", 10, 15, MUTED), ("対照\nR1 のまま\n学習だけ延長", 10, 15, ORANGE), ("R2\n+人の手助け\n11 区間", 12, 15, BLUE),
        ("R3\n+人の手助け\n8 区間", 12, 15, BLUE)]), big="67% → 80%", dur=10, chapter="学び直し", caps=[
        (0, 5, "人が引き取った操作を学習に戻すと 67% → 80%。学習を延ばしただけ(対照)は 67% のまま",
         "With the human assists added: 67% → 80%. Training longer without them (control): still 67%", None),
        (5, 10, "ただし差は 15 試行中 2 回。言い切るには試行を増やす必要がある",
         "But the gap is 2 of 15 trials — more trials are needed to be sure", None),
    ]),
    S("end", card=("github.com/142291/pai-final-vla-tidybot", "コード・試行ごとの評価記録・説明資料(PDF)を公開 / Code, per-trial logs and the report",
                   "実行の仕方を変えるだけで 33% → 80%(同じモデル・同じ晩・同じ 15 配置)\n"
                   "人の手助けを学習に戻すと 67% → 80%(学習を延ばしただけの対照は 67%)\n"
                   "通し動作「片付けて」: 自力で 15/24、残りは人を呼んで 24/24\n"
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
        png = os.path.join(WORK, f"s{i:02d}_bg.png")
        if "card" in sc:
            card_png(png, *sc["card"])
        elif "bars" in sc:
            bars_png(png, sc["bars"][0], sc["bars"][2], sc["bars"][1], sc.get("big", ""))
        elif "why" in sc:
            why_png(png, *sc["why"])
        elif "system" in sc:
            system_png(png, *sc["system"])
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
