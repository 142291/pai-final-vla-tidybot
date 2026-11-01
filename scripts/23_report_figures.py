"""説明資料(PDF)と動画に使うグラフを docs/submission/figures/ に書き出す。

使い方(matplotlib が入った Python で動かす。.venv には入れていない):
    python3 scripts/23_report_figures.py

数値は docs/evaluation.md の表から写している。評価をやり直したらここも直すこと。

  fig_horizon.png  予測の何ステップ先ほど外れるか(3 方策)。「止まって考える」で使う範囲を網掛け
  fig_progress.png 実行方式の改善 → データを足したラウンドごとの成功率 → 通し動作の自律化率
"""

import os
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "submission", "figures")

# 色: データ可視化の既定パレットの先頭 3 色(色覚多様性の検査済み)。文字は色に頼らずインクで書く
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "Hiragino Kaku Gothic Pro", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return c - h, c + h


def fig_horizon():
    # 学習に使っていない 4 本の、動いている区間での関節誤差(docs/evaluation.md「比較実験」)
    mids = [2.5, 10, 20, 30, 42.5]
    series = [
        ("SmolVLA", ORANGE, "s", mids, [3.1, 5.7, 7.4, 10.0, 11.3]),
        ("ACT", AQUA, "^", mids, [4.1, 6.2, 8.8, 11.7, 13.4]),
        ("π0.5", BLUE, "o", [2.5, 30, 42.5], [3.3, 11.5, 13.1]),   # 5–25 は未計測
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ax.axvspan(0, 15, color=BLUE, alpha=0.08, lw=0)
    ax.text(7.5, 14.6, "止まって考える実行で\n使う範囲(先頭 15 個)", ha="center", va="top", fontsize=9.5, color=INK2)
    nudge = {"ACT": 0.45, "π0.5": -0.45}   # 終点の名前が重ならないように少しずらす
    for name, col, mk, x, y in series:
        if name == "π0.5":   # 測っていない 5–25 は点線でつなぐ
            ax.plot(x[:2], y[:2], color=col, lw=2, ls=(0, (3, 3)), zorder=3)
            ax.plot(x[1:], y[1:], color=col, lw=2, zorder=3)
            ax.plot(x, y, ls="none", color=col, marker=mk, ms=8, mec=SURFACE, mew=2, label=name, zorder=4)
        else:
            ax.plot(x, y, color=col, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, label=name, zorder=3)
        ax.text(x[-1] + 1.2, y[-1] + nudge.get(name, 0), name, va="center", fontsize=10, color=INK)
    ax.text(51.5, 0.4, "π0.5 は 5–25 ステップ目を測っていない(点線)", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 15)
    ax.set_xticks(mids, ["0–5", "5–15", "15–25", "25–35", "35–50"])
    ax.set_xlabel("予測チャンクの何ステップ目か(1 ステップ = 1/30 秒)")
    ax.set_ylabel("人の操作との誤差(関節の平均、-100〜100 の単位)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.06), fontsize=9.5)
    ax.set_title("先の予測ほど外れる ― 遅延 約 1 秒の間に、外れやすい後半を実行していた", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_horizon.png"))


def fig_progress():
    panels = [
        ("① 実行の仕方(同じモデル・同じ 15 か所)", [("非同期\n10/31", 5, 15), ("止まって考える\n10/31", 12, 15), ("止まって考える\n10/28", 14, 15)]),
        ("② 同じ 15 配置での成功率", [("R0\n150 本", 8, 15), ("R1\n+すき間\n45 本", 10, 15),
                                     ("R2\n+手助け", 12, 15), ("R3\n+手助け", 12, 15),
                                     ("対照\nR1 のまま\n学習を延長", 14, 15)]),
        ("③ 通し動作「片付けて」の自律化率", [("ラウンド 1", 13, 24), ("ラウンド 2", 15, 24)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), dpi=200, gridspec_kw={"width_ratios": [2.8, 4.6, 2]})
    for ax, (title, bars) in zip(axes, panels):
        xs = range(len(bars))
        for x, (label, k, n) in zip(xs, bars):
            p = k / n
            lo, hi = wilson(k, n)
            ax.bar(x, p * 100, width=0.55, color=ORANGE if label.startswith("対照") else BLUE, zorder=2)
            ax.errorbar(x, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], color=INK2, capsize=4, lw=1.2, zorder=3)
            ax.text(x, 3, f"{k}/{n}", ha="center", va="bottom", fontsize=9.5, color="white", zorder=4)
            ax.text(x, hi * 100 + 2, f"{p * 100:.0f}%", ha="center", va="bottom", fontsize=10.5, color=INK, zorder=4)
        ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=9.5)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("成功率(%)  縦線は 95% 信頼区間")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_progress.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_horizon()
    fig_progress()
    print(f"保存: {OUT}")
