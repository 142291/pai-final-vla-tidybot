"""失敗した試行の画像から、何が起きたかを Claude に判定させ、ロボットが読み上げる一言を作る。

使い方(単体で、記録した評価の試行を説明させる):
    .venv/bin/python scripts/explain.py outputs/eval/rec/<試行のフォルダ> [red|yellow|purple|eraser]

16_orchestrator.py は、試行が失敗したらこの explain() を呼び、「黄色を押し出してしまいました。もう一度やります」の
ように読み上げてからやり直す。型は録画を見て分類してきた失敗(2026-10-26〜28)から決めた:
    near_miss   対象の真上・すぐ横まで来たが、掴めない/閉じない
    pushed      掴む前にグリッパが当たって、対象を押し出した
    lost        対象を見失い、別の場所へ離れていった
    hesitate    対象とほかの物の間で迷うように止まった
    wrong       違う物を掴んだ/運んだ
    dropped     掴んだが、運ぶ途中や箱の手前で落とした
    other       上のどれでもない

2026-10-31: 「失敗に気づいてやり直す」だけでは、何がまずかったのかが人に伝わらない。現場で人が手伝うときに、
ロボット自身が状況を説明できると助けやすいので作った。
"""

import base64
import os
import sys
from typing import Literal

import cv2
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objects import JA  # noqa: E402

MODEL = os.environ.get("EXPLAIN_MODEL", "claude-opus-5-5")
KINDS = ("near_miss", "pushed", "lost", "hesitate", "wrong", "dropped", "other")


class Explanation(BaseModel):
    kind: Literal["near_miss", "pushed", "lost", "hesitate", "wrong", "dropped", "other"] = Field(
        description="失敗の型")
    say: str = Field(description="ロボットが一人称で読み上げる短い説明(日本語、1 文、20 字前後)。例: 黄色を押し出してしまいました")
    reason: str = Field(description="画像のどこからそう判断したか(日本語、1 文)")


PROMPT = """卓上ロボットアームが「{obj}を掴んで箱(白い底の黒い枠のトレー)に入れる」ことに失敗しました。
試行中の画像を時刻順に {n} 組渡します(各組は、上が俯瞰カメラ、下がグリッパに付いた手首カメラ)。
{mark}型の見分け方: アームが対象の真上かすぐ横で止まっているなら near_miss、対象から離れた場所(ほかの物の近くや物の間)で
止まっているなら hesitate、対象の位置が途中で動いていれば pushed、対象が画面から消えたりアームが全く違う方向へ行ったなら lost。
何が起きて失敗したかを、次の型から 1 つ選んでください。
- near_miss: 対象の真上やすぐ横まで来たが、掴めない/グリッパを閉じない
- pushed: 掴む前にグリッパが当たって、対象を押し出した(対象の位置が変わっている)
- lost: 対象を見失い、別の場所へ離れていった
- hesitate: 対象とほかの物の間で迷うように止まった
- wrong: 違う物を掴んだ/運んだ
- dropped: 掴んだが、運ぶ途中や箱の手前で落とした
- other: どれでもない
say はロボットが自分のこととして読み上げる短い一言にしてください(例:「赤のすぐ上までは行けましたが、掴めませんでした」)。"""


def _pair(top_rgb, wrist_rgb):
    t = cv2.resize(top_rgb, (480, 270))
    w = cv2.resize(wrist_rgb, (480, 270))
    return cv2.vconcat([t, w])


def explain(frames, obj: str, target_xy=None) -> Explanation:
    """frames は (俯瞰 RGB, 手首 RGB) の時刻順の並び(4〜6 組が目安)。target_xy は最初の俯瞰画像での対象の位置(あれば丸で示す)。"""
    import anthropic

    content = []
    for i, (top, wrist) in enumerate(frames):
        if i == 0 and target_xy is not None:
            top = top.copy()
            cv2.circle(top, (int(target_xy[0]), int(target_xy[1])), 22, (0, 255, 0), 3)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(_pair(top, wrist), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(jpg.tobytes()).decode()}})
    mark = "最初の組の俯瞰画像で、緑の丸が対象の最初の位置です。\n" if target_xy is not None else ""
    content.append({"type": "text", "text": PROMPT.format(obj=JA.get(obj, obj), n=len(frames), mark=mark)})
    resp = anthropic.Anthropic().beta.messages.parse(
        model=MODEL, max_tokens=2000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": content}], output_format=Explanation)
    if resp.parsed_output is None:
        raise RuntimeError(f"説明が返らなかった(stop_reason={resp.stop_reason})")
    return resp.parsed_output


def target_from_frame(top_rgb, obj):
    """キューブなら色検出で最初の位置を返す(無ければ None)。"""
    from importlib import import_module
    audit = import_module("06d_audit_dataset")
    try:
        return audit.find_cube(cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR), obj)
    except Exception:
        return None


def frames_from_rec(rec_dir, n=5):
    """評価の記録フォルダ(14_eval.py の rec/)から、時刻順に n 組を等間隔で取り出す(後半を厚めに)。"""
    import glob

    tops = sorted(glob.glob(os.path.join(rec_dir, "*_base_0_rgb.jpg")))
    if not tops:
        raise FileNotFoundError(f"{rec_dir} に画像が無い")
    idx = sorted({min(len(tops) - 1, int(len(tops) * f)) for f in (0.15, 0.4, 0.6, 0.8, 0.98)[:n]})
    out = []
    for i in idx:
        t = cv2.cvtColor(cv2.imread(tops[i]), cv2.COLOR_BGR2RGB)
        w = cv2.cvtColor(cv2.imread(tops[i].replace("_base_0_rgb", "_left_wrist_0_rgb")), cv2.COLOR_BGR2RGB)
        out.append((t, w))
    return out


if __name__ == "__main__":
    rec = sys.argv[1]
    obj = sys.argv[2] if len(sys.argv) > 2 else "red"
    fr = frames_from_rec(rec)
    e = explain(fr, obj, target_from_frame(fr[0][0], obj))
    print(f"{e.kind}: {e.say}  ({e.reason})")
