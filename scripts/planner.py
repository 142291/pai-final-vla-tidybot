"""指示と俯瞰画像から「どの色のキューブを、どの順に箱へ入れるか」の段取りを作る(System 2)。

使い方(単体で試す):
    .venv/bin/python scripts/planner.py "紫以外を片付けて" [俯瞰画像.jpg]
    PLANNER=rule .venv/bin/python scripts/planner.py "片付けて" ...   # VLM を使わない

π0.5(System 1)が実行できるのは学習した 3 種類の指示文だけ:
    "Pick up the {red|yellow|purple} cube and put it in the box"
なので段取りは「色の列」に落とす。あいまいな指示(「片付けて」「信号の色のものだけ」
「さっきのもう一回」)の解釈を VLM に任せ、実行は学習済みの指示文に限る。

VLM は Claude(ANTHROPIC_API_KEY が必要。config.env に export で書く)。
使えないとき(鍵が無い、通信できない、拒否された)はキーワードのルールに切り替える。
画像の色検出(06d の検出器)の結果も渡し、卓上に無い色を段取りに入れないよう検証する。

まだ教わっていない物(例: 消しゴム)も、頼まれたら steps の最後に入れて試す(new に入る)。
返事で「初めてなので、うまくいかなかったら手伝ってください」と添え、失敗したら人が手伝う(その動きが教材になる)。
扱えない物(指示文が無い物)は unknown に入れて断る。教わった物は objects.py の taught()(outputs/skills.json)。
2026-10-28: 新しい物を教わって育つ場面のために、キューブ 3 色以外(消しゴム)を扱えるようにした。
"""

from __future__ import annotations

import base64
import os
import sys
from typing import List, Literal, Optional

import cv2
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

audit = import_module("06d_audit_dataset")
objects = import_module("objects")

COLORS = tuple(objects.CUBES)
JA = objects.JA
TASK = "Pick up the {} cube and put it in the box"
MODEL = os.environ.get("PLANNER_MODEL", "claude-opus-5-5")


class Plan(BaseModel):
    steps: List[Literal["red", "yellow", "purple", "eraser"]] = Field(
        description="箱に入れる順の物(キューブは色、消しゴムは eraser)。教わった物だけ。何もしない場合は空")
    unknown: List[str] = Field(default_factory=list,
                               description="片付けるよう言われたが、このロボットでは扱えない物の日本語名(例: ペン)。無ければ空")
    new: List[str] = Field(default_factory=list, description="(こちらで付ける)steps のうち、まだ教わっていない物")
    reply: str = Field(description="ロボットとして人に返す短い日本語の一言(読み上げる)")
    reason: str = Field(description="そう解釈した理由(日本語、1 文)")
    source: Optional[str] = None   # "vlm" か "rule"(こちらで付ける)


def scene(top_bgr) -> dict:
    """画像から各色のキューブが卓上にあるか・箱の中にあるかを判定する。"""
    box = audit.find_box(top_bgr)
    out = {}
    for c in COLORS:
        p = audit.find_cube(top_bgr, c)
        out[c] = "none" if p is None else ("in_box" if audit.inside(p, box) else "table")
    return out


def rule_plan(instruction: str, sc: dict) -> Plan:
    """VLM が使えないときのキーワード対応。"""
    s = instruction
    names = {"red": ("赤", "red", "あか"), "yellow": ("黄", "yellow", "きいろ"), "purple": ("紫", "purple", "むらさき")}
    # 箱に入っていない色を対象にする(色検出が照明で外れることがあるので、見つからない色も残す)
    on_table = [c for c in COLORS if sc.get(c) != "in_box"]
    named = [c for c in COLORS if any(n in s for n in names[c])]
    extra = []   # キューブ以外の物(教わっていなくても steps の最後に入れて試す)
    if any(n in s for n in ("消しゴム", "けしごむ", "eraser")):
        extra = ["eraser"]
    if "以外" in s or "除いて" in s or "except" in s.lower():
        steps = [c for c in on_table if c not in named]
    elif named:
        steps = [c for c in named if c in on_table]
    elif any(k in s for k in ("片付け", "かたづけ", "全部", "すべて", "tidy", "clean")):
        steps = on_table
    else:
        steps = []
    steps += [e for e in extra if e not in steps]
    reply = ("、".join(JA[c] for c in steps) + "を箱に入れます。") if steps else "すみません、何を片付けるか分かりませんでした。"
    return Plan(steps=steps, reply=reply, reason="キーワードで判定", source="rule")


SYSTEM = """あなたは卓上の片付けロボットの段取り係です。
ロボットができるのは「物を 1 個つかんで箱に入れる」ことです。
教わった物: {taught}
まだ教わっていないが、試すことはできる物: {new}
人の指示と俯瞰カメラの画像から、箱に入れる物を、入れる順に並べてください(steps。キューブは色 red / yellow / purple、消しゴムは eraser)。

- 卓上(箱の外)にある物だけが対象です。すでに箱の中にあるもの、見当たらないものは入れません。
- 「片付けて」のように対象を言わなければ、卓上の教わった物を全部入れます。順番は、画像で箱に近いものから。
- 「〜以外」「〜だけ」「暖色」「信号の色」などは意味を解釈してください。
- まだ教わっていない物も、片付けるよう言われたら(または「片付けて」で卓上にあれば)steps の最後に入れます。
  そのときの reply では「〇〇は初めてなので、うまくいかなかったら手伝ってください」と添えます。
- 上のどちらにも無い物(ペンなど)は steps に入れず unknown に日本語名を入れ、reply で扱えないことを伝えます。
- できない指示(箱から出す、存在しない物など)なら steps を空にし、reply でできないことを丁寧に伝えます。
- reply はロボットが読み上げる一言です。短く、自然な日本語で。"""


def vlm_plan(instruction: str, top_bgr, sc: dict) -> Plan:
    import anthropic

    ok, jpg = cv2.imencode(".jpg", top_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img = base64.standard_b64encode(jpg.tobytes()).decode()
    detected = ", ".join(f"{c}: {'卓上' if v == 'table' else '箱の中' if v == 'in_box' else '見当たらない'}" for c, v in sc.items())
    client = anthropic.Anthropic()
    resp = client.beta.messages.parse(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM.format(
            taught="、".join(("消しゴム(eraser)" if t == "eraser" else f"{JA[t]}のキューブ({t})") for t in objects.taught()),
            new="、".join(f"{JA[t]}({t})" for t in objects.OBJECTS if t not in objects.taught()) or "なし"),
        output_config={"effort": "low"},    # 段取りは単純で、待ち時間を短くしたい
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
            {"type": "text", "text": f"色検出の結果(参考): {detected}\n指示: {instruction}"},
        ]}],
        output_format=Plan,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"VLM が段取りを返さなかった(stop_reason={resp.stop_reason})")
    plan = resp.parsed_output
    plan.source = "vlm"
    return plan


def make_plan(instruction: str, top_bgr) -> Plan:
    sc = scene(top_bgr)
    plan = None
    if os.environ.get("PLANNER", "vlm") == "vlm" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            plan = vlm_plan(instruction, top_bgr, sc)
        except Exception as e:  # 通信・鍵・拒否などはルールで続行する(デモを止めない)
            print(f"[planner] VLM を使えずルールに切り替え: {type(e).__name__}: {e}", file=sys.stderr)
    if plan is None:
        plan = rule_plan(instruction, sc)
    # 検証: まだ教わっていない物は最後に回して試す(失敗したら人が手伝い、その動きが教材になる)
    # 2026-10-28: 最初は「教わっていない物は断って教えてと頼む」にしたが、教わった物の一覧を引いているだけで
    # 中身が浅い。「まず試して、駄目なら人が手伝う」に変え、キューブと同じ改善の輪で新しい物も覚えるようにした。
    by_ja = {v: k for k, v in JA.items()}
    for name in list(plan.unknown):          # VLM が「扱えない」に入れた物でも、指示文がある物は試す
        if name in by_ja and by_ja[name] not in plan.steps:
            plan.steps.append(by_ja[name])
            plan.unknown.remove(name)
    taught = objects.taught()
    plan.new = [c for c in plan.steps if c not in taught]
    plan.steps = [c for c in plan.steps if c in taught] + plan.new
    if plan.new and "初めて" not in plan.reply:
        plan.reply += f" {'、'.join(JA[c] for c in plan.new)}は初めてなので、うまくいかなかったら手伝ってください。"
    # 検証: すでに箱の中にある色は実行しない。重複も除く
    # (見つからない色は残す。2026-10-25: 白飛びした黄を検出できず「黄色と紫」の黄が消えたため)
    seen, steps = set(), []
    for c in plan.steps:
        if sc.get(c) != "in_box" and c not in seen:
            steps.append(c)
            seen.add(c)
    if steps != plan.steps:
        print(f"[planner] 箱の中の色・重複を除外: {plan.steps} → {steps}", file=sys.stderr)
        plan.steps = steps
    return plan


if __name__ == "__main__":
    instr = sys.argv[1] if len(sys.argv) > 1 else "片付けて"
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "outputs", "cam_test", "ref_top_so101_cube_v3_s1.jpg")
    img = cv2.resize(cv2.imread(path), (640, 360))
    print("検出:", scene(img))
    p = make_plan(instr, img)
    print(p.model_dump_json(indent=2))
    for c in p.steps:
        print("  →", TASK.format(c))
