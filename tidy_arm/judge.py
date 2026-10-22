"""計画の採点。ROS なし版（scripts/evaluate.py）と ROS 2 版（tidy_arm_ros の evaluate）で共有する。

scene は物と皿の位置を返すもの（`item_pos(name)`・`plate_pos(name)`）。ROS なし版は Sim そのもの、
ROS 2 版は sim ノードの get_state から作った `SceneState` を渡す。

結果（1行＝1試行）:
  ok            正解の物が正解の皿に載った（実行できない指示では、計画を出さず理由を返した）。
                位置関係が僅差（2cm 未満）の配置では、どちらを選んでも・保留しても正解
  plan:<kind>   計画を得られなかった（refusal / bad_output / out_of_image）
  no_plan       実行できる指示なのに計画を出さなかった
  should_refuse 実行できない指示なのに計画を出した
  wrong_count   運ぶ件数が違う
  wrong_item    つかむ画素が正解の物に当たっていない（取り違え・位置ずれ）
  wrong_place   置く画素が正解の皿に当たっていない
  exec_fail     計画は正しいが、運び終えたあと皿に載っていない（把持・運搬の失敗）
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from tidy_arm.camera import Camera
from tidy_arm.planner import ITEM_TOP_Z, PLATE_TOP_Z, Plan
from tidy_arm.sim import ITEMS, PLATE_RADIUS, PLATES

# つかむ画素を机上へ戻した点が、物の中心からこれ以内なら「その物を指した」とみなす
PICK_TOL = 0.035

Pairs = list[tuple[str | None, str | None]]


@dataclass
class SceneState:
    """物と皿の位置だけを持つ scene（ROS 2 版で sim ノードから受け取ったもの）。"""

    items: dict[str, np.ndarray]
    plates: dict[str, np.ndarray]

    def item_pos(self, name: str) -> np.ndarray:
        return self.items[name]

    def plate_pos(self, name: str) -> np.ndarray:
        return self.plates[name]


def ground(scene, cam: Camera, plan: Plan) -> tuple[Pairs, list[float]]:
    """計画の各画素が、実際にはどの物・どの皿を指しているか。つかむ点の誤差（m）も返す。"""
    pairs, errors = [], []
    for m in plan.moves:
        pick = cam.pixel_to_world([m.pick.u, m.pick.v], z=ITEM_TOP_Z)[:2]
        place = cam.pixel_to_world([m.place.u, m.place.v], z=PLATE_TOP_Z)[:2]
        d_item = {n: float(np.linalg.norm(pick - scene.item_pos(n)[:2])) for n in ITEMS}
        d_plate = {n: float(np.linalg.norm(place - scene.plate_pos(n)[:2])) for n in PLATES}
        item = min(d_item, key=d_item.get)
        plate = min(d_plate, key=d_plate.get)
        errors.append(d_item[item])
        pairs.append(
            (
                item if d_item[item] < PICK_TOL else None,
                plate if d_plate[plate] < PLATE_RADIUS else None,
            )
        )
    return pairs, errors


def check_plan(
    scene, cam: Camera, plan: Plan, answers: list, problem_ok: bool
) -> tuple[str | None, list | None]:
    """動かす前に決まる結果。決まれば (結果, None)、正しい計画なら (None, 合致した正解)。

    answers＝正解になる計画の候補（僅差・解釈の割れで複数あり得る）。
    """
    if not plan.moves:
        return ("ok" if problem_ok else "no_plan"), None
    if not answers:
        return "should_refuse", None
    same_count = [a for a in answers if len(a) == len(plan.moves)]
    if not same_count:
        return "wrong_count", None
    pairs, _ = ground(scene, cam, plan)
    items = Counter(i for i, _ in pairs)
    same_items = [a for a in same_count if Counter(i for i, _ in a) == items]
    if not same_items:
        return "wrong_item", None
    match = next((a for a in same_items if Counter(a) == Counter(pairs)), None)
    if match is None:
        return "wrong_place", None
    return None, match


def summarize(rows: list[dict]) -> None:
    """評価の CSV の行（dict）から、言語ごと・区分ごとの成功数と失敗の内訳を表示する。"""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[(r["lang"], r["category"])].append(r["result"] == "ok")
    for lang in dict.fromkeys(r["lang"] for r in rows):
        lr = [r for r in rows if r["lang"] == lang]
        print(f"\n[{lang}] 成功 {sum(r['result'] == 'ok' for r in lr)}/{len(lr)}")
        for (lg, cat), oks in by_cat.items():
            if lg == lang:
                print(f"  {cat:12s} {sum(oks)}/{len(oks)}")
        fails = Counter(r["result"] for r in lr if r["result"] != "ok")
        if fails:
            print("  失敗の内訳:", dict(fails))
        tin = sum(int(r["input_tokens"]) for r in lr)
        tout = sum(int(r["output_tokens"]) for r in lr)
        if tin:
            print(f"  トークン 入力 {tin:,}・出力 {tout:,}")
