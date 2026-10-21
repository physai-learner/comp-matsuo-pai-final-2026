"""評価に使う指示と、その正解（どの物をどの皿へ）。

正解が配置で決まる指示（「一番左の物」など）は、シミュレータの正解位置から都度求める。
画像の左＝ワールド +y、画像の下（ロボット側）＝ワールド x が小さい側。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tidy_arm.sim import ITEMS, PLATES, Sim

ITEM_JA = {"red_box": "赤い箱", "green_cylinder": "緑の円柱", "purple_box": "紫の箱"}
PLATE_JA = {"plate_blue": "青い皿", "plate_yellow": "黄色い皿"}

Moves = list[tuple[str, str]]
# 位置関係の比べる量の差がこれ未満なら僅差とみなし、どちらを選んでも、判断を保留しても正解にする
TIE = 0.02


@dataclass(frozen=True)
class Task:
    name: str
    category: str  # basic / paraphrase / multi / spatial / infeasible
    instruction: str
    # 関数なら配置から正解の候補を求める（僅差で複数あれば、どれでも・保留しても正解）
    moves: Moves | Callable[[Sim], list[Moves]] = ()
    # True なら計画係は moves を空にして problem を返すのが正解
    expect_problem: bool = False
    # expect_problem の指示でも、計画を出して正解になる読み方（解釈が割れるもの）
    also: tuple[Moves, ...] = ()

    def accepted(self, sim: Sim) -> tuple[list[Moves], bool]:
        """正解になる計画の候補と、保留（moves を空にして problem）が正解かどうか。"""
        if self.expect_problem:
            return [list(m) for m in self.also], True
        if callable(self.moves):
            answers = self.moves(sim)
            return answers, len(answers) > 1
        return [list(self.moves)], False

    def expected(self, sim: Sim) -> Moves:
        """代表の正解（正解計画とテストで使う）。保留が正解の指示では空。"""
        return [] if self.expect_problem else self.accepted(sim)[0][0]


def _xy(sim: Sim, name: str) -> np.ndarray:
    return (sim.item_pos(name) if name in ITEMS else sim.plate_pos(name))[:2]


def _best(names, key) -> list[str]:
    """key が最小のもの。差が TIE 未満の僅差は全部返す（最小のものが先頭）。"""
    ranked = sorted(names, key=key)
    return [n for n in ranked if key(n) - key(ranked[0]) < TIE]


def _leftmost(sim: Sim) -> list[str]:
    return _best(ITEMS, lambda n: -_xy(sim, n)[1])


def _rightmost(sim: Sim) -> list[str]:
    return _best(ITEMS, lambda n: _xy(sim, n)[1])


def _nearest_robot(sim: Sim) -> list[str]:
    return _best(ITEMS, lambda n: np.linalg.norm(_xy(sim, n)))


def _nearest(sim: Sim, target: str, names) -> list[str]:
    return _best(names, lambda n: np.linalg.norm(_xy(sim, n) - _xy(sim, target)))


def _farthest(sim: Sim, target: str, names) -> list[str]:
    return _best(names, lambda n: -np.linalg.norm(_xy(sim, n) - _xy(sim, target)))


def _basic() -> list[Task]:
    return [
        Task(f"basic_{i}_{p}", "basic", f"{ITEM_JA[i]}を{PLATE_JA[p]}に置いて", [(i, p)])
        for i in ITEMS
        for p in PLATES
    ]


TASKS: list[Task] = [
    *_basic(),
    # 言い換え：同じ物を別の言葉で呼ぶ・語順を変える
    Task("para_block", "paraphrase", "赤いブロックを青いお皿へ移して", [("red_box", "plate_blue")]),
    Task(
        "para_tube", "paraphrase", "緑の筒を黄色の皿にのせて", [("green_cylinder", "plate_yellow")]
    ),
    Task(
        "para_cube",
        "paraphrase",
        "紫のキューブ、青い皿に片付けておいて",
        [("purple_box", "plate_blue")],
    ),
    Task("para_order", "paraphrase", "黄色い皿に赤い箱を運んで", [("red_box", "plate_yellow")]),
    Task("para_shape", "paraphrase", "円柱を青い皿へ", [("green_cylinder", "plate_blue")]),
    Task(
        "para_thing", "paraphrase", "緑色のやつを黄色い皿に", [("green_cylinder", "plate_yellow")]
    ),
    # 複数：順に運ぶ・同じ皿に2つ置く
    Task(
        "multi_two",
        "multi",
        "赤い箱は青い皿、緑の円柱は黄色い皿へ",
        [("red_box", "plate_blue"), ("green_cylinder", "plate_yellow")],
    ),
    Task(
        "multi_boxes",
        "multi",
        "箱を全部青い皿に集めて",
        [("red_box", "plate_blue"), ("purple_box", "plate_blue")],
    ),
    Task(
        "multi_all",
        "multi",
        "全部片付けて。赤いのは青い皿、ほかは黄色い皿に",
        [
            ("red_box", "plate_blue"),
            ("green_cylinder", "plate_yellow"),
            ("purple_box", "plate_yellow"),
        ],
    ),
    Task(
        "multi_swap_color",
        "multi",
        "紫の箱を黄色い皿に、それから緑の円柱を青い皿に",
        [("purple_box", "plate_yellow"), ("green_cylinder", "plate_blue")],
    ),
    Task(
        "multi_except",
        "multi",
        "赤い箱以外を青い皿へ",
        [("green_cylinder", "plate_blue"), ("purple_box", "plate_blue")],
    ),
    # 位置関係：正解は配置で決まる
    Task(
        "spatial_left",
        "spatial",
        "一番左にある物を青い皿へ",
        lambda s: [[(i, "plate_blue")] for i in _leftmost(s)],
    ),
    Task(
        "spatial_robot",
        "spatial",
        "ロボットに一番近い物を黄色い皿へ",
        lambda s: [[(i, "plate_yellow")] for i in _nearest_robot(s)],
    ),
    Task(
        "spatial_near_plate",
        "spatial",
        "青い皿に一番近い物を青い皿にのせて",
        lambda s: [[(i, "plate_blue")] for i in _nearest(s, "plate_blue", ITEMS)],
    ),
    Task(
        "spatial_far_plate",
        "spatial",
        "一番右の物を、その物から遠い方の皿へ",
        lambda s: [[(i, p)] for i in _rightmost(s) for p in _farthest(s, i, PLATES)],
    ),
    Task(
        "spatial_near_red",
        "spatial",
        "赤い箱を、赤い箱に近い方の皿に置いて",
        lambda s: [[("red_box", p)] for p in _nearest(s, "red_box", PLATES)],
    ),
    # 実行できない・決められない：problem を返すのが正解
    Task("infeasible_item", "infeasible", "青い箱を黄色い皿へ", expect_problem=True),
    Task("infeasible_plate", "infeasible", "赤い箱を緑の皿に置いて", expect_problem=True),
    # 真上からは円柱が球にも見える（初回の評価後に判明）。視点の限界の例として正解は変えない
    Task("infeasible_ball", "infeasible", "ボールを青い皿へ", expect_problem=True),
    # 「箱を」は「箱を全部」とも読める＝箱2つを運ぶのも正解
    Task(
        "ambiguous_box",
        "infeasible",
        "箱を青い皿へ",
        expect_problem=True,
        also=([("red_box", "plate_blue"), ("purple_box", "plate_blue")],),
    ),
    # 上の2件と同じ狙いで、解釈が割れないように書いたもの
    Task("infeasible_white_ball", "infeasible", "白いボールを青い皿へ", expect_problem=True),
    Task("ambiguous_that_box", "infeasible", "さっきの箱を青い皿へ戻して", expect_problem=True),
]
