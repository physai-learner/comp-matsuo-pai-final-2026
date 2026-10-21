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


@dataclass(frozen=True)
class Task:
    name: str
    category: str  # basic / paraphrase / multi / spatial / infeasible
    instruction: str
    moves: Moves | Callable[[Sim], Moves] = ()
    # True なら計画係は moves を空にして problem を返すのが正解
    expect_problem: bool = False

    def expected(self, sim: Sim) -> Moves:
        return self.moves(sim) if callable(self.moves) else list(self.moves)


def _xy(sim: Sim, name: str) -> np.ndarray:
    return (sim.item_pos(name) if name in ITEMS else sim.plate_pos(name))[:2]


def _leftmost(sim: Sim) -> str:
    return max(ITEMS, key=lambda n: _xy(sim, n)[1])


def _rightmost(sim: Sim) -> str:
    return min(ITEMS, key=lambda n: _xy(sim, n)[1])


def _nearest_robot(sim: Sim) -> str:
    return min(ITEMS, key=lambda n: np.linalg.norm(_xy(sim, n)))


def _nearest(sim: Sim, target: str, names) -> str:
    return min(names, key=lambda n: np.linalg.norm(_xy(sim, n) - _xy(sim, target)))


def _farthest(sim: Sim, target: str, names) -> str:
    return max(names, key=lambda n: np.linalg.norm(_xy(sim, n) - _xy(sim, target)))


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
        lambda s: [(_leftmost(s), "plate_blue")],
    ),
    Task(
        "spatial_robot",
        "spatial",
        "ロボットに一番近い物を黄色い皿へ",
        lambda s: [(_nearest_robot(s), "plate_yellow")],
    ),
    Task(
        "spatial_near_plate",
        "spatial",
        "青い皿に一番近い物を青い皿にのせて",
        lambda s: [(_nearest(s, "plate_blue", ITEMS), "plate_blue")],
    ),
    Task(
        "spatial_far_plate",
        "spatial",
        "一番右の物を、その物から遠い方の皿へ",
        lambda s: [(_rightmost(s), _farthest(s, _rightmost(s), PLATES))],
    ),
    Task(
        "spatial_near_red",
        "spatial",
        "赤い箱を、赤い箱に近い方の皿に置いて",
        lambda s: [("red_box", _nearest(s, "red_box", PLATES))],
    ),
    # 実行できない・決められない：problem を返すのが正解
    Task("infeasible_item", "infeasible", "青い箱を黄色い皿へ", expect_problem=True),
    Task("infeasible_plate", "infeasible", "赤い箱を緑の皿に置いて", expect_problem=True),
    Task("infeasible_ball", "infeasible", "ボールを青い皿へ", expect_problem=True),
    Task("ambiguous_box", "infeasible", "箱を青い皿へ", expect_problem=True),
]
