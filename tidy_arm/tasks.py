"""評価に使う指示と、その正解（どの物をどの皿へ）。指示は日本語と英語の2通り（意味を揃えて訳した）。

正解が配置で決まる指示（「一番左の物」など）は、シミュレータの正解位置から都度求める。
画像の左＝ワールド +y、画像の下（ロボット側）＝ワールド x が小さい側。
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tidy_arm.sim import ITEMS, PLATES, Sim

ITEM_JA = {"red_box": "赤い箱", "green_cylinder": "緑の円柱", "purple_box": "紫の箱"}
PLATE_JA = {"plate_blue": "青い皿", "plate_yellow": "黄色い皿"}
ITEM_EN = {"red_box": "red box", "green_cylinder": "green cylinder", "purple_box": "purple box"}
PLATE_EN = {"plate_blue": "blue plate", "plate_yellow": "yellow plate"}
LANGS = ("ja", "en")

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
    en: str = ""  # 英語の指示
    # 英語で also が変わるとき（冠詞・単複で数が決まり、日本語で割れた読みが割れなくなるもの）
    en_also: tuple[Moves, ...] | None = None

    def text(self, lang: str = "ja") -> str:
        return self.en if lang == "en" else self.instruction

    def accepted(self, sim: Sim, lang: str = "ja") -> tuple[list[Moves], bool]:
        """正解になる計画の候補と、保留（moves を空にして problem）が正解かどうか。"""
        if self.expect_problem:
            also = self.en_also if lang == "en" and self.en_also is not None else self.also
            return [list(m) for m in also], True
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
        Task(
            f"basic_{i}_{p}",
            "basic",
            f"{ITEM_JA[i]}を{PLATE_JA[p]}に置いて",
            [(i, p)],
            en=f"Put the {ITEM_EN[i]} on the {PLATE_EN[p]}",
        )
        for i in ITEMS
        for p in PLATES
    ]


TASKS: list[Task] = [
    *_basic(),
    # 言い換え：同じ物を別の言葉で呼ぶ・語順を変える
    Task(
        "para_block",
        "paraphrase",
        "赤いブロックを青いお皿へ移して",
        [("red_box", "plate_blue")],
        en="Move the red block over to the blue dish",
    ),
    Task(
        "para_tube",
        "paraphrase",
        "緑の筒を黄色の皿にのせて",
        [("green_cylinder", "plate_yellow")],
        en="Set the green tube on the yellow plate",
    ),
    Task(
        "para_cube",
        "paraphrase",
        "紫のキューブ、青い皿に片付けておいて",
        [("purple_box", "plate_blue")],
        en="The purple cube, tidy it away onto the blue plate",
    ),
    Task(
        "para_order",
        "paraphrase",
        "黄色い皿に赤い箱を運んで",
        [("red_box", "plate_yellow")],
        en="Onto the yellow plate, carry the red box",
    ),
    Task(
        "para_shape",
        "paraphrase",
        "円柱を青い皿へ",
        [("green_cylinder", "plate_blue")],
        en="Cylinder to the blue plate",
    ),
    Task(
        "para_thing",
        "paraphrase",
        "緑色のやつを黄色い皿に",
        [("green_cylinder", "plate_yellow")],
        en="The green thing, onto the yellow plate",
    ),
    # 複数：順に運ぶ・同じ皿に2つ置く
    Task(
        "multi_two",
        "multi",
        "赤い箱は青い皿、緑の円柱は黄色い皿へ",
        [("red_box", "plate_blue"), ("green_cylinder", "plate_yellow")],
        en="Red box to the blue plate, green cylinder to the yellow plate",
    ),
    Task(
        "multi_boxes",
        "multi",
        "箱を全部青い皿に集めて",
        [("red_box", "plate_blue"), ("purple_box", "plate_blue")],
        en="Gather all the boxes on the blue plate",
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
        en="Clear everything away. The red one goes on the blue plate, the rest on the yellow one",
    ),
    Task(
        "multi_swap_color",
        "multi",
        "紫の箱を黄色い皿に、それから緑の円柱を青い皿に",
        [("purple_box", "plate_yellow"), ("green_cylinder", "plate_blue")],
        en="Put the purple box on the yellow plate, then the green cylinder on the blue plate",
    ),
    Task(
        "multi_except",
        "multi",
        "赤い箱以外を青い皿へ",
        [("green_cylinder", "plate_blue"), ("purple_box", "plate_blue")],
        en="Everything except the red box to the blue plate",
    ),
    # 位置関係：正解は配置で決まる
    Task(
        "spatial_left",
        "spatial",
        "一番左にある物を青い皿へ",
        lambda s: [[(i, "plate_blue")] for i in _leftmost(s)],
        en="Put the leftmost object on the blue plate",
    ),
    Task(
        "spatial_robot",
        "spatial",
        "ロボットに一番近い物を黄色い皿へ",
        lambda s: [[(i, "plate_yellow")] for i in _nearest_robot(s)],
        en="Put the object closest to the robot on the yellow plate",
    ),
    Task(
        "spatial_near_plate",
        "spatial",
        "青い皿に一番近い物を青い皿にのせて",
        lambda s: [[(i, "plate_blue")] for i in _nearest(s, "plate_blue", ITEMS)],
        en="Put the object closest to the blue plate onto the blue plate",
    ),
    Task(
        "spatial_far_plate",
        "spatial",
        "一番右の物を、その物から遠い方の皿へ",
        lambda s: [[(i, p)] for i in _rightmost(s) for p in _farthest(s, i, PLATES)],
        en="Take the rightmost object to whichever plate is farther from it",
    ),
    Task(
        "spatial_near_red",
        "spatial",
        "赤い箱を、赤い箱に近い方の皿に置いて",
        lambda s: [[("red_box", p)] for p in _nearest(s, "red_box", PLATES)],
        en="Put the red box on the plate closer to the red box",
    ),
    # 実行できない・決められない：problem を返すのが正解
    Task(
        "infeasible_item",
        "infeasible",
        "青い箱を黄色い皿へ",
        expect_problem=True,
        en="Blue box to the yellow plate",
    ),
    Task(
        "infeasible_plate",
        "infeasible",
        "赤い箱を緑の皿に置いて",
        expect_problem=True,
        en="Put the red box on the green plate",
    ),
    # 真上からは円柱が球にも見える（初回の評価後に判明）。視点の限界の例として正解は変えない
    Task(
        "infeasible_ball",
        "infeasible",
        "ボールを青い皿へ",
        expect_problem=True,
        en="Ball to the blue plate",
    ),
    # 「箱を」は「箱を全部」とも読める＝箱2つを運ぶのも正解。
    # 英語は単数の "the box" なので1個に決まる＝どれか決められず保留するのだけが正解
    Task(
        "ambiguous_box",
        "infeasible",
        "箱を青い皿へ",
        expect_problem=True,
        also=([("red_box", "plate_blue"), ("purple_box", "plate_blue")],),
        en="The box to the blue plate",
        en_also=(),
    ),
    # 上の2件と同じ狙いで、解釈が割れないように書いたもの
    Task(
        "infeasible_white_ball",
        "infeasible",
        "白いボールを青い皿へ",
        expect_problem=True,
        en="White ball to the blue plate",
    ),
    Task(
        "ambiguous_that_box",
        "infeasible",
        "さっきの箱を青い皿へ戻して",
        expect_problem=True,
        en="Put the box from before back on the blue plate",
    ),
]

assert all(t.en for t in TASKS), "英語の指示が無いタスクがある"
