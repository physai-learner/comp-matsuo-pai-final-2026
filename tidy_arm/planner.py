"""日本語の指示と真上画像から「どれを・どこへ」運ぶかを画素で決め、アームで実行する。

計画係は2種類。`ClaudePlanner` は画像と指示を Claude に渡して画素を返させる。
`OraclePlanner` はシミュレータの正解位置から同じ形の計画を作る（キー無しで配線を通す・上限の目安）。
"""

import base64
from dataclasses import dataclass, field
from typing import Protocol

import imageio.v3 as iio
import numpy as np
from pydantic import BaseModel

from tidy_arm.camera import Camera
from tidy_arm.sim import Sim

MODEL = "claude-opus-5-5"
# 画素を机上の点へ戻すときの高さ。物の上面（箱 0.044・円柱 0.06）の中間に固定する。
# 1cm ずれても視差は作業範囲で 3mm 程度＝指の開き 8cm に対して無視できる
ITEM_TOP_Z = 0.05
PLATE_TOP_Z = 0.008
# 把持点の高さ。置くときも同じ高さまで下ろす。低いと円柱（高さ 6cm）の上端が指の付け根に当たり、
# 指が閉じ切らない（2.5cm で失敗・3cm で改善を実測）。箱（高さ 4.4cm）も指先の当たる範囲に収まる
GRASP_Z = 0.03
# 同じ皿に2つ置くときの間隔（皿の半径 0.07 の内側に収まる）
PLATE_SPACING = 0.06

SYSTEM_PROMPT = """\
あなたはロボットアームの作業計画を立てる係です。
画像は机を真上から撮った写真です。机の上に物がいくつかと、皿が置いてあります。
画像の下端側にロボットアームの根元があり、アームは作業の対象ではありません。

利用者の指示に従って、運ぶ物ごとに1件ずつ、実行する順に moves へ書いてください。
- item: 運ぶ物の見た目を短い日本語で（例: 赤い箱）
- pick: その物の上面の中心の画素
- place_target: 置き先の見た目を短い日本語で（例: 青い皿）
- place: 置き先の皿の中心の画素。同じ皿に2つ置くときは、重ならないよう
  皿の中心から左右へ約15画素ずつずらした点にする（皿の直径は約80画素）
画素は画像の左上が原点で、u が右向き、v が下向き、単位は画素です。
画像の大きさは利用者のメッセージに書きます。

指示に合う物や置き先が画像に無いとき、どれを指すか決められないときは、
moves を空にして problem に理由を1文で書いてください。問題が無いときの problem は空文字です。
"""


class Pixel(BaseModel):
    u: float
    v: float


class Move(BaseModel):
    item: str
    pick: Pixel
    place_target: str
    place: Pixel


class Plan(BaseModel):
    moves: list[Move]
    problem: str


class PlanError(Exception):
    """計画を得られなかった（拒否・形式違反・画像の外の画素など）。評価では失敗の分類に使う。"""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


class Planner(Protocol):
    def plan(self, image: np.ndarray, instruction: str) -> Plan: ...


def check_bounds(plan: Plan, width: int, height: int) -> None:
    for i, m in enumerate(plan.moves):
        for role, p in (("pick", m.pick), ("place", m.place)):
            if not (0 <= p.u < width and 0 <= p.v < height):
                raise PlanError("out_of_image", f"moves[{i}].{role}=({p.u}, {p.v})")


@dataclass
class ClaudePlanner:
    model: str = MODEL
    effort: str = "medium"
    max_tokens: int = 16000
    client: object | None = None
    usage_log: list[dict] = field(default_factory=list)

    def _client(self):
        if self.client is None:
            import anthropic
            from dotenv import load_dotenv

            load_dotenv()
            self.client = anthropic.Anthropic()
        return self.client

    @staticmethod
    def user_content(image: np.ndarray, instruction: str) -> list[dict]:
        h, w = image.shape[:2]
        png = iio.imwrite("<bytes>", image, extension=".png")
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode("ascii"),
                },
            },
            {
                "type": "text",
                "text": f"画像の大きさ: 幅 {w} 画素 × 高さ {h} 画素\n指示: {instruction}",
            },
        ]

    def plan(self, image: np.ndarray, instruction: str) -> Plan:
        resp = self._client().beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self.user_content(image, instruction)}],
            output_config={"effort": self.effort},
            output_format=Plan,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage_log.append(
            {
                "model": resp.model,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        )
        if resp.stop_reason == "refusal":
            detail = resp.stop_details.category if resp.stop_details else None
            raise PlanError("refusal", f"category={detail}")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise PlanError("bad_output", f"stop_reason={resp.stop_reason}")
        plan = resp.parsed_output
        h, w = image.shape[:2]
        check_bounds(plan, w, h)
        return plan


@dataclass
class OraclePlanner:
    """正解位置を投影して計画を作る。指示と画像は見ない。"""

    sim: Sim
    moves: list[tuple[str, str]]  # (物, 皿) の実行順
    problem: str = ""  # 空でなければ moves を無視して「計画できない」と返す

    def plan(self, image: np.ndarray, instruction: str) -> Plan:
        if self.problem:
            return Plan(moves=[], problem=self.problem)
        cam = self.sim.camera()

        def px(point: np.ndarray) -> Pixel:
            u, v = cam.world_to_pixel(point)
            return Pixel(u=float(u), v=float(v))

        plates = [p for _, p in self.moves]
        out = []
        for k, (item, plate) in enumerate(self.moves):
            half_h = self.sim.model.geom_size[self.sim.model.body(item).geomadr[0]][-1]
            top = self.sim.item_pos(item) + np.array([0, 0, half_h])
            # 同じ皿に複数置くときは、画像の左右（ワールド y）へずらして重ねない
            n, idx = plates.count(plate), plates[:k].count(plate)
            place = self.sim.plate_pos(plate) + np.array(
                [0, (idx - (n - 1) / 2) * PLATE_SPACING, 0]
            )
            out.append(Move(item=item, pick=px(top), place_target=plate, place=px(place)))
        return Plan(moves=out, problem="")


def to_world(cam: Camera, move: Move) -> tuple[np.ndarray, np.ndarray]:
    """1件の計画の画素を、つかむ点・置く点（机上の xy・z は把持高さ）へ戻す。"""
    pick = cam.pixel_to_world([move.pick.u, move.pick.v], z=ITEM_TOP_Z)
    place = cam.pixel_to_world([move.place.u, move.place.v], z=PLATE_TOP_Z)
    pick[2] = place[2] = GRASP_Z
    return pick, place


def execute(sim: Sim, plan: Plan) -> None:
    """計画の画素を机上の点へ戻し、順に運ぶ。カメラは固定なので最初に1回だけ取る。"""
    cam = sim.camera()
    for m in plan.moves:
        sim.pick_and_place(*to_world(cam, m))
    sim.settle(0.5)
