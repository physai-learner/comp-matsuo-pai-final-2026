"""計画（画素）→ 机上の点 → 把持・運搬の配線。Claude は呼ばず、正解計画と偽のクライアントで通す。"""

import base64
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest

from tidy_arm.planner import (
    ClaudePlanner,
    Move,
    OraclePlanner,
    Pixel,
    Plan,
    PlanError,
    check_bounds,
    execute,
)
from tidy_arm.sim import Sim


def test_oracle_plan_executes_two_moves():
    sim = Sim(seed=3)
    try:
        moves = [("red_box", "plate_blue"), ("green_cylinder", "plate_yellow")]
        plan = OraclePlanner(sim, moves).plan(sim.render(), "（指示は見ない）")
        execute(sim, plan)
        for item, plate in moves:
            assert sim.on_plate(item, plate), f"{item} -> {plate}: {sim.item_pos(item)}"
    finally:
        sim.close()


def test_check_bounds_rejects_outside_pixels():
    ok = Move(item="a", pick=Pixel(u=10, v=10), place_target="b", place=Pixel(u=639, v=479))
    check_bounds(Plan(moves=[ok], problem=""), 640, 480)
    bad = ok.model_copy(update={"place": Pixel(u=640, v=10)})
    with pytest.raises(PlanError) as e:
        check_bounds(Plan(moves=[bad], problem=""), 640, 480)
    assert e.value.kind == "out_of_image"


def test_user_content_carries_png_and_size():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[10, 20] = (255, 0, 0)
    content = ClaudePlanner.user_content(img, "赤い箱を青い皿へ")
    decoded = iio.imread(base64.standard_b64decode(content[0]["source"]["data"]))
    np.testing.assert_array_equal(decoded, img)
    assert "幅 640" in content[1]["text"] and "赤い箱を青い皿へ" in content[1]["text"]


def _fake_client(**resp):
    usage = SimpleNamespace(input_tokens=1, output_tokens=2)
    base = dict(model="m", usage=usage, stop_reason="end_turn", stop_details=None)
    response = SimpleNamespace(**{**base, **resp})
    parse = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))


IMG = np.zeros((480, 640, 3), dtype=np.uint8)


def test_claude_planner_returns_parsed_plan():
    move = Move(item="赤い箱", pick=Pixel(u=1, v=2), place_target="青い皿", place=Pixel(u=3, v=4))
    planner = ClaudePlanner(client=_fake_client(parsed_output=Plan(moves=[move], problem="")))
    assert planner.plan(IMG, "x").moves == [move]
    assert planner.usage_log == [{"model": "m", "input_tokens": 1, "output_tokens": 2}]


def test_claude_planner_classifies_refusal():
    details = SimpleNamespace(category="other")
    planner = ClaudePlanner(
        client=_fake_client(stop_reason="refusal", stop_details=details, parsed_output=None)
    )
    with pytest.raises(PlanError) as e:
        planner.plan(IMG, "x")
    assert e.value.kind == "refusal"


def test_claude_planner_classifies_truncation():
    planner = ClaudePlanner(client=_fake_client(stop_reason="max_tokens", parsed_output=None))
    with pytest.raises(PlanError) as e:
        planner.plan(IMG, "x")
    assert e.value.kind == "bad_output"
