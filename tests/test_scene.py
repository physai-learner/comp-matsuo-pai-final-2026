import json

import imageio.v3 as iio
import numpy as np

from tidy_arm.annotate import draw_plan
from tidy_arm.planner import Move, Pixel, Plan
from tidy_arm.scene import record_scene


def test_draw_plan_marks_pick_and_place():
    image = np.full((480, 640, 3), 128, np.uint8)
    plan = Plan(
        moves=[
            Move(item="a", pick=Pixel(u=100, v=100), place_target="b", place=Pixel(u=300, v=200))
        ],
        problem="",
    )
    out = draw_plan(image, plan)
    assert out.shape == image.shape
    assert (out[100, 100 - 18] != 128).any()  # つかむ点の輪
    assert (out[100, 100] == 128).all()  # 輪の内側（物）は隠さない
    assert (out[200, 300 - 9] != 128).any()  # 置く点の輪
    assert (out[400, 600] == 128).all()  # 関係ない所は触らない


def test_record_scene_writes_materials(tmp_path):
    record = record_scene(
        "赤い箱を青い皿に置いて", tmp_path, seed=1, expect=[("red_box", "plate_blue")], video=False
    )
    assert record["ok"]
    assert {p.name for p in tmp_path.iterdir()} == {"overhead.png", "plan.png", "plan.json"}
    saved = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert saved["instruction"] == "赤い箱を青い皿に置いて"
    assert iio.imread(tmp_path / "plan.png").shape == iio.imread(tmp_path / "overhead.png").shape
