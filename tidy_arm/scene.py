"""1つの指示を通しで動かし、動画の素材を1フォルダに書き出す。

overhead.png  計画係に渡した真上画像
plan.png      その画像に計画（つかむ点→置く点）を重ねたもの
plan.json     指示・計画・断った理由・判定・トークン
video.mp4     正面カメラの動画（断ったときも、アームが動かない様子を数秒残す）
"""

import json
from pathlib import Path

import imageio.v3 as iio

from tidy_arm.annotate import draw_plan
from tidy_arm.planner import ClaudePlanner, OraclePlanner, PlanError, execute
from tidy_arm.sim import Sim

FPS = 30
# 断ったとき・計画を得られなかったときに残す静止の長さ
IDLE_SECONDS = 3.0


def record_scene(
    instruction: str,
    out_dir: Path,
    *,
    planner: str = "oracle",
    seed: int = 0,
    expect: list[tuple[str, str]] | None = None,
    effort: str = "medium",
    video: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = Sim(seed=seed, record_camera="front" if video else None)
    try:
        image = sim.render("overhead")
        iio.imwrite(out_dir / "overhead.png", image)
        if planner == "oracle":
            p = OraclePlanner(sim, expect or [])
        else:
            p = ClaudePlanner(effort=effort)
        record = {"instruction": instruction, "planner": planner, "seed": seed}
        try:
            plan = p.plan(image, instruction)
        except PlanError as e:
            plan = None
            record["error"] = {"kind": e.kind, "message": str(e)}
        if plan is not None:
            record["plan"] = plan.model_dump()
            iio.imwrite(out_dir / "plan.png", draw_plan(image, plan))
        if plan is not None and plan.moves:
            execute(sim, plan)
        else:
            sim.settle(IDLE_SECONDS)
        if expect:
            record["ok"] = all(sim.on_plate(i, pl) for i, pl in expect)
        if isinstance(p, ClaudePlanner):
            record["usage"] = p.usage_log
        (out_dir / "plan.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if sim.frames:
            iio.imwrite(out_dir / "video.mp4", sim.frames, fps=FPS)
        return record
    finally:
        sim.close()
