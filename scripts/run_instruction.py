"""1つの指示を通しで動かす：真上画像 → 計画（画素）→ 把持・運搬 → 判定。

uv run python scripts/run_instruction.py "赤い箱を青い皿に置いて" --expect red_box:plate_blue
uv run python scripts/run_instruction.py "..." --expect ... --planner claude   # .env のキーを使う
"""

import argparse
from pathlib import Path

import imageio.v3 as iio

from tidy_arm.planner import ClaudePlanner, OraclePlanner, PlanError, execute
from tidy_arm.sim import Sim

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction")
    ap.add_argument(
        "--expect", nargs="+", required=True, metavar="ITEM:PLATE", help="正解の運び先（判定用）"
    )
    ap.add_argument("--planner", choices=["oracle", "claude"], default="oracle")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    expect = [tuple(s.split(":")) for s in args.expect]
    OUT.mkdir(exist_ok=True)

    sim = Sim(seed=args.seed, record_camera="front" if args.video else None)
    try:
        image = sim.render("overhead")
        if args.planner == "oracle":
            planner = OraclePlanner(sim, expect)
        else:
            planner = ClaudePlanner(effort=args.effort)
        try:
            plan = planner.plan(image, args.instruction)
        except PlanError as e:
            print(f"計画できず: {e}")
            return
        print(plan.model_dump_json(indent=2))
        if isinstance(planner, ClaudePlanner):
            print(planner.usage_log)
        execute(sim, plan)
        ok = all(sim.on_plate(item, plate) for item, plate in expect)
        print(
            "OK" if ok else "NG", {item: sim.item_pos(item).round(3).tolist() for item, _ in expect}
        )
        if sim.frames:
            iio.imwrite(OUT / f"{args.planner}_seed{args.seed}.mp4", sim.frames, fps=30)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
