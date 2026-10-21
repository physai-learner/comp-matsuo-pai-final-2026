"""動画の場面（docs/video-plan.md）の素材をまとめて書き出す。

uv run python scripts/record_scenes.py                    # 正解計画で下見（API を使わない）
uv run python scripts/record_scenes.py --planner claude   # 本番（1場面1回の呼び出し）
uv run python scripts/record_scenes.py --planner claude --redraw   # plan.png だけ描き直す
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio

from tidy_arm.annotate import draw_plan
from tidy_arm.planner import Plan
from tidy_arm.scene import record_scene
from tidy_arm.sim import Sim
from tidy_arm.tasks import TASKS

OUT = Path(__file__).resolve().parent.parent / "outputs" / "scenes"

# (フォルダ名, 指示の名前 tidy_arm/tasks.py, 配置)
SCENES = [
    ("01_basic", "basic_red_box_plate_blue", 0),
    ("02_multi", "multi_all", 0),
    ("03_spatial", "spatial_left", 3),  # 一番左＝紫の箱（01 と違う物・僅差なし）
    ("04_refuse", "infeasible_item", 0),
    ("05_ball", "infeasible_ball", 0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["oracle", "claude"], default="oracle")
    ap.add_argument("--only", nargs="*", help="フォルダ名の前方一致で絞る（例: 01 04）")
    ap.add_argument("--redraw", action="store_true", help="API もシミュレータも動かさない")
    args = ap.parse_args()

    tasks = {t.name: t for t in TASKS}
    for folder, name, seed in SCENES:
        if args.only and not any(folder.startswith(p) for p in args.only):
            continue
        task = tasks[name]
        out_dir = OUT / args.planner / folder
        if args.redraw:
            saved = json.loads((out_dir / "plan.json").read_text(encoding="utf-8"))
            if "plan" in saved:
                image = iio.imread(out_dir / "overhead.png")
                plan = Plan.model_validate(saved["plan"])
                iio.imwrite(out_dir / "plan.png", draw_plan(image, plan))
            print(f"{folder:12s} 描き直した")
            continue
        sim = Sim(seed=seed)
        try:
            expect = task.expected(sim)
        finally:
            sim.close()
        record = record_scene(
            task.instruction,
            out_dir,
            planner=args.planner,
            seed=seed,
            expect=expect,
        )
        plan = record.get("plan") or {}
        moves = [(m["item"], m["place_target"]) for m in plan.get("moves", [])]
        print(f"{folder:12s} {task.instruction}")
        print(f"{'':12s} 計画 {moves} 判定 {record.get('ok')} {plan.get('problem', '')}")


if __name__ == "__main__":
    main()
