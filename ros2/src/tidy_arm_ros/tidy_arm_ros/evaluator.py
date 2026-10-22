"""評価を ROS 2 の action と service だけで回す（sim と planner を launch した状態で使う）。

    ros2 run tidy_arm_ros evaluate --seeds 5 --lang ja en [--tasks basic] [--resume]

1試行＝reset(seed) → get_state（物と皿の位置＝正解を求める）→ plan → 画素の採点
→ 正しい計画なら pick_place を順に → get_state（皿に載ったか）。
採点の規則は ROS なし版（scripts/evaluate.py）と同じ tidy_arm/judge.py を使う。
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from tidy_arm_interfaces.action import PickPlace
from tidy_arm_interfaces.action import Plan as PlanAction
from tidy_arm_interfaces.srv import GetState, Reset

from tidy_arm.judge import SceneState, check_plan, ground, summarize
from tidy_arm.planner import Move, Pixel, Plan
from tidy_arm.tasks import LANGS, TASKS, Task
from tidy_arm_ros.commander import Commander
from tidy_arm_ros.planner_node import overhead_camera

OUT = Path("/workspace/outputs")
FIELDS = [
    "task",
    "category",
    "lang",
    "seed",
    "instruction",
    "expected",
    "problem_ok",
    "result",
    "plan_moves",
    "problem",
    "pick_err_cm",
    "plan_sec",
    "input_tokens",
    "output_tokens",
    "exec_sec",
]


def xyz(p) -> np.ndarray:
    return np.array([p.x, p.y, p.z])


def scene_of(state) -> SceneState:
    return SceneState(
        items={n: xyz(p) for n, p in zip(state.items, state.item_positions, strict=True)},
        plates={n: xyz(p) for n, p in zip(state.plate_names, state.plate_positions, strict=True)},
    )


def to_plan(result) -> Plan:
    """action の結果を、採点に使う形（tidy_arm.planner.Plan）へ戻す。"""
    return Plan(
        moves=[
            Move(
                item=m.item,
                pick=Pixel(u=m.pick_u, v=m.pick_v),
                place_target=m.place_target,
                place=Pixel(u=m.place_u, v=m.place_v),
            )
            for m in result.moves
        ],
        problem=result.problem,
    )


class Evaluator(Commander):
    def __init__(self):
        super().__init__()
        self.camera = overhead_camera()

    def run_one(self, task: Task, seed: int, lang: str) -> dict:
        self.call(self.reset_client, Reset.Request(seed=seed))
        scene = scene_of(self.call(self.state_client, GetState.Request()))
        instruction = task.text(lang)
        answers, problem_ok = task.accepted(scene, lang)
        t0 = time.perf_counter()
        res = self.run_action(self.plan_client, PlanAction.Goal(instruction=instruction))
        plan_sec = time.perf_counter() - t0
        plan = to_plan(res)
        errors, exec_sec = [], 0.0
        if res.error:
            result = f"plan:{res.error}"
        else:
            _, errors = ground(scene, self.camera, plan)
            result, match = check_plan(scene, self.camera, plan, answers, problem_ok)
            if result is None:
                t1 = time.perf_counter()
                for m in res.moves:
                    self.run_action(
                        self.pick_place_client, PickPlace.Goal(pick=m.pick, place=m.place)
                    )
                exec_sec = time.perf_counter() - t1
                after = self.call(self.state_client, GetState.Request())
                on = dict(zip(after.items, after.plates, strict=True))
                result = "ok" if all(on[i] == p for i, p in match) else "exec_fail"
        return {
            "task": task.name,
            "category": task.category,
            "lang": lang,
            "seed": seed,
            "instruction": instruction,
            "expected": answers,
            "problem_ok": problem_ok,
            "result": result,
            "plan_moves": [(m.item, m.place_target) for m in plan.moves],
            "problem": res.problem,
            "pick_err_cm": round(100 * max(errors), 1) if errors else "",
            "plan_sec": round(plan_sec, 1),
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "exec_sec": round(exec_sec, 1),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="配置の乱数 S..S+N-1")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", help="名前の前方一致で絞る")
    ap.add_argument("--lang", nargs="+", choices=LANGS, default=list(LANGS))
    ap.add_argument("--out", type=Path, default=OUT / "eval_ros2.csv")
    ap.add_argument("--resume", action="store_true", help="CSV に既にある試行を飛ばして追記する")
    args, ros_args = ap.parse_known_args()

    tasks = [t for t in TASKS if not args.tasks or any(t.name.startswith(p) for p in args.tasks)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: list[dict] = []
    if args.resume and args.out.exists():
        with args.out.open(encoding="utf-8-sig") as f:
            done = list(csv.DictReader(f))
    seen = {(r["task"], int(r["seed"]), r["lang"]) for r in done}

    rclpy.init(args=ros_args)
    node = Evaluator()
    rows = list(done)
    try:
        mode = "a" if done else "w"
        with args.out.open(mode, newline="", encoding="utf-8" if done else "utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if not done:
                writer.writeheader()
            # 言語は内側で回す＝同じ配置で日本語と英語を続けて比べる（途中で止めても対が揃う）
            for task in tasks:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    for lang in args.lang:
                        if (task.name, seed, lang) in seen:
                            continue
                        row = node.run_one(task, seed, lang)
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                        print(
                            f"{task.name:22s} seed={seed} {lang} {row['result']}"
                            f"  plan {row['plan_sec']}s",
                            flush=True,
                        )
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    summarize(rows)
    print(f"→ {args.out}")
