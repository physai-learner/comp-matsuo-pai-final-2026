"""指示ごと・配置（seed）ごとに通しで動かし、成功率と失敗の分類を CSV に出す（ROS なし版）。

uv run python scripts/evaluate.py                       # 正解計画（配線と上限の確認）
uv run python scripts/evaluate.py --planner claude      # .env のキーを使う・1行1回の呼び出し
uv run python scripts/evaluate.py --lang en             # 英語の指示で

結果の分類は tidy_arm/judge.py の冒頭。ROS 2 版は ros2/src/tidy_arm_ros の evaluate。
"""

import argparse
import csv
import time
from pathlib import Path

from tidy_arm.judge import check_plan, summarize
from tidy_arm.judge import ground as _ground
from tidy_arm.planner import ClaudePlanner, OraclePlanner, Plan, PlanError, execute
from tidy_arm.sim import Sim
from tidy_arm.tasks import LANGS, TASKS, Task

OUT = Path(__file__).resolve().parent.parent / "outputs"


def ground(sim: Sim, plan: Plan):
    return _ground(sim, sim.camera(), plan)


def judge(sim: Sim, plan: Plan, answers: list, problem_ok: bool) -> str:
    result, match = check_plan(sim, sim.camera(), plan, answers, problem_ok)
    if result is not None:
        return result
    execute(sim, plan)
    return "ok" if all(sim.on_plate(i, p) for i, p in match) else "exec_fail"


def run_one(
    task: Task, seed: int, planner_name: str, claude: ClaudePlanner | None, lang: str = "ja"
) -> dict:
    sim = Sim(seed=seed)
    try:
        instruction = task.text(lang)
        answers, problem_ok = task.accepted(sim, lang)
        image = sim.render("overhead")
        if planner_name == "oracle":
            planner = OraclePlanner(
                sim, task.expected(sim), problem="実行できない" * task.expect_problem
            )
        else:
            planner = claude
        n_usage = len(claude.usage_log) if claude else 0
        t0 = time.perf_counter()
        try:
            plan = planner.plan(image, instruction)
            plan_sec = time.perf_counter() - t0
            _, errors = ground(sim, plan)
            result = judge(sim, plan, answers, problem_ok)
            moves = [(m.item, m.place_target) for m in plan.moves]
            problem = plan.problem
        except PlanError as e:
            plan_sec, errors, moves, problem = time.perf_counter() - t0, [], [], str(e)
            result = f"plan:{e.kind}"
        usage = claude.usage_log[n_usage:] if claude else []
        return {
            "task": task.name,
            "category": task.category,
            "lang": lang,
            "seed": seed,
            "instruction": instruction,
            "expected": answers,
            "problem_ok": problem_ok,
            "result": result,
            "plan_moves": moves,
            "problem": problem,
            "pick_err_cm": round(100 * max(errors), 1) if errors else "",
            "plan_sec": round(plan_sec, 1),
            "input_tokens": sum(u["input_tokens"] for u in usage),
            "output_tokens": sum(u["output_tokens"] for u in usage),
        }
    finally:
        sim.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["oracle", "claude"], default="oracle")
    ap.add_argument("--seeds", type=int, default=3, help="配置の乱数 S..S+N-1")
    ap.add_argument("--seed-start", type=int, default=0, help="S（既に回した配置を足すとき）")
    ap.add_argument("--tasks", nargs="*", help="名前の前方一致で絞る（例: basic spatial_left）")
    ap.add_argument("--lang", nargs="+", choices=LANGS, default=["ja"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    tasks = [t for t in TASKS if not args.tasks or any(t.name.startswith(p) for p in args.tasks)]
    claude = ClaudePlanner(effort=args.effort) if args.planner == "claude" else None
    out = args.out or OUT / f"eval_{args.planner}.csv"
    out.parent.mkdir(exist_ok=True)

    rows = []
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = None
        for task in tasks:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for lang in args.lang:
                    row = run_one(task, seed, args.planner, claude, lang)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    f.flush()  # 途中で止めても、そこまでの結果が残る
                    print(f"{task.name:22s} seed={seed} {lang} {row['result']}")

    summarize(rows)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
