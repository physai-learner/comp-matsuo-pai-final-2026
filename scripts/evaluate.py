"""指示ごと・配置（seed）ごとに通しで動かし、成功率と失敗の分類を CSV に出す。

uv run python scripts/evaluate.py                       # 正解計画（配線と上限の確認）
uv run python scripts/evaluate.py --planner claude      # .env のキーを使う・1行1回の呼び出し

結果（1行＝1試行）:
  ok            正解の物が正解の皿に載った（実行できない指示では、計画を出さず理由を返した）。
                位置関係が僅差（2cm 未満）の配置では、どちらを選んでも・保留しても正解
  plan:<kind>   計画を得られなかった（refusal / bad_output / out_of_image）
  no_plan       実行できる指示なのに計画を出さなかった
  should_refuse 実行できない指示なのに計画を出した
  wrong_count   運ぶ件数が違う
  wrong_item    つかむ画素が正解の物に当たっていない（取り違え・位置ずれ）
  wrong_place   置く画素が正解の皿に当たっていない
  exec_fail     計画は正しいが、運び終えたあと皿に載っていない（把持・運搬の失敗）
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tidy_arm.planner import (
    ITEM_TOP_Z,
    PLATE_TOP_Z,
    ClaudePlanner,
    OraclePlanner,
    Plan,
    PlanError,
    execute,
)
from tidy_arm.sim import ITEMS, PLATE_RADIUS, PLATES, Sim
from tidy_arm.tasks import TASKS, Task

OUT = Path(__file__).resolve().parent.parent / "outputs"
# つかむ画素を机上へ戻した点が、物の中心からこれ以内なら「その物を指した」とみなす
PICK_TOL = 0.035


def ground(sim: Sim, plan: Plan) -> tuple[list[tuple[str | None, str | None]], list[float]]:
    """計画の各画素が、実際にはどの物・どの皿を指しているか。つかむ点の誤差（m）も返す。"""
    cam = sim.camera()
    pairs, errors = [], []
    for m in plan.moves:
        pick = cam.pixel_to_world([m.pick.u, m.pick.v], z=ITEM_TOP_Z)[:2]
        place = cam.pixel_to_world([m.place.u, m.place.v], z=PLATE_TOP_Z)[:2]
        d_item = {n: float(np.linalg.norm(pick - sim.item_pos(n)[:2])) for n in ITEMS}
        d_plate = {n: float(np.linalg.norm(place - sim.plate_pos(n)[:2])) for n in PLATES}
        item = min(d_item, key=d_item.get)
        plate = min(d_plate, key=d_plate.get)
        errors.append(d_item[item])
        pairs.append(
            (
                item if d_item[item] < PICK_TOL else None,
                plate if d_plate[plate] < PLATE_RADIUS else None,
            )
        )
    return pairs, errors


def judge(sim: Sim, plan: Plan, answers: list, problem_ok: bool) -> str:
    """answers＝正解になる計画の候補（僅差・解釈の割れで複数あり得る）。"""
    if not plan.moves:
        return "ok" if problem_ok else "no_plan"
    if not answers:
        return "should_refuse"
    same_count = [a for a in answers if len(a) == len(plan.moves)]
    if not same_count:
        return "wrong_count"
    pairs, _ = ground(sim, plan)
    items = Counter(i for i, _ in pairs)
    same_items = [a for a in same_count if Counter(i for i, _ in a) == items]
    if not same_items:
        return "wrong_item"
    match = next((a for a in same_items if Counter(a) == Counter(pairs)), None)
    if match is None:
        return "wrong_place"
    execute(sim, plan)
    return "ok" if all(sim.on_plate(i, p) for i, p in match) else "exec_fail"


def run_one(task: Task, seed: int, planner_name: str, claude: ClaudePlanner | None) -> dict:
    sim = Sim(seed=seed)
    try:
        expected = task.expected(sim)
        answers, problem_ok = task.accepted(sim)
        image = sim.render("overhead")
        if planner_name == "oracle":
            planner = OraclePlanner(sim, expected, problem="実行できない" * task.expect_problem)
        else:
            planner = claude
        n_usage = len(claude.usage_log) if claude else 0
        t0 = time.perf_counter()
        try:
            plan = planner.plan(image, task.instruction)
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
            "seed": seed,
            "instruction": task.instruction,
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
                row = run_one(task, seed, args.planner, claude)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()  # 途中で止めても、そこまでの結果が残る
                print(f"{task.name:20s} seed={seed} {row['result']}")

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["result"] == "ok")
    print(f"\n成功 {sum(r['result'] == 'ok' for r in rows)}/{len(rows)}")
    for cat, oks in by_cat.items():
        print(f"  {cat:12s} {sum(oks)}/{len(oks)}")
    fails = Counter(r["result"] for r in rows if r["result"] != "ok")
    if fails:
        print("失敗の内訳:", dict(fails))
    if claude:
        tin = sum(r["input_tokens"] for r in rows)
        tout = sum(r["output_tokens"] for r in rows)
        print(f"トークン 入力 {tin:,}・出力 {tout:,}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
