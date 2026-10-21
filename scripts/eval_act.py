"""学習した ACT で「赤い箱を青い皿へ」を配置を変えて試し、成功率を出す。

1試行1行の CSV（outputs/eval_act.csv）。`--policy scripted` は同じ配置で IK の動き（正解位置）を
流す＝上限の目安。Claude 方式との比較は
`evaluate.py --planner claude --tasks basic_red_box_plate_blue` を同じ配置で回す。
"""

import argparse
import csv
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from tidy_arm.demos import TASK, run_policy, task_points
from tidy_arm.sim import Sim

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["act", "scripted"], default="act")
    ap.add_argument("--ckpt", type=Path, help="act.py の final.pt / last.pt")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--video", type=int, default=1, help="先頭から何試行を正面カメラの MP4 に残すか"
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    out = args.out or OUT / f"eval_{args.policy}.csv"

    policy = None
    if args.policy == "act":
        from tidy_arm.act import ActPolicy  # torch は学習用の依存グループにだけある

        policy = ActPolicy(args.ckpt)

    ok = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "ok", "dist_cm", "lifted", "sec"])
        for i, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
            sim = Sim(seed=seed, record_camera="front" if i < args.video else None)
            try:
                t0 = time.time()
                start = sim.item_pos(TASK[0])
                if policy is None:
                    sim.pick_and_place(*task_points(sim))
                    sim.settle(0.5)
                else:
                    policy.reset()
                    run_policy(sim, policy)
                success = sim.on_plate(*TASK)
                end = sim.item_pos(TASK[0])
                dist = np.linalg.norm(end[:2] - sim.plate_pos(TASK[1])[:2]) * 100
                # 動かせたか（失敗の分類用）
                lifted = bool(np.linalg.norm(end[:2] - start[:2]) > 0.03)
                ok += success
                w.writerow(
                    [seed, int(success), f"{dist:.1f}", int(lifted), f"{time.time() - t0:.1f}"]
                )
                f.flush()
                print(f"seed {seed}: {'OK' if success else 'NG'}  皿まで {dist:.1f}cm", flush=True)
                if sim.frames:
                    iio.imwrite(OUT / f"{out.stem}_seed{seed}.mp4", sim.frames, fps=30)
            finally:
                sim.close()
    print(f"success {ok}/{args.seeds} → {out}")


if __name__ == "__main__":
    main()
