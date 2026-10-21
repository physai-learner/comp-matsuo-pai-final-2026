"""ACT の学習用デモを集める：正解位置＋IK で「赤い箱を青い皿へ」を配置を変えて繰り返す。

評価（配置 0〜）と重ならないよう、既定では配置 1000 から使う。IK が失敗した配置は捨てる。
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

from tidy_arm.act import save_demos
from tidy_arm.demos import collect_episode

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes", type=int, default=100, help="集める本数（成功したものだけ数える）"
    )
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT / "demos.npz")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    episodes, failed = [], []
    seed = args.seed_start
    with Pool(args.jobs) as pool:
        while len(episodes) < args.episodes:
            seeds = list(range(seed, seed + args.episodes - len(episodes)))
            seed += len(seeds)
            for s, ep in zip(seeds, pool.map(collect_episode, seeds), strict=True):
                if ep is None:
                    failed.append(s)
                else:
                    episodes.append(ep)
                    print(f"seed {s}: {len(ep['state'])} 歩", flush=True)
    save_demos(args.out, episodes)
    frames = sum(len(e["state"]) for e in episodes)
    mb = args.out.stat().st_size / 1e6
    print(f"{len(episodes)}本・{frames}枚・{mb:.0f}MB・失敗 {failed}・{time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
