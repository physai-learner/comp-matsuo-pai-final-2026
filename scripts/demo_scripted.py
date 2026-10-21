"""AI なしの動作確認：正解の座標を直接渡して、物を皿へ運ぶ。成功率と動画を出す。"""

import argparse
from pathlib import Path

import imageio.v3 as iio

from tidy_arm.sim import ITEMS, PLATES, Sim

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="1回目の試行を MP4 に書き出す")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    ok = 0
    for t in range(args.trials):
        sim = Sim(seed=t, record_camera="front" if args.video and t == 0 else None)
        if t == 0:
            iio.imwrite(OUT / "overhead_seed0.png", sim.render("overhead"))
        item, plate = ITEMS[t % len(ITEMS)], PLATES[t % len(PLATES)]
        pick = sim.item_pos(item)
        place = sim.plate_pos(plate)
        place[2] = pick[2]
        sim.pick_and_place(pick, place)
        sim.settle(0.5)
        success = sim.on_plate(item, plate)
        ok += success
        mark = "OK" if success else "NG"
        print(f"trial {t}: {item} -> {plate}: {mark}  final={sim.item_pos(item).round(3)}")
        if sim.frames:
            iio.imwrite(OUT / "scripted_seed0.mp4", sim.frames, fps=30)
            iio.imwrite(OUT / "front_seed0_end.png", sim.frames[-1])
        sim.close()
    print(f"success {ok}/{args.trials}")


if __name__ == "__main__":
    main()
