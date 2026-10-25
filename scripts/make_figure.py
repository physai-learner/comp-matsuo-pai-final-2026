"""README とレポートの冒頭に載せる図（計画と実行後を横に並べた1枚）を書き出す。

uv run python scripts/make_figure.py   # 先に record_scenes.py --planner claude を済ませておく
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "outputs" / "scenes" / "claude" / "02_multi"
OUT = ROOT / "docs" / "images" / "overview.png"
GAP = 8  # 2枚の間の白い隙間（画素）


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    plan = Image.open(args.scene / "plan.png").convert("RGB")
    frames = iio.imread(args.scene / "video.mp4")
    after = Image.fromarray(np.asarray(frames[-1])).convert("RGB")
    # 実行後の正面映像を計画画像の高さにそろえる
    after = after.resize((round(after.width * plan.height / after.height), plan.height))

    canvas = Image.new("RGB", (plan.width + GAP + after.width, plan.height), "white")
    canvas.paste(plan, (0, 0))
    canvas.paste(after, (plan.width + GAP, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(f"{args.out} ({canvas.width}×{canvas.height}, {args.out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
