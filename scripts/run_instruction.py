"""1つの指示を通しで動かす：真上画像 → 計画（画素）→ 把持・運搬 → 判定。

uv run python scripts/run_instruction.py "赤い箱を青い皿に置いて" --expect red_box:plate_blue
uv run python scripts/run_instruction.py "..." --planner claude --video   # .env のキーを使う

素材は outputs/scenes/<--name>/ に出る（中身は tidy_arm/scene.py を参照）。
"""

import argparse
import json
from pathlib import Path

from tidy_arm.scene import record_scene

OUT = Path(__file__).resolve().parent.parent / "outputs" / "scenes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction")
    ap.add_argument(
        "--expect", nargs="*", default=[], metavar="ITEM:PLATE", help="正解の運び先（判定用）"
    )
    ap.add_argument("--planner", choices=["oracle", "claude"], default="oracle")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--name", help="書き出し先のフォルダ名（既定 <planner>_seed<seed>）")
    args = ap.parse_args()
    expect = [tuple(s.split(":")) for s in args.expect]
    if args.planner == "oracle" and not expect:
        ap.error("正解計画（oracle）には --expect が要る")

    out_dir = OUT / (args.name or f"{args.planner}_seed{args.seed}")
    record = record_scene(
        args.instruction,
        out_dir,
        planner=args.planner,
        seed=args.seed,
        expect=expect,
        effort=args.effort,
        video=args.video,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
