"""ROS 2 版の評価 CSV から、日本語と英語の指示を同じ指示・同じ配置の対で比べる。

    uv run python scripts/compare_lang.py [outputs/eval_ros2.csv]

区分ごとの成功数、計画時間・出力トークン・つかむ点の誤差の平均、結果が食い違った対の一覧を表示する。
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tidy_arm.judge import summarize

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_ros2.csv")
with path.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
summarize(rows)


def nums(rs, key):
    return [float(r[key]) for r in rs if r[key] not in ("", None)]


print("\n区分         ja      en")
for cat in dict.fromkeys(r["category"] for r in rows):
    cells = []
    for lang in ("ja", "en"):
        rs = [r for r in rows if r["category"] == cat and r["lang"] == lang]
        cells.append(f"{sum(r['result'] == 'ok' for r in rs)}/{len(rs)}")
    print(f"{cat:12s} {cells[0]:7s} {cells[1]}")

print("\n         計画秒  出力トークン  誤差cm(平均/最大)")
for lang in ("ja", "en"):
    rs = [r for r in rows if r["lang"] == lang]
    err = nums(rs, "pick_err_cm")
    print(
        f"{lang}       {mean(nums(rs, 'plan_sec')):5.1f}  {mean(nums(rs, 'output_tokens')):10.0f}"
        f"  {mean(err):.2f}/{max(err):.1f}"
    )

pairs = defaultdict(dict)
for r in rows:
    pairs[(r["task"], r["seed"])][r["lang"]] = r
diff = [p for p in pairs.values() if len(p) == 2 and p["ja"]["result"] != p["en"]["result"]]
print(f"\n結果が食い違った対: {len(diff)}/{sum(len(p) == 2 for p in pairs.values())}")
for p in diff:
    ja, en = p["ja"], p["en"]
    print(f"- {ja['task']} seed={ja['seed']}: ja={ja['result']} en={en['result']}")
    for r in (ja, en):
        print(f"    [{r['lang']}] {r['instruction']} → {r['plan_moves']} {r['problem'][:80]}")
