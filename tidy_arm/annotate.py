"""計画を真上画像に重ねて描く（動画・レポート用）。

つかむ点＝物を囲む輪、置く点＝十字の付いた輪、その間を矢印でつなぎ、実行順の番号を添える。
文字は数字だけにする（日本語の字幕は動画編集で入れる）。
"""

import numpy as np
from PIL import Image, ImageDraw

from tidy_arm.planner import Plan

# 1件目・2件目…の色。机（灰）・皿（青・黄）・物（赤・紫・緑）のどれとも紛れない色にする
COLORS = [(255, 255, 255), (0, 230, 230), (255, 140, 0)]
OUTLINE = (0, 0, 0)
R = 9


def _arrow(draw: ImageDraw.ImageDraw, a, b, color) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.linalg.norm(d)
    if n < 2 * R:
        return
    d /= n
    start, end = a + d * (R + 9), b - d * (R + 4)
    draw.line([*start, *end], fill=OUTLINE, width=6)
    draw.line([*start, *end], fill=color, width=3)
    side = np.array([-d[1], d[0]])
    head = [tuple(end + d * 4), tuple(end - d * 10 + side * 7), tuple(end - d * 10 - side * 7)]
    draw.polygon(head, fill=color, outline=OUTLINE)


def draw_plan(image: np.ndarray, plan: Plan) -> np.ndarray:
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for i, m in enumerate(plan.moves):
        color = COLORS[i % len(COLORS)]
        pick, place = (m.pick.u, m.pick.v), (m.place.u, m.place.v)
        _arrow(draw, pick, place, color)
        # つかむ点は輪だけにして物を隠さない。番号は左上へずらした小さな丸に書く
        x, y = pick
        r = R + 9  # 箱の対角（約20画素）を囲む
        draw.ellipse([x - r, y - r, x + r, y + r], outline=OUTLINE, width=6)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        bx, by = x - r - 6, y - r - 6
        draw.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=color, outline=OUTLINE, width=2)
        draw.text((bx, by), str(i + 1), fill=OUTLINE, anchor="mm")
        x, y = place
        draw.ellipse([x - R, y - R, x + R, y + R], outline=OUTLINE, width=5)
        draw.ellipse([x - R, y - R, x + R, y + R], outline=color, width=3)
        draw.line([x - R, y, x + R, y], fill=color, width=1)
        draw.line([x, y - R, x, y + R], fill=color, width=1)
    return np.asarray(img)
