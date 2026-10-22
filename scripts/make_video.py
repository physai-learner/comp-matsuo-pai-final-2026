"""提出用の動画（docs/video-plan.md）を場面の素材からつなぐ。編集ソフトを使わない。

uv run python scripts/make_video.py                   # outputs/video.mp4（720p・約2分・無音）
uv run python scripts/make_video.py --repo-url <URL>  # 最後の画面にコードの URL を出す
uv run python scripts/make_video.py --preview         # 代表コマを outputs/video_preview/ へ

素材は scripts/record_scenes.py --planner claude で書き出したもの（outputs/scenes/claude/）と、
ROS 2 版の実行ログ（ros2/record_run.sh → outputs/ros2/）。
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "outputs" / "scenes" / "claude"

W, H = 1280, 720
FPS = 30
FADE = 0.3  # 場面の頭と終わりの暗転（秒）
# 左に正面カメラ（640×480 を 1.5倍）、右に指示と計画の欄
VIDEO_W = 960
PANEL_X = VIDEO_W + 24
PANEL_W = W - PANEL_X - 24

BG = (24, 26, 30)
FG = (236, 238, 242)
MUTED = (150, 156, 166)
ACCENT = (255, 196, 64)
OK = (110, 200, 130)
NG = (240, 110, 100)
BOX = (48, 52, 60)

FONT_CANDIDATES = [
    "C:/Windows/Fonts/BIZ-UDGothicR.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/BIZ-UDGothicB.ttc",
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

# (フォルダ, 見出し, 補足)
SCENE_TEXT = [
    ("01_basic", "1. 基本の指示", ""),
    ("02_multi", "2. まとめて頼む", "色で置き先を分ける指示を、3回の運搬に分解した"),
    ("03_spatial", "3. 位置関係", "どれが「一番左」かは、配置を見て決める"),
    ("04_refuse", "4. 無い物は運ばない", "計画を出さず、理由を返す"),
]
LIMIT = (
    "05_ball",
    "限界：ボールを",
    "真上からは円柱が円に見えるため、円柱を「ボール」として運んだ",
)


class Fonts:
    def __init__(self, regular: str | None):
        self.regular = regular or _find(FONT_CANDIDATES)
        self.bold = _find(BOLD_CANDIDATES) if regular is None else regular

    def __call__(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        path = self.bold if bold else self.regular
        # BIZ UD は2番目がプロポーショナル（BIZ UDPGothic）＝英字が等幅にならない
        return ImageFont.truetype(path, size, index=1 if "BIZ-UD" in path else 0)


def _find(paths: list[str]) -> str:
    for p in paths:
        if Path(p).exists():
            return p
    raise SystemExit("日本語フォントが見つからない。--font で .ttf/.ttc を指定する")


NO_LINE_START = set("」）、。・ー")


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    """1文字ずつ幅を測って折り返す（日本語は単語の区切りが無いため）。"""
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            # 閉じ括弧・句読点は行頭に送らず、はみ出しても前の行に付ける
            if line and ch not in NO_LINE_START and draw.textlength(line + ch, font=font) > width:
                lines.append(line)
                line = ch
            else:
                line += ch
        lines.append(line)
    return lines


def text_block(draw, xy, text, font, fill, width, gap=8) -> int:
    """折り返して書き、書き終えた下端の y を返す。"""
    x, y = xy
    size = font.size
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += size + gap
    return y


def fit(image: np.ndarray, w: int, h: int) -> Image.Image:
    return Image.fromarray(image).resize((w, h), Image.LANCZOS)


# ---- 場面ごとのコマ ------------------------------------------------------


def panel(canvas: Image.Image, f: Fonts, title: str, record: dict, note: str) -> None:
    """右の欄＝見出し・指示・Claude の返答。"""
    d = ImageDraw.Draw(canvas)
    y = 32
    y = text_block(d, (PANEL_X, y), title, f(28, bold=True), ACCENT, PANEL_W) + 18
    d.text((PANEL_X, y), "指示", font=f(18), fill=MUTED)
    y = text_block(d, (PANEL_X, y + 28), f"「{record['instruction']}」", f(26), FG, PANEL_W)
    y += 22
    d.text((PANEL_X, y), "Claude の返答", font=f(18), fill=MUTED)
    y += 30
    plan = record.get("plan") or {}
    moves = plan.get("moves", [])
    for i, m in enumerate(moves, 1):
        y = text_block(
            d, (PANEL_X, y), f"{i}. {m['item']} → {m['place_target']}", f(22), FG, PANEL_W
        )
    if plan.get("problem"):
        y = text_block(d, (PANEL_X, y), plan["problem"], f(22), NG, PANEL_W)
    if note:
        text_block(d, (PANEL_X, max(y + 24, H - 150)), note, f(20), MUTED, PANEL_W)


def scene_frames(folder: str, title: str, note: str, f: Fonts) -> Iterator[np.ndarray]:
    src = SCENES / folder
    record = json.loads((src / "plan.json").read_text(encoding="utf-8"))
    base = Image.new("RGB", (W, H), BG)
    panel(base, f, title, record, note)

    # 頭の3秒＝Claude に渡した真上画像に、計画（つかむ点→置く点）を重ねたもの
    head = base.copy()
    head.paste(fit(iio.imread(src / "plan.png"), VIDEO_W, H), (0, 0))
    d = ImageDraw.Draw(head)
    d.rectangle((0, H - 44, 420, H), fill=BG)
    d.text((16, H - 38), "真上カメラの画像と Claude の計画", font=f(22), fill=FG)
    yield from hold(np.asarray(head), 3.0)

    # 本編＝正面カメラ。断った場面は短いので、最後のコマで6秒まで延ばす
    frames = iio.imread(src / "video.mp4")
    n = len(frames)
    for i in range(max(n, int(6.0 * FPS))):
        c = base.copy()
        c.paste(fit(frames[min(i, n - 1)], VIDEO_W, H), (0, 0))
        yield np.asarray(c)


def hold(frame: np.ndarray, seconds: float) -> Iterator[np.ndarray]:
    for _ in range(int(seconds * FPS)):
        yield frame


def title_frames(f: Fonts) -> Iterator[np.ndarray]:
    first = iio.imread(SCENES / "01_basic" / "video.mp4", index=0)
    c = Image.fromarray(first).resize((W, int(W * 3 / 4)), Image.LANCZOS)
    c = c.crop((0, (c.height - H) // 2, W, (c.height - H) // 2 + H))
    c = Image.blend(c, Image.new("RGB", (W, H), BG), 0.55)
    d = ImageDraw.Draw(c)
    d.rectangle((0, 240, W, 430), fill=BG)
    title = "言葉で頼めるお片付けロボットアーム"
    ft = f(56, bold=True)
    d.text(((W - d.textlength(title, font=ft)) / 2, 270), title, font=ft, fill=FG)
    sub = "日本語の指示とカメラ画像から、どれを・どこへ運ぶかを Claude が決める"
    fs = f(26)
    d.text(((W - d.textlength(sub, font=fs)) / 2, 370), sub, font=fs, fill=MUTED)
    yield from hold(np.asarray(c), 6.0)


def flow_frames(f: Fonts) -> Iterator[np.ndarray]:
    """仕組みの図：指示＋真上画像 → Claude → 画素 → 机上の位置 → IK。"""
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((60, 44), "仕組み", font=f(36, bold=True), fill=ACCENT)
    thumb = fit(iio.imread(SCENES / "01_basic" / "plan.png"), 320, 240)
    c.paste(thumb, (60, 150))
    d.text((60, 400), "真上カメラの画像", font=f(20), fill=MUTED)
    d.text((60, 430), "＋ 日本語の指示", font=f(20), fill=MUTED)
    steps = [
        ("Claude", "どれを・どこへを\n画像の画素で答える"),
        ("座標の計算", "カメラの式で\n机の上の点へ戻す"),
        ("アーム", "IK でつかむ・運ぶ・\n離す"),
    ]
    x = 440
    for i, (head, body) in enumerate(steps):
        d.rounded_rectangle((x, 180, x + 230, 400), radius=16, fill=BOX)
        d.text((x + 20, 200), head, font=f(28, bold=True), fill=FG)
        text_block(d, (x + 20, 256), body, f(20), MUTED, 200)
        if i < len(steps) - 1:
            ax = x + 236
            d.polygon([(ax, 280), (ax + 20, 290), (ax, 300)], fill=ACCENT)
        x += 262
    d.polygon([(392, 280), (412, 290), (392, 300)], fill=ACCENT)
    note = "学習はしない。言葉の解釈は Claude、位置の計算と動作はプログラムが受け持つ"
    text_block(d, (60, 520), note, f(24), FG, W - 120)
    yield from hold(np.asarray(c), 9.0)


def table(d, x, y, rows, f: Fonts, col=(320, 140)) -> None:
    for i, (a, b, color) in enumerate(rows):
        font = f(26, bold=i == 0)
        d.text((x, y), a, font=font, fill=MUTED if i == 0 else FG)
        d.text((x + col[0], y), b, font=font, fill=MUTED if i == 0 else color)
        y += 46


def result_frames(f: Fonts) -> Iterator[np.ndarray]:
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((60, 44), "結果：28通りの指示 × 物の配置5通り", font=f(36, bold=True), fill=ACCENT)
    rows = [
        ("区分", "成功", FG),
        ("基本", "30/30", OK),
        ("言い換え", "30/30", OK),
        ("複数", "25/25", OK),
        ("位置関係", "25/25", OK),
        ("実行できない・曖昧", "25/30", NG),
        ("合計", "135/140", FG),
    ]
    table(d, 100, 140, rows, f)
    notes = (
        "無い物・曖昧な指示には、計画を出さずに理由を返すのを正解とした\n\n"
        "失敗5件はすべて「ボールを青い皿へ」\n\n"
        "1回の計画 平均3.2秒・約1.4円"
    )
    text_block(d, (640, 150), notes, f(24), FG, W - 700)
    yield from hold(np.asarray(c), 9.0)


def act_frames(f: Fonts) -> Iterator[np.ndarray]:
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((60, 44), "追加実験：模倣学習（ACT）との比較", font=f(36, bold=True), fill=ACCENT)
    d.text((60, 104), "「赤い箱を青い皿へ」・学習に使っていない配置20通り", font=f(24), fill=MUTED)
    rows = [
        ("方式", "成功", FG),
        ("IK（正解の位置を与える）", "20/20", OK),
        ("Claude ＋ IK（本方式）", "20/20", OK),
        ("ACT（手本200本で学習）", "13/20", NG),
    ]
    table(d, 100, 180, rows, f, col=(420, 140))
    notes = (
        "ACT の失敗7件はすべてつかみ損ね。運ぶ動きの流れは身に付けたが、"
        "縮めた画像では箱が小さく、つかむ位置がずれる\n\n"
        "本方式は学習なしで、指示を変えれば別の物・別の皿にも応じられる"
    )
    text_block(d, (100, 400), notes, f(24), FG, W - 200)
    yield from hold(np.asarray(c), 10.0)


def arrow(d, a, b, width: int = 3) -> None:
    """a から b への矢印（先端は b）。"""
    d.line((a, b), fill=ACCENT, width=width)
    v = np.subtract(b, a) / np.hypot(*np.subtract(b, a))
    n = np.array([-v[1], v[0]])
    tip = np.array(b)
    d.polygon([tuple(tip), tuple(tip - 16 * v + 8 * n), tuple(tip - 16 * v - 8 * n)], fill=ACCENT)


def ros2_log_lines() -> tuple[str, list[str]]:
    """ros2/record_run.sh の記録から、指示と tidy の表示（1件目の運搬だけ段階まで）を取る。"""
    src = ROOT / "outputs" / "ros2"
    launch = (src / "launch.log").read_text(encoding="utf-8", errors="replace")
    instruction = launch.split("plan: ", 1)[1].splitlines()[0]
    msgs = [
        line.split("[tidy]: ", 1)[1]
        for line in (src / "tidy.log").read_text(encoding="utf-8").splitlines()
        if "[tidy]: " in line
    ]
    out, moves = [], 0
    for m in msgs:
        if " -> " in m:
            moves += 1
            out.append(m.split("  (")[0])  # 机上の座標は画面では省く
        elif m.startswith("  "):
            if moves == 1:
                out.append("    " + m.strip())
        else:
            out.append(m)
    return instruction, out


def ros2_frames(f: Fonts) -> Iterator[np.ndarray]:
    """ROS 2 版：ノードのつながりと、実行ログ。"""
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((60, 44), "ROS 2 版：同じ処理をノードに分けた", font=f(36, bold=True), fill=ACCENT)
    sub = "講義と同じ ROS 2 Humble の環境で、指示1つから計画・運搬・確認までをつないだ"
    d.text((60, 104), sub, font=f(22), fill=MUTED)

    nodes = {
        "tidy": ((60, 330, 270, 440), "指示を受け、\n順に呼び出す"),
        "planner": ((420, 170, 680, 280), "Claude で計画"),
        "sim": ((420, 500, 680, 610), "MuJoCo の\nアームとカメラ"),
    }
    for name, (box, role) in nodes.items():
        d.rounded_rectangle(box, radius=14, fill=BOX)
        d.text((box[0] + 18, box[1] + 14), name, font=f(26, bold=True), fill=FG)
        text_block(d, (box[0] + 18, box[1] + 52), role, f(18), MUTED, box[2] - box[0] - 30, 4)
    arrow(d, (270, 365), (420, 240))
    d.text((250, 250), "action /plan", font=f(18), fill=FG)
    arrow(d, (270, 410), (420, 545))
    text_block(d, (60, 470), "action /pick_place\nservice /reset・/get_state", f(18), FG, 300, 4)
    arrow(d, (550, 500), (550, 280))
    d.multiline_text((325, 345), "topic\n/overhead/image_raw", font=f(18), fill=FG, spacing=4)

    x0, y0 = 730, 170
    instruction, lines = ros2_log_lines()
    d.rounded_rectangle((x0, y0, W - 40, y0 + 70 + 27 * len(lines)), radius=14, fill=BOX)
    y = text_block(d, (x0 + 20, y0 + 16), f"$ tidy「{instruction}」", f(20, bold=True), ACCENT, 470)
    for line in lines:
        y = text_block(d, (x0 + 20, y + 4), line, f(19), FG, 470, 4)
    note = "計画 10.4秒、運搬は1回 約15秒。実機に移すときは sim をロボットのドライバに置き換える"
    text_block(d, (60, 650), note, f(22), FG, W - 120)
    yield from hold(np.asarray(c), 10.0)


def end_frames(f: Fonts, repo_url: str | None) -> Iterator[np.ndarray]:
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    lines = [("言葉で頼めるお片付けロボットアーム", f(40, bold=True), FG)]
    if repo_url:
        lines.append((f"コード：{repo_url}", f(24), MUTED))
    lines.append(("MuJoCo・MuJoCo Menagerie（Franka Panda）・Claude", f(22), MUTED))
    y = 280
    for text, font, fill in lines:
        d.text(((W - d.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)
        y += font.size + 30
    yield from hold(np.asarray(c), 4.0)


# ---- つなぐ ----------------------------------------------------------------


def fade(frames: list[np.ndarray] | Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    frames = list(frames)
    k = int(FADE * FPS)
    for i, fr in enumerate(frames):
        a = min(1.0, (i + 1) / k, (len(frames) - i) / k)
        yield fr if a >= 1.0 else (fr * a).astype(np.uint8)


def segments(f: Fonts, repo_url: str | None) -> list[tuple[str, Iterator[np.ndarray]]]:
    segs = [("title", title_frames(f)), ("flow", flow_frames(f))]
    segs += [(folder, scene_frames(folder, t, n, f)) for folder, t, n in SCENE_TEXT]
    segs += [("result", result_frames(f)), ("act", act_frames(f)), ("ros2", ros2_frames(f))]
    segs += [(LIMIT[0], scene_frames(*LIMIT, f)), ("end", end_frames(f, repo_url))]
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "video.mp4")
    ap.add_argument("--repo-url", help="最後の画面に出すコードの URL")
    ap.add_argument("--font", help="日本語フォント（.ttf/.ttc）。既定は OS の標準から探す")
    ap.add_argument("--preview", action="store_true", help="各場面の代表コマを PNG で出すだけ")
    args = ap.parse_args()
    f = Fonts(args.font)

    if args.preview:
        out = ROOT / "outputs" / "video_preview"
        out.mkdir(parents=True, exist_ok=True)
        for name, gen in segments(f, args.repo_url):
            frames = list(gen)
            iio.imwrite(out / f"{name}_a.png", frames[0])
            iio.imwrite(out / f"{name}_b.png", frames[len(frames) * 2 // 3])
        print(f"代表コマ → {out}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", quality=8
    )
    writer.send(None)
    total = 0
    for name, gen in segments(f, args.repo_url):
        n = 0
        for fr in fade(gen):
            writer.send(np.ascontiguousarray(fr))
            n += 1
        total += n
        print(f"{name:12s} {n / FPS:5.1f}秒")
    writer.close()
    print(f"合計 {total / FPS:.1f}秒 → {args.out}")


if __name__ == "__main__":
    main()
