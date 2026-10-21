"""模倣学習（ACT）用：IK の動きをデモとして記録し、学習した方策でアームを動かす。

方策が見るもの＝真上画像（半分の 240×320）と関節状態（腕 7 関節＋指の開き）。
方策が出すもの＝0.1秒後の目標（腕 7 関節の角度＋指を閉じるか）。
言葉は受け取らない＝課題は1つに固定する。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tidy_arm.planner import GRASP_Z
from tidy_arm.sim import ARM_JOINTS, GRIP_CLOSED, GRIP_OPEN, Sim

TASK = ("red_box", "plate_blue")
CONTROL_DT = 0.1
# 0.002s × 50 = 0.1s
STEPS_PER_CONTROL = 50
# IK の動き（約10秒）の後に止まっている時間。方策に「終わったら止まる」を覚えさせる
HOLD_SECONDS = 1.0
# 方策に任せる長さ。IK の動き＋余裕
ROLLOUT_SECONDS = 14.0


def observe(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """真上画像を縦横半分（2×2 の平均）にしたものと、関節状態。"""
    image = sim.render("overhead")
    h, w, c = image.shape
    small = image.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3)).astype(np.uint8)
    q = sim.data.qpos
    state = np.concatenate([q[:ARM_JOINTS], [q[ARM_JOINTS] + q[ARM_JOINTS + 1]]])
    return small, state.astype(np.float32)


def command(sim: Sim) -> np.ndarray:
    """今の目標（腕の関節角＋指を閉じる度合い 0〜1）。"""
    ctrl = sim.data.ctrl
    closed = (GRIP_OPEN - ctrl[ARM_JOINTS]) / (GRIP_OPEN - GRIP_CLOSED)
    return np.concatenate([ctrl[:ARM_JOINTS], [closed]]).astype(np.float32)


@dataclass
class RecordingSim(Sim):
    """`recording` の間、0.1秒ごとに画像・関節状態・目標を記録する。"""

    recording: bool = field(init=False, default=False)
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        init=False, default_factory=list
    )

    def start_recording(self) -> None:
        self.recording = True
        self._since = 0
        self._sample()

    def _sample(self) -> None:
        self.samples.append((*observe(self), command(self)))

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            super().step()
            if self.recording:
                self._since += 1
                if self._since % STEPS_PER_CONTROL == 0:
                    self._sample()


def task_points(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """正解位置から把持点と置く点を作る（`planner.execute` と同じ高さ）。"""
    item, plate = TASK
    pick, place = sim.item_pos(item), sim.plate_pos(plate)
    pick[2] = place[2] = GRASP_Z
    return pick, place


def collect_episode(seed: int) -> dict | None:
    """1配置ぶんのデモ。IK が失敗した配置は None（学習に混ぜない）。

    観測 o_t には、次の 0.1 秒で向かう目標＝次の記録時点の目標 c_{t+1} を対にする。
    """
    sim = RecordingSim(seed=seed)
    try:
        sim.start_recording()
        sim.pick_and_place(*task_points(sim))
        sim.settle(HOLD_SECONDS)
        if not sim.on_plate(*TASK):
            return None
        images, states, commands = (np.stack(x) for x in zip(*sim.samples, strict=True))
        return {"images": images[:-1], "state": states[:-1], "action": commands[1:]}
    finally:
        sim.close()


def run_policy(
    sim: Sim,
    policy: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seconds: float = ROLLOUT_SECONDS,
) -> None:
    """0.1秒ごとに方策を呼び、腕は目標へ直線補間・指は 0.5 を境に開閉する。"""
    for _ in range(int(seconds / CONTROL_DT)):
        action = policy(*observe(sim))
        q0 = sim.data.ctrl[:ARM_JOINTS].copy()
        sim.data.ctrl[ARM_JOINTS] = GRIP_CLOSED if action[ARM_JOINTS] > 0.5 else GRIP_OPEN
        for i in range(1, STEPS_PER_CONTROL + 1):
            sim.data.ctrl[:ARM_JOINTS] = q0 + (action[:ARM_JOINTS] - q0) * i / STEPS_PER_CONTROL
            sim.step()
