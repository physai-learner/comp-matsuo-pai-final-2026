"""MuJoCo の机上シーンと、IK による把持・運搬。AI は使わず、座標を受け取って動くところまで。"""

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from tidy_arm.camera import Camera

SCENE = Path(__file__).resolve().parent.parent / "assets" / "franka_emika_panda" / "tabletop.xml"

ITEMS = ("red_box", "green_cylinder", "purple_box")
PLATES = ("plate_blue", "plate_yellow")
ARM_JOINTS = 7
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
# hand ボディ原点から指先の間（把持点）までの距離
TCP_OFFSET = 0.1034
# 物を置ける範囲（ロボット正面の机上）
WORKSPACE_X = (0.35, 0.65)
WORKSPACE_Y = (-0.3, 0.3)
PLATE_RADIUS = 0.07
# 待機姿勢の把持点。真上カメラに作業範囲が映るよう、ロボットの根元側へ引いておく
PARK_POS = np.array([0.2, 0.0, 0.55])
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480


@dataclass
class Sim:
    seed: int | None = None
    model: mujoco.MjModel = field(init=False)
    data: mujoco.MjData = field(init=False)
    frames: list[np.ndarray] = field(init=False, default_factory=list)
    record_camera: str | None = None
    record_every: int = 17  # 0.002s × 17 ≒ 30fps

    def __post_init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        # panda.xml の根元真上の光は、アームの影を机の手前へ放射状に落とす＝影だけ切る
        self.model.light_castshadow[self.model.light("top").id] = 0
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.seed)
        self._hand = self.model.body("hand").id
        self._ik_data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self.reset()

    # ---- シーン -------------------------------------------------------------

    def reset(self, randomize: bool = True) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        if randomize:
            self._place_randomly()
        mujoco.mj_forward(self.model, self.data)
        self._down_rot = self.data.xmat[self._hand].reshape(3, 3).copy()
        park = self.ik(PARK_POS)
        self.data.qpos[:ARM_JOINTS] = park
        self.data.ctrl[:ARM_JOINTS] = park
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.3)

    def _place_randomly(self) -> None:
        """皿と物を、互いに重ならないよう作業範囲に置く。"""
        placed: list[tuple[np.ndarray, float]] = []

        def sample(radius: float) -> np.ndarray:
            for _ in range(200):
                p = np.array([self.rng.uniform(*WORKSPACE_X), self.rng.uniform(*WORKSPACE_Y)])
                if all(np.linalg.norm(p - q) > radius + r + 0.02 for q, r in placed):
                    placed.append((p, radius))
                    return p
            raise RuntimeError("配置が見つからない")

        for name in PLATES:
            xy = sample(PLATE_RADIUS)
            mocap = self.model.body(name).mocapid[0]
            self.data.mocap_pos[mocap] = [*xy, 0.004]
        for name in ITEMS:
            xy = sample(0.035)
            qadr = self.model.jnt_qposadr[self.model.body(name).jntadr[0]]
            z = self.model.geom_size[self.model.body(name).geomadr[0]][-1] + 0.001
            yaw = self.rng.uniform(-np.pi, np.pi)
            self.data.qpos[qadr : qadr + 7] = [*xy, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def item_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def plate_pos(self, name: str) -> np.ndarray:
        return self.data.mocap_pos[self.model.body(name).mocapid[0]].copy()

    def on_plate(self, item: str, plate: str) -> bool:
        p, q = self.item_pos(item), self.plate_pos(plate)
        return bool(np.linalg.norm(p[:2] - q[:2]) < PLATE_RADIUS and p[2] < 0.08)

    # ---- 描画 ---------------------------------------------------------------

    def render(self, camera: str = "overhead") -> np.ndarray:
        # Windows では Renderer を閉じずに2つ目を作ると真っ黒になる＝1つを使い回す
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, IMAGE_HEIGHT, IMAGE_WIDTH)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def camera(self, name: str = "overhead") -> Camera:
        """`render(name)` の画像に対応する画素⇔ワールド座標の変換。"""
        return Camera.from_model(self.model, self.data, name, IMAGE_WIDTH, IMAGE_HEIGHT)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- 動作 ---------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1
            if self.record_camera and self._step_count % self.record_every == 0:
                self.frames.append(self.render(self.record_camera))

    def settle(self, seconds: float) -> None:
        self.step(int(seconds / self.model.opt.timestep))

    def ik(self, target: np.ndarray, iters: int = 200, tol: float = 1e-3) -> np.ndarray:
        """把持点を target（真下向き）へ運ぶ関節角を、減衰最小二乗法で求める。"""
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            rot = d.xmat[self._hand].reshape(3, 3)
            tcp = d.xpos[self._hand] + rot[:, 2] * TCP_OFFSET
            err_p = target - tcp
            err_r = 0.5 * sum(np.cross(rot[:, i], self._down_rot[:, i]) for i in range(3))
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 1e-2:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, tcp, self._hand)
            jac = np.vstack([jacp, jacr])[:, :ARM_JOINTS]
            err = np.concatenate([err_p, err_r])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            d.qpos[:ARM_JOINTS] += dq
            lo, hi = self.model.jnt_range[:ARM_JOINTS].T
            d.qpos[:ARM_JOINTS] = np.clip(d.qpos[:ARM_JOINTS], lo, hi)
        return d.qpos[:ARM_JOINTS].copy()

    def move_to(self, target: np.ndarray, seconds: float = 1.0) -> None:
        """関節空間で直線補間しながら目標へ動く。"""
        q0 = self.data.ctrl[:ARM_JOINTS].copy()
        q1 = self.ik(np.asarray(target, dtype=float))
        n = int(seconds / self.model.opt.timestep)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # 始めと終わりを滑らかに
            self.data.ctrl[:ARM_JOINTS] = q0 + (q1 - q0) * s
            self.step()
        self.settle(0.2)

    def grip(self, closed: bool) -> None:
        self.data.ctrl[ARM_JOINTS] = GRIP_CLOSED if closed else GRIP_OPEN
        self.settle(0.6)

    def pick_and_place(self, pick: np.ndarray, place: np.ndarray, hover: float = 0.15) -> None:
        """机上の点 pick の物をつかみ、点 place に置く（どちらも机上の xy、z は把持高さ）。"""
        pick, place = np.asarray(pick, float), np.asarray(place, float)
        up = np.array([0, 0, hover])
        self.grip(False)
        self.move_to(pick + up, 1.2)
        self.move_to(pick, 0.8)
        self.grip(True)
        self.move_to(pick + up, 0.8)
        self.move_to(place + up, 1.2)
        self.move_to(place + np.array([0, 0, 0.03]), 0.6)
        self.grip(False)
        self.move_to(place + up, 0.6)
