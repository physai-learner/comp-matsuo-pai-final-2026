"""真上カメラの画素と机上の座標の相互変換。

MuJoCo のカメラはピンホール。カメラ座標系は -z が視線、+y が画像の上。
画像の原点は左上、u が右、v が下。fovy と画像の高さから焦点距離を決める（画素は正方形）。
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Camera:
    """1つのカメラの内部・外部パラメータ。`from_sim` で MuJoCo から取る。"""

    pos: np.ndarray  # ワールド座標でのカメラ位置 (3,)
    rot: np.ndarray  # カメラ→ワールドの回転 (3,3)。列が x/y/z 軸
    width: int
    height: int
    fovy_deg: float

    @classmethod
    def from_model(
        cls, model: mujoco.MjModel, data: mujoco.MjData, name: str, width: int, height: int
    ) -> "Camera":
        cam = model.camera(name)
        return cls(
            pos=np.array(data.cam_xpos[cam.id], dtype=float),
            rot=np.array(data.cam_xmat[cam.id], dtype=float).reshape(3, 3),
            width=width,
            height=height,
            fovy_deg=float(model.cam_fovy[cam.id]),
        )

    @property
    def focal(self) -> float:
        return (self.height / 2) / np.tan(np.deg2rad(self.fovy_deg) / 2)

    @property
    def center(self) -> tuple[float, float]:
        return self.width / 2, self.height / 2

    def world_to_pixel(self, point: np.ndarray) -> np.ndarray:
        """ワールド座標の点 (…,3) → 画素 (…,2)。画素は連続値（u 右・v 下）。"""
        p = np.asarray(point, dtype=float)
        pc = (p - self.pos) @ self.rot  # カメラ座標。rot.T @ (p - pos) をまとめて
        depth = -pc[..., 2]
        if np.any(depth <= 0):
            raise ValueError("カメラの後ろの点は投影できない")
        cx, cy = self.center
        u = cx + self.focal * pc[..., 0] / depth
        v = cy - self.focal * pc[..., 1] / depth
        return np.stack([u, v], axis=-1)

    def pixel_to_world(self, pixel: np.ndarray, z: float = 0.0) -> np.ndarray:
        """画素 (…,2) → 高さ z の水平面と視線の交点 (…,3)。

        物の中心や把持点の高さ z を渡す。机上なら 0、物の上面なら物の高さ。
        真上から見ているので、z を間違えると視差分だけ横にずれる。
        """
        px = np.asarray(pixel, dtype=float)
        cx, cy = self.center
        ray_cam = np.stack(
            [
                (px[..., 0] - cx) / self.focal,
                (cy - px[..., 1]) / self.focal,
                -np.ones(px.shape[:-1]),
            ],
            axis=-1,
        )
        ray = ray_cam @ self.rot.T  # ワールド方向。rot @ ray_cam をまとめて
        t = (z - self.pos[2]) / ray[..., 2]
        if np.any(t <= 0):
            raise ValueError("視線が高さ z の面と前方で交わらない")
        return self.pos + t[..., None] * ray
