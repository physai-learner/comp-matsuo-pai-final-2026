"""真上カメラの画素⇔机上座標の変換。正解座標との往復と、実際の描画との照合。"""

import numpy as np
import pytest

from tidy_arm.camera import Camera
from tidy_arm.sim import IMAGE_HEIGHT, IMAGE_WIDTH, ITEMS, PLATES, Sim

# 真上カメラで描画したときの上面の色（RGB 0-255・2026-10-21 実測）。
# tabletop.xml の rgba は照明で飽和するので、描画後の値を基準にする
COLORS = {
    "red_box": (255, 77, 77),
    "green_cylinder": (114, 255, 149),
    "purple_box": (223, 122, 255),
    "plate_blue": (111, 183, 255),
    "plate_yellow": (255, 255, 102),
}
COLOR_TOL = 40


@pytest.fixture(scope="module")
def sim():
    s = Sim(seed=0)
    yield s
    s.close()


def test_axes_match_scene_description(sim):
    """画像中央＝カメラ直下 (0.5, 0)。右＝ワールド -y、上＝+x。"""
    cam = sim.camera()
    cx, cy = cam.center
    assert cam.pixel_to_world([cx, cy], z=0.0) == pytest.approx([0.5, 0.0, 0.0], abs=1e-9)
    right = cam.pixel_to_world([cx + 100, cy], z=0.0)
    up = cam.pixel_to_world([cx, cy - 100], z=0.0)
    assert right[1] < 0 and abs(right[0] - 0.5) < 1e-9
    assert up[0] > 0.5 and abs(up[1]) < 1e-9


def test_roundtrip_random_points(sim):
    cam = sim.camera()
    rng = np.random.default_rng(1)
    pts = np.column_stack(
        [rng.uniform(0.2, 0.8, 50), rng.uniform(-0.4, 0.4, 50), rng.uniform(0.0, 0.1, 50)]
    )
    px = cam.world_to_pixel(pts)
    assert np.all(px >= 0) and np.all(px[:, 0] <= IMAGE_WIDTH) and np.all(px[:, 1] <= IMAGE_HEIGHT)
    back = np.array([cam.pixel_to_world(p, z=z) for p, z in zip(px, pts[:, 2], strict=True)])
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_parallax_of_item_height(sim):
    """物の上面の画素を z=0 で逆変換すると、カメラ直下から離れるほど外側へずれる。"""
    cam = sim.camera()
    top = np.array([0.65, 0.3, 0.05])
    px = cam.world_to_pixel(top)
    wrong = cam.pixel_to_world(px, z=0.0)
    right = cam.pixel_to_world(px, z=top[2])
    np.testing.assert_allclose(right, top, atol=1e-9)
    # 比例配分：ずれ = 水平距離 × h / (H - h)
    expected = np.linalg.norm(top[:2] - cam.pos[:2]) * top[2] / (cam.pos[2] - top[2])
    assert np.linalg.norm(wrong[:2] - top[:2]) == pytest.approx(expected, rel=1e-6)
    assert expected > 0.01  # 無視できない大きさ（1cm 超）


def _color_mask(img: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    return np.linalg.norm(img.astype(float) - np.array(rgb, float), axis=-1) < COLOR_TOL


def _top_center(sim: Sim, name: str) -> np.ndarray:
    """正解の3次元位置から、見えている上面の中心を出す。"""
    pos = sim.item_pos(name) if name in ITEMS else sim.plate_pos(name)
    half_h = sim.model.geom_size[sim.model.body(name).geomadr[0]][-1]
    return pos + np.array([0, 0, half_h])


@pytest.mark.parametrize("name", ITEMS + PLATES)
def test_projection_lands_on_rendered_object(sim, name):
    """正解の3次元位置を投影した画素が、描画上でその物の色になっている（軸の向きの検証）。"""
    img = sim.render("overhead")
    cam = sim.camera()
    u, v = np.round(cam.world_to_pixel(_top_center(sim, name))).astype(int)
    mask = _color_mask(img, COLORS[name])
    assert mask.sum() > 20, f"{name}: 対象色の画素が見つからない"
    assert mask[v, u], f"{name}: 投影先 ({u},{v}) の色 {img[v, u]} が対象色でない"
    # 上面の色の重心とも数画素以内（皿は平らなので側面の偏りが無い）
    vs, us = np.nonzero(mask)
    centroid = np.array([us.mean(), vs.mean()])
    tol = 3.0 if name in PLATES else 8.0
    assert np.linalg.norm([u, v] - centroid) < tol, f"{name}: centroid={centroid}"


def test_rendered_centroid_back_to_world(sim):
    """描画上の物の色の重心を、上面の高さで逆変換すると正解位置に戻る。

    箱は側面も見えるので重心がカメラ直下から外側へ偏る＝Claude が「物の中心」を
    指したときに同じ偏りが乗る。許容 2.5cm は Panda の指の開き（8cm）に対して十分小さい。
    """
    img = sim.render("overhead")
    cam = sim.camera()
    for name in ITEMS:
        vs, us = np.nonzero(_color_mask(img, COLORS[name]))
        truth = sim.item_pos(name)
        est = cam.pixel_to_world([us.mean(), vs.mean()], z=_top_center(sim, name)[2])
        err = np.linalg.norm(est[:2] - truth[:2])
        assert err < 0.025, f"{name}: err={err:.3f} est={est} truth={truth}"


def test_camera_is_pure_function_of_parameters():
    cam = Camera(
        pos=np.array([0.5, 0, 1.1]),
        rot=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float).T,  # xyaxes 0 -1 0 / 1 0 0
        width=640,
        height=480,
        fovy_deg=45,
    )
    assert cam.focal == pytest.approx(240 / np.tan(np.deg2rad(22.5)))
    assert cam.world_to_pixel([0.5, 0, 0]) == pytest.approx([320, 240])
