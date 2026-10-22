"""コンテナの中で rclpy・MuJoCo（画面なしの描画）・tidy_arm がそろって使えるかを確かめる。"""

import sys

sys.path.insert(0, "/workspace")

import mujoco  # noqa: E402
import rclpy  # noqa: E402

from tidy_arm import scene  # noqa: E402, F401

rclpy.init()
node = rclpy.create_node("check_env")
print("rclpy:", node.get_name())
node.destroy_node()
rclpy.shutdown()

xml = (
    '<mujoco><worldbody><light pos="0 0 1"/><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, 240, 320) as renderer:
    renderer.update_scene(data)
    print("mujoco", mujoco.__version__, "render mean:", round(float(renderer.render().mean()), 1))
print("OK")
