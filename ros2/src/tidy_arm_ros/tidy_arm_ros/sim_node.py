"""MuJoCo の机とアームを ROS 2 のノードとして動かす（実機ならロボットとカメラのドライバに当たる）。

publish   overhead/image_raw  真上カメラ（sensor_msgs/Image・rgb8）
          front/image_raw     正面カメラ
          joint_states        アームと指の関節角
action    pick_place          机上の点 pick の物をつかんで点 place に置く（段階を feedback で返す）
service   reset               配置を seed で作り直す
          get_state           物ごとに、乗っている皿

MuJoCo の世界は1つしか持てないので、カメラとアームを1つのノードにまとめている。
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # 画面の無いコンテナで描く。mujoco を読む前に決める

import array  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.action import ActionServer  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from tidy_arm_interfaces.action import PickPlace  # noqa: E402
from tidy_arm_interfaces.srv import GetState, Reset  # noqa: E402

from tidy_arm.sim import ITEMS, PLATES, Sim  # noqa: E402

# 動いている間に publish する間隔（シミュレータの時間で 0.002s × 50 = 0.1s）
TICK_STEPS = 50
# 止まっている間に publish する間隔 [s]（計画係が最新の真上画像を持てるように）
IDLE_PERIOD = 0.5


def image_msg(image: np.ndarray, stamp, frame_id: str) -> Image:
    h, w = image.shape[:2]
    msg = Image(height=h, width=w, encoding="rgb8", is_bigendian=0, step=w * 3)
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    # bytes をそのまま渡すと1要素ずつ検査されて遅い＝array で渡す
    msg.data = array.array("B", image.tobytes())
    return msg


class TickingSim(Sim):
    """TICK_STEPS ごとに on_tick を呼ぶ Sim。publish と実時間合わせに使う。"""

    on_tick = None

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            super().step()
            if self.on_tick is not None and self._step_count % TICK_STEPS == 0:
                self.on_tick()


class SimNode(Node):
    def __init__(self):
        super().__init__("sim")
        seed = self.declare_parameter("seed", 0).value
        # true ならシミュレータの時間を実時間に合わせる（rqt や rosbag で自然な速さに見える）
        self.realtime = self.declare_parameter("realtime", True).value
        self.sim = TickingSim(seed=seed)
        self.sim.on_tick = self._tick
        m = self.sim.model
        joints = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.joint_names = [m.joint(j).name for j in joints]
        self.joint_qpos = [m.jnt_qposadr[j] for j in joints]
        self.pub_overhead = self.create_publisher(Image, "overhead/image_raw", 1)
        self.pub_front = self.create_publisher(Image, "front/image_raw", 1)
        self.pub_joints = self.create_publisher(JointState, "joint_states", 10)
        self.create_service(Reset, "reset", self._reset)
        self.create_service(GetState, "get_state", self._get_state)
        ActionServer(self, PickPlace, "pick_place", self._pick_place)
        self.create_timer(IDLE_PERIOD, self.publish)
        self._clock0: tuple[float, float] | None = None  # (実時間, シミュレータの時間)
        self.get_logger().info(f"seed={seed} joints={self.joint_names}")

    def publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.pub_overhead.publish(image_msg(self.sim.render("overhead"), stamp, "overhead"))
        self.pub_front.publish(image_msg(self.sim.render("front"), stamp, "front"))
        js = JointState(name=self.joint_names)
        js.header.stamp = stamp
        js.position = [float(self.sim.data.qpos[a]) for a in self.joint_qpos]
        self.pub_joints.publish(js)

    def _tick(self) -> None:
        self.publish()
        if self._clock0 is not None:
            wall0, sim0 = self._clock0
            ahead = (self.sim.data.time - sim0) - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)

    def _pick_place(self, goal_handle):
        g = goal_handle.request
        pick = np.array([g.pick.x, g.pick.y, g.pick.z])
        place = np.array([g.place.x, g.place.y, g.place.z])
        self.get_logger().info(f"pick_place {pick.round(3)} -> {place.round(3)}")

        def on_phase(name: str) -> None:
            goal_handle.publish_feedback(PickPlace.Feedback(phase=name))

        if self.realtime:
            self._clock0 = (time.monotonic(), self.sim.data.time)
        try:
            self.sim.pick_and_place(pick, place, on_phase=on_phase)
            self.sim.settle(0.5)
        finally:
            self._clock0 = None
        goal_handle.succeed()
        return PickPlace.Result(success=True)

    def _reset(self, request, response):
        self.sim.rng = np.random.default_rng(request.seed)
        self.sim.reset()
        self.publish()
        self.get_logger().info(f"reset seed={request.seed}")
        response.success = True
        return response

    def _get_state(self, request, response):
        response.items = list(ITEMS)
        response.plates = [next((p for p in PLATES if self.sim.on_plate(i, p)), "") for i in ITEMS]
        return response


def main() -> None:
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.close()
        node.destroy_node()
        rclpy.try_shutdown()
