"""計画係を ROS 2 のノードにする。最新の真上画像と指示文から、運ぶ物と置き先を決める。

subscribe overhead/image_raw
action    plan   指示文 → 計画（画素と机上の点）。運ばないときは problem に理由

画素を机上の点へ戻すのに使うカメラの位置と画角は、シーンの定義から読む（実機なら較正値に当たる）。
"""

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import Image
from tidy_arm_interfaces.action import Plan
from tidy_arm_interfaces.msg import Move

from tidy_arm.camera import Camera
from tidy_arm.planner import ClaudePlanner, PlanError, to_world
from tidy_arm.sim import IMAGE_HEIGHT, IMAGE_WIDTH, SCENE


def overhead_camera() -> Camera:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return Camera.from_model(model, data, "overhead", IMAGE_WIDTH, IMAGE_HEIGHT)


def point(xyz: np.ndarray) -> Point:
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner")
        effort = self.declare_parameter("effort", "medium").value
        self.planner = ClaudePlanner(effort=effort)
        self.camera = overhead_camera()
        self.image: np.ndarray | None = None
        self.create_subscription(Image, "overhead/image_raw", self._on_image, 1)
        ActionServer(self, Plan, "plan", self._plan)

    def _on_image(self, msg: Image) -> None:
        self.image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _plan(self, goal_handle):
        instruction = goal_handle.request.instruction
        result = Plan.Result()
        if self.image is None:
            result.problem = "真上画像をまだ受け取っていない（sim_node が動いているか確認）"
            goal_handle.abort()
            return result
        image = self.image.copy()
        self.get_logger().info(f"plan: {instruction}")
        goal_handle.publish_feedback(Plan.Feedback(status="asking Claude"))
        try:
            plan = self.planner.plan(image, instruction)
        except PlanError as e:
            result.problem = str(e)
            goal_handle.succeed()
            return result
        for m in plan.moves:
            pick, place = to_world(self.camera, m)
            result.moves.append(
                Move(
                    item=m.item,
                    place_target=m.place_target,
                    pick_u=m.pick.u,
                    pick_v=m.pick.v,
                    place_u=m.place.u,
                    place_v=m.place.v,
                    pick=point(pick),
                    place=point(place),
                )
            )
        result.problem = plan.problem
        usage = self.planner.usage_log[-1]
        self.get_logger().info(
            f"moves={len(result.moves)} problem={plan.problem!r} "
            f"tokens in={usage['input_tokens']} out={usage['output_tokens']}"
        )
        goal_handle.succeed()
        return result


def main() -> None:
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
