"""計画係を ROS 2 のノードにする。最新の真上画像と指示文から、運ぶ物と置き先を決める。

subscribe overhead/image_raw
action    plan   指示文 → 計画（画素と机上の点）。運ばないときは problem に理由

画素を机上の点へ戻すのに使うカメラの位置と画角は、シーンの定義から読む（実機なら較正値に当たる）。
"""

import threading
import time

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tidy_arm_interfaces.action import Plan
from tidy_arm_interfaces.msg import Move

from tidy_arm.camera import Camera
from tidy_arm.planner import ClaudePlanner, PlanError, to_world
from tidy_arm.sim import IMAGE_HEIGHT, IMAGE_WIDTH, SCENE

# goal を受けてから撮られた画像をこの時間まで待つ（sim は止まっていても 0.5 秒ごとに publish する）
FRESH_TIMEOUT = 3.0


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
        self.image_stamp = Time(clock_type=self.get_clock().clock_type)
        self.image_cond = threading.Condition()
        # 計画中も画像を受け取れるよう、購読と action を別のグループにして並行に動かす
        self.create_subscription(
            Image,
            "overhead/image_raw",
            self._on_image,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        ActionServer(
            self, Plan, "plan", self._plan, callback_group=MutuallyExclusiveCallbackGroup()
        )

    def _on_image(self, msg: Image) -> None:
        image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        with self.image_cond:
            self.image = image
            self.image_stamp = Time.from_msg(msg.header.stamp)
            self.image_cond.notify_all()

    def fresh_image(self, since: Time) -> np.ndarray | None:
        """since より後に撮られた真上画像（reset や運搬の直後に古い画像で計画しないため）。"""
        deadline = time.monotonic() + FRESH_TIMEOUT
        with self.image_cond:
            while self.image is None or self.image_stamp < since:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.image_cond.wait(left)
            return self.image.copy()

    def _plan(self, goal_handle):
        instruction = goal_handle.request.instruction
        result = Plan.Result()
        image = self.fresh_image(self.get_clock().now())
        if image is None:
            result.problem = "新しい真上画像が届かない（sim_node が動いているか確認）"
            result.error = "no_image"
            goal_handle.abort()
            return result
        self.get_logger().info(f"plan: {instruction}")
        goal_handle.publish_feedback(Plan.Feedback(status="asking Claude"))
        n_usage = len(self.planner.usage_log)
        try:
            plan = self.planner.plan(image, instruction)
        except PlanError as e:
            result.problem = str(e)
            result.error = e.kind
            plan = None
        usage = self.planner.usage_log[n_usage:]
        result.input_tokens = sum(u["input_tokens"] for u in usage)
        result.output_tokens = sum(u["output_tokens"] for u in usage)
        if plan is not None:
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
        self.get_logger().info(
            f"moves={len(result.moves)} problem={result.problem!r} "
            f"tokens in={result.input_tokens} out={result.output_tokens}"
        )
        goal_handle.succeed()
        return result


def main() -> None:
    rclpy.init()
    node = PlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
