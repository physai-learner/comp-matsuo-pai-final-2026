"""指示文を1つ受けて、計画 → 運搬 → 結果の確認までを順に呼ぶ。

    ros2 run tidy_arm_ros tidy "赤い箱を青い皿に置いて" [--seed 3]

--seed を付けると、先に配置を作り直す。
"""

import argparse

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from tidy_arm_interfaces.action import PickPlace, Plan
from tidy_arm_interfaces.srv import GetState, Reset


class Commander(Node):
    def __init__(self):
        super().__init__("tidy")
        self.plan_client = ActionClient(self, Plan, "plan")
        self.pick_place_client = ActionClient(self, PickPlace, "pick_place")
        self.reset_client = self.create_client(Reset, "reset")
        self.state_client = self.create_client(GetState, "get_state")

    def _wait(self, future):
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"service {client.srv_name} が見つからない")
        return self._wait(client.call_async(request))

    def run_action(self, client, goal, on_feedback=None):
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"action {client._action_name} が見つからない")
        handle = self._wait(client.send_goal_async(goal, feedback_callback=on_feedback))
        if not handle.accepted:
            raise RuntimeError("goal を受け付けなかった")
        return self._wait(handle.get_result_async()).result

    def tidy(self, instruction: str, seed: int | None) -> None:
        log = self.get_logger()
        if seed is not None:
            self.call(self.reset_client, Reset.Request(seed=seed))
            log.info(f"reset seed={seed}")
        plan = self.run_action(self.plan_client, Plan.Goal(instruction=instruction))
        if plan.problem:
            log.info(f"problem: {plan.problem}")
        for m in plan.moves:
            log.info(
                f"{m.item} -> {m.place_target}  "
                f"({m.pick.x:.3f}, {m.pick.y:.3f}) -> ({m.place.x:.3f}, {m.place.y:.3f})"
            )
            self.run_action(
                self.pick_place_client,
                PickPlace.Goal(pick=m.pick, place=m.place),
                lambda fb: log.info(f"  {fb.feedback.phase}"),
            )
        state = self.call(self.state_client, GetState.Request())
        for item, plate in zip(state.items, state.plates, strict=True):
            log.info(f"{item}: {plate or '-'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument("--seed", type=int)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Commander()
    try:
        node.tidy(args.instruction, args.seed)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
