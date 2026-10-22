#!/bin/bash
# ROS 2 版の評価を回す（sim と planner を起動 → evaluate → 止める）。Claude を試行の数だけ呼ぶ
#   docker exec pai_final bash /workspace/ros2/run_eval.sh [evaluate の引数...]
#   例: run_eval.sh --seeds 5 --lang ja en --resume
# 出力: outputs/eval_ros2.csv（--out で変更）・outputs/ros2/eval_launch.log
set -o pipefail
source /workspace/ros2/env.sh
mkdir -p /workspace/outputs/ros2

# 評価では実時間に合わせない（シミュレータを最速で回す）
ros2 launch tidy_arm_ros tidy.launch.py realtime:=false > /workspace/outputs/ros2/eval_launch.log 2>&1 &
sleep 8
ros2 run tidy_arm_ros evaluate "$@"
status=$?

# 非対話の bash では背景ジョブが SIGINT を無視する＝ノードは名前で止める
pkill -f "lib/tidy_arm_ros/(sim_node|planner_node)"
pkill -f "ros2 launch tidy_arm_ros"
sleep 2
exit $status
