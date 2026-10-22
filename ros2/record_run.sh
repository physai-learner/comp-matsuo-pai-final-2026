#!/bin/bash
# レポートと動画の素材として、ROS 2 版の通し実行を1回記録する（Claude を1回呼ぶ）
#   docker exec pai_final bash /workspace/ros2/record_run.sh [指示] [seed]
# 出力: outputs/ros2/（graph.txt＝ノード・トピック・action・service の一覧、tidy.log、launch.log）
set -o pipefail
INSTRUCTION=${1:-全部の物を黄色い皿に置いて}
SEED=${2:-5}
OUT=/workspace/outputs/ros2
source /workspace/ros2/env.sh
mkdir -p $OUT

ros2 launch tidy_arm_ros tidy.launch.py seed:=$SEED > $OUT/launch.log 2>&1 &
sleep 8
{
  echo "## nodes";    ros2 node list
  echo "## topics";   ros2 topic list -t
  echo "## actions";  ros2 action list -t
  echo "## services"; ros2 service list -t | grep -vE "parameter|/rosout"
  for n in /sim /planner; do echo "## info $n"; ros2 node info $n; done
} > $OUT/graph.txt 2>&1
ros2 run tidy_arm_ros tidy "$INSTRUCTION" 2>&1 | tee $OUT/tidy.log

# 非対話の bash では背景ジョブが SIGINT を無視する＝ノードは名前で止める
pkill -f "lib/tidy_arm_ros/(sim_node|planner_node)"
pkill -f "ros2 launch tidy_arm_ros"
sleep 2
echo "記録 → $OUT"
