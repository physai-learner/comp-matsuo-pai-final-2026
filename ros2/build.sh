#!/bin/bash
# コンテナ内で ROS 2 のパッケージをビルドする
#   docker exec pai_final bash /workspace/ros2/build.sh
set -eo pipefail
source /workspace/ros2/env.sh
mkdir -p /root/ros2_ws
cd /root/ros2_ws
# colcon を venv の python で動かす＝ノードも venv の python で起動する
python -m colcon build --symlink-install --base-paths /workspace/ros2/src
