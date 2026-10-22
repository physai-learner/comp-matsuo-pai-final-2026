# コンテナ内で source する: ROS 2 Humble・venv・ビルド済みのワークスペースを読み込む
#   source /workspace/ros2/env.sh
set +u  # ROS の setup.bash は未定義の変数を参照する
source /opt/ros/humble/setup.bash
source /root/venv/bin/activate
if [ -f /root/ros2_ws/install/setup.bash ]; then
  source /root/ros2_ws/install/setup.bash
fi
export MUJOCO_GL=egl
