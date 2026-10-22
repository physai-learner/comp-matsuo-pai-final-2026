#!/usr/bin/env bash
# run_container.ps1 で起動したコンテナの中に、tidy_arm を動かす Python 環境を作る。
# コンテナを作り直したら1回だけ実行する。
#
#   docker exec pai_final bash /workspace/ros2/setup_container.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# イメージに入っている ROS の apt 署名鍵は期限切れなので、公式の新しい鍵に差し替える
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
apt-get update -qq
apt-get install -y -qq python3.10-venv >/dev/null

# rclpy と torch はイメージのものをそのまま使うので、システムのパッケージも見える venv にする
if [ ! -x /root/venv/bin/python ]; then
  python3 -m venv --system-site-packages /root/venv
fi
/root/venv/bin/pip install -q --upgrade pip
/root/venv/bin/pip install -q "mujoco>=3.16" anthropic imageio imageio-ffmpeg python-dotenv "pillow>=10"
# システムの pytest は古く、anyio のプラグインと合わないので venv 側に入れ直す
/root/venv/bin/pip install -q --ignore-installed "pytest>=8"
# システムの brotli 1.0.9 は古く、anthropic の HTTP クライアントが圧縮応答を展開できない
/root/venv/bin/pip install -q --ignore-installed "brotli>=1.2"
# tidy_arm は pyproject が Python 3.13 以上を求めて pip install できないので、パスを通して読む
site=$(/root/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo /workspace > "$site/tidy_arm.pth"

# ROS の setup.bash は未定義の変数を参照するので、読み込む間だけ set -u を外す
set +u
source /opt/ros/humble/setup.bash
set -u
MUJOCO_GL=egl /root/venv/bin/python /workspace/ros2/check_env.py
