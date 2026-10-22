# ROS 2 版（3ノード構成）

講義で使った Docker イメージ（ROS 2 Humble・Ubuntu 22.04・Python 3.10）の上で、
tidy_arm をそのまま ROS 2 のノードに載せたもの。物理とロボットの制御は `tidy_arm` と同じコードを使う。

## 構成

```
            /overhead/image_raw
  sim ───────────────────────────▶ planner
  (MuJoCo の机とアーム)              (Claude で計画)
   ▲  /front/image_raw                 ▲
   │  /joint_states                    │ action /plan
   │                                   │
   │ action /pick_place          tidy（指示を1つ受けて順に呼ぶ）
   │ service /reset /get_state ◀───────┘
```

| ノード | 役割 | 出すもの・受けるもの |
|---|---|---|
| `sim` | 実機ならロボットとカメラのドライバに当たる | 画像2本と関節角を publish・`pick_place`（段階を feedback）・`reset`・`get_state` |
| `planner` | 真上画像と指示文から運ぶ物と置き先を決める | `overhead/image_raw` を subscribe・`plan`（画素を机上の点へ戻して返す） |
| `tidy` | 指示を1つ受けて計画 → 運搬 → 確認を順に呼ぶ | 上の action と service の呼び出し側 |

カメラとアームは同じ MuJoCo の世界を共有するので、`sim` に1つにまとめている。
インターフェースの定義は `src/tidy_arm_interfaces`（`Move.msg`・`Plan.action`・`PickPlace.action`・`Reset.srv`・`GetState.srv`）。

## 動かし方（Windows の PowerShell から）

```powershell
# 1. コンテナを起動する（初回はイメージの取得に時間がかかる）
.\ros2\run_container.ps1
# 2. Python 環境を作る（コンテナを作り直したときだけ）
docker exec pai_final bash /workspace/ros2/setup_container.sh
# 3. ビルドする（インターフェースを変えたときだけ。ノードの .py はそのまま反映される）
docker exec pai_final bash /workspace/ros2/build.sh
```

端末を2つ使う。Claude の API キーはリポジトリ直下の `.env`（`ANTHROPIC_API_KEY`）から読む。

```powershell
# 端末1: sim と planner を起動する（seed で配置が変わる）
docker exec -it pai_final bash -c "source /workspace/ros2/env.sh && ros2 launch tidy_arm_ros tidy.launch.py seed:=3"
# 端末2: 指示を出す（--seed を付けると先に配置を作り直す）
docker exec -it pai_final bash -c "source /workspace/ros2/env.sh && ros2 run tidy_arm_ros tidy '赤い箱を青い皿に置いて'"
```

`tidy` は計画した運搬と、各段階（approach・grasp・carry・release・retreat）、最後に物ごとの皿を表示する。
launch の引数は `seed`（既定 0）・`realtime`（既定 true＝実時間に合わせて動かす）・`effort`（既定 medium）。

## 動いている様子を見る

```powershell
docker exec -it pai_final bash -c "source /workspace/ros2/env.sh && ros2 topic hz /overhead/image_raw"
docker exec -it pai_final bash -c "source /workspace/ros2/env.sh && ros2 action list -t"
```

ブラウザで http://localhost:6080/ を開くと講義と同じデスクトップが出るので、`rqt_image_view` で画像を見られる。

## 注意

- 描画は EGL（`MUJOCO_GL=egl`）。コンテナに GPU は無いのでソフトウェア描画になり、`sim` は CPU を数コア使う。
- コンテナ内で pytest を回すときは、ROS の pytest プラグインを外す:
  `python -m pytest -q -p no:launch_testing -p no:launch_ros -p no:ament_lint`
