# 言葉で頼めるお片付けロボットアーム

日本語の指示（例：「赤い箱を青い皿に乗せて」）を受けて、シミュレータ上のロボットアームが物をつかんで運ぶ。

- シミュレータ：MuJoCo
- 指示の解釈：Claude（推論のみ）にカメラ画像と指示を渡し、対象物と置き場所の画像座標を返させる
- 動作：画像座標を3次元位置に変換し、逆運動学（IK）で把持・運搬する
- 評価：指示と物の配置を変えた試行を繰り返し、成功率と失敗の内訳を記録する

## 環境

- Python 3.13・[uv](https://docs.astral.sh/uv/)
- `uv sync` で依存を入れる
- Claude の API キーは環境変数 `ANTHROPIC_API_KEY`（または `.env`）で渡す（リポジトリに含めない）

## 使用したOSS

- [MuJoCo](https://github.com/google-deepmind/mujoco)（Apache-2.0）
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)（MIT）
