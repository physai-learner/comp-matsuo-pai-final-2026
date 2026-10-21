import numpy as np
import pytest

torch = pytest.importorskip("torch")  # 学習用の依存グループ（uv run --group train）を入れたときだけ

from tidy_arm.act import ACT, ActConfig, ActPolicy, Demos, save_demos  # noqa: E402
from tidy_arm.demos import collect_episode  # noqa: E402

CFG = ActConfig(chunk=5, d_model=32, n_heads=4, ff_dim=64, enc_layers=1, vae_layers=1)


def _episode(n: int, offset: float) -> dict:
    return {
        "images": np.zeros((n, 64, 96, 3), np.uint8),
        "state": np.full((n, 8), offset, np.float32),
        "action": (np.arange(n, dtype=np.float32)[:, None] + offset).repeat(8, 1),
    }


def test_split_pads_past_episode_end(tmp_path):
    path = tmp_path / "d.npz"
    save_demos(path, [_episode(7, 0), _episode(4, 100), _episode(3, 200)])
    train, val = Demos.split(path, n_val=1)
    assert (train.n_episodes, len(train), val.n_episodes, len(val)) == (2, 11, 1, 3)
    # 1本目の最後から2枚目：残り2歩＋詰め物3つ（最後の行動の繰り返し）
    _, _, actions, pad = train.batch(np.array([5]), 5, "cpu")
    assert actions[0, :, 0].tolist() == [5, 6, 6, 6, 6]
    assert pad[0].tolist() == [False, False, True, True, True]
    # 検証用は3本目だけ＝2本目の行動が混ざらない
    _, _, actions, _ = val.batch(np.array([0]), 5, "cpu")
    assert actions[0, :, 0].tolist() == [200, 201, 202, 202, 202]


def test_loss_and_predict_shapes():
    model = ACT(CFG, pretrained=False)
    image = torch.rand(2, 3, 64, 96)
    state = torch.rand(2, 8)
    actions = torch.rand(2, 5, 8)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    loss, parts = model.loss(image, state, actions, pad)
    loss.backward()
    assert np.isfinite(parts["l1"]) and np.isfinite(parts["kl"])
    assert model.predict(image, state).shape == (2, 5, 8)


def test_policy_round_trips_checkpoint(tmp_path):
    from dataclasses import asdict

    model = ACT(CFG, pretrained=False)
    model.set_stats(
        np.random.rand(10, 8).astype(np.float32), np.random.rand(10, 8).astype(np.float32)
    )
    torch.save({"cfg": asdict(CFG), "model": model.state_dict()}, tmp_path / "c.pt")
    policy = ActPolicy(tmp_path / "c.pt")
    image, state = np.zeros((64, 96, 3), np.uint8), np.zeros(8, np.float32)
    for _ in range(7):  # chunk 5 を越えて古い予測が捨てられる
        assert policy(image, state).shape == (8,)
    assert len(policy.chunks) == CFG.chunk


def test_collect_episode_pairs_observation_with_next_command():
    ep = collect_episode(1000)
    n = len(ep["state"])
    assert ep["images"].shape == (n, 240, 320, 3) and ep["action"].shape == (n, 8)
    grip = ep["action"][:, 7]
    assert set(np.unique(grip)) <= {0.0, 1.0} and grip.max() == 1.0  # 途中で閉じ、最後は開く
    assert grip[-1] == 0.0
