"""ACT（Action Chunking with Transformers, Zhao et al. 2023）の小さな実装と学習ループ。

mujoco にも tidy_arm の他のファイルにも依存しない＝このファイルとデモ（.npz）だけを
Colab へ持っていけば学習できる。

    python act.py --data demos.npz --out ckpt --steps 20000

構成は論文と LeRobot の既定に合わせた：ResNet18（BatchNorm は固定）の特徴＋関節状態＋潜在変数を
Transformer の encoder に入れ、decoder が先 K 歩ぶんの目標をまとめて出す。学習時だけ CVAE の
encoder が正解の行動列から潜在変数を作り、推論時は 0 を入れる。
"""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ActConfig:
    state_dim: int = 8
    action_dim: int = 8
    chunk: int = 20  # 0.1秒刻みで 2秒先まで
    d_model: int = 256
    n_heads: int = 8
    ff_dim: int = 1024
    enc_layers: int = 4
    dec_layers: int = 1
    vae_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0


def sine_2d(h: int, w: int, d: int) -> torch.Tensor:
    """画像特徴の位置埋め込み（h*w, d）。前半が縦・後半が横の正弦波。"""
    q = d // 4
    freq = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32) / q))
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )

    def enc(v: torch.Tensor) -> torch.Tensor:
        a = v.reshape(-1, 1) * freq
        return torch.cat([a.sin(), a.cos()], dim=1)

    return torch.cat([enc(ys), enc(xs)], dim=1)


def _encoder(cfg: ActConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        cfg.d_model, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ACT(nn.Module):
    def __init__(self, cfg: ActConfig, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B, 512, H/32, W/32)
        self.img_proj = nn.Conv2d(512, d, 1)
        self.state_proj = nn.Linear(cfg.state_dim, d)
        self.latent_proj = nn.Linear(cfg.latent_dim, d)
        self.token_pos = nn.Embedding(2, d)  # 潜在変数・関節状態のトークン
        self.encoder = _encoder(cfg, cfg.enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d, cfg.n_heads, cfg.ff_dim, cfg.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)
        self.queries = nn.Embedding(cfg.chunk, d)
        self.head = nn.Linear(d, cfg.action_dim)
        # CVAE の encoder：[CLS, 関節状態, 行動 K 個] → 潜在変数の平均と分散
        self.vae_cls = nn.Embedding(1, d)
        self.vae_state = nn.Linear(cfg.state_dim, d)
        self.vae_action = nn.Linear(cfg.action_dim, d)
        self.vae_pos = nn.Embedding(cfg.chunk + 2, d)
        self.vae_encoder = _encoder(cfg, cfg.vae_layers)
        self.vae_out = nn.Linear(d, 2 * cfg.latent_dim)
        # 正規化の統計（学習データから `set_stats` で入れ、チェックポイントに一緒に保存される）
        for name, dim in [("state", cfg.state_dim), ("action", cfg.action_dim)]:
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_std", torch.ones(dim))
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def set_stats(self, state: np.ndarray, action: np.ndarray) -> None:
        for name, x in [("state", state), ("action", action)]:
            getattr(self, f"{name}_mean").copy_(torch.from_numpy(x.mean(0)))
            # 動かない成分（ほぼ 0 か 1 の指など）で割り算が暴れないよう下限を置く
            getattr(self, f"{name}_std").copy_(torch.from_numpy(x.std(0)).clamp(min=1e-2))

    def forward(self, image, state, actions=None, pad=None):
        """image (B,3,H,W) 0〜1・state (B,S)・actions (B,K,A)・pad (B,K) True=詰め物。

        返り値＝正規化した行動の予測 (B,K,A)・潜在変数の平均・対数分散（推論時は None）。
        """
        cfg = self.cfg
        b = state.shape[0]
        s = (state - self.state_mean) / self.state_std
        if actions is not None:
            a = (actions - self.action_mean) / self.action_std
            tokens = (
                torch.cat(
                    [
                        self.vae_cls.weight.expand(b, 1, -1),
                        self.vae_state(s)[:, None],
                        self.vae_action(a),
                    ],
                    dim=1,
                )
                + self.vae_pos.weight[None]
            )
            keep = torch.zeros(b, 2, dtype=torch.bool, device=state.device)
            h = self.vae_encoder(tokens, src_key_padding_mask=torch.cat([keep, pad], dim=1))
            mu, logvar = self.vae_out(h[:, 0]).chunk(2, dim=-1)
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(b, cfg.latent_dim, device=state.device)

        f = self.img_proj(self.backbone((image - self.img_mean) / self.img_std))
        fh, fw = f.shape[-2:]
        img_tokens = f.flatten(2).transpose(1, 2) + sine_2d(fh, fw, cfg.d_model).to(f)[None]
        src = torch.cat(
            [
                (self.latent_proj(z) + self.token_pos.weight[0])[:, None],
                (self.state_proj(s) + self.token_pos.weight[1])[:, None],
                img_tokens,
            ],
            dim=1,
        )
        memory = self.encoder(src)
        out = self.decoder(self.queries.weight.expand(b, -1, -1), memory)
        return self.head(out), mu, logvar

    def loss(self, image, state, actions, pad) -> tuple[torch.Tensor, dict]:
        pred, mu, logvar = self(image, state, actions, pad)
        target = (actions - self.action_mean) / self.action_std
        valid = (~pad)[..., None].float()
        l1 = (F.l1_loss(pred, target, reduction="none") * valid).sum() / (
            valid.sum() * self.cfg.action_dim
        )
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.cfg.kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}

    @torch.no_grad()
    def predict(self, image, state) -> torch.Tensor:
        """元の単位に戻した行動列 (B,K,A)。"""
        pred, _, _ = self(image, state)
        return pred * self.action_std + self.action_mean


# ---- データ ------------------------------------------------------------------


def save_demos(path: Path, episodes: list[dict]) -> None:
    """エピソードを1つの .npz にまとめる。episode_ends は各エピソードの終わり（累積）。"""
    ends = np.cumsum([len(e["state"]) for e in episodes])
    np.savez_compressed(
        path,
        images=np.concatenate([e["images"] for e in episodes]),
        state=np.concatenate([e["state"] for e in episodes]),
        action=np.concatenate([e["action"] for e in episodes]),
        episode_ends=ends,
    )


class Demos:
    """全フレームをメモリに載せ、(画像, 関節状態, 先 K 歩の行動, 詰め物の印) を取り出す。

    画像は 200本で約 4.5GB＝学習用と検証用はコピーせず、同じ配列の前後を切り分けて使う。
    """

    def __init__(self, arrays: dict, ends: np.ndarray):
        lengths = np.diff(np.concatenate([[0], ends]))
        self.images, self.state, self.action = arrays["images"], arrays["state"], arrays["action"]
        self.end_of = np.repeat(ends, lengths)  # 各フレームが属するエピソードの終わり
        self.n_episodes = len(ends)

    @classmethod
    def split(cls, path: Path, n_val: int) -> tuple["Demos", "Demos"]:
        """後ろの n_val 本を検証用にする。"""
        z = np.load(path)
        ends = z["episode_ends"]
        arrays = {k: z[k] for k in ("images", "state", "action")}
        cut = ends[-n_val - 1]
        head = {k: v[:cut] for k, v in arrays.items()}
        tail = {k: v[cut:] for k, v in arrays.items()}
        return cls(head, ends[:-n_val]), cls(tail, ends[-n_val:] - cut)

    def __len__(self) -> int:
        return len(self.state)

    def batch(self, idx: np.ndarray, chunk: int, device) -> tuple[torch.Tensor, ...]:
        offsets = idx[:, None] + np.arange(chunk)[None]
        end = self.end_of[idx][:, None]
        pad = offsets >= end
        # エピソードの終わりを越えた分は最後の行動で埋め、損失からは外す
        offsets = np.minimum(offsets, end - 1)
        image = torch.from_numpy(self.images[idx]).to(device).permute(0, 3, 1, 2).float() / 255
        state = torch.from_numpy(self.state[idx]).to(device)
        actions = torch.from_numpy(self.action[offsets]).to(device)
        return image, state, actions, torch.from_numpy(pad).to(device)


# ---- 推論 --------------------------------------------------------------------


class ActPolicy:
    """毎歩 K 歩先まで予測し、同じ時刻への過去の予測を重み付き平均する（temporal ensembling）。

    重みは exp(-m*i)、i=0 が最も古い予測（論文の既定 m=0.01）。
    """

    def __init__(self, ckpt: Path, device: str = "cpu", m: float = 0.01):
        blob = torch.load(ckpt, map_location=device, weights_only=True)
        self.model = ACT(ActConfig(**blob["cfg"]), pretrained=False).to(device).eval()
        self.model.load_state_dict(blob["model"])
        self.device = device
        self.m = m
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.chunks: list[tuple[int, np.ndarray]] = []

    def __call__(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float() / 255
        s = torch.from_numpy(state).to(self.device)[None]
        chunk = self.model.predict(x, s)[0].cpu().numpy()
        k = self.model.cfg.chunk
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < k]
        self.chunks.append((self.t, chunk))
        preds = np.stack([c[self.t - t0] for t0, c in self.chunks])
        w = np.exp(-self.m * np.arange(len(preds)))
        self.t += 1
        return (preds * w[:, None]).sum(0) / w.sum()


# ---- 学習 --------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_eps = len(np.load(args.data)["episode_ends"])
    data, val = Demos.split(args.data, max(1, int(n_eps * args.val_frac)))
    print(f"device={device} train={data.n_episodes}本/{len(data)}枚", end=" ")
    print(f"val={val.n_episodes}本/{len(val)}枚")

    cfg = ActConfig(chunk=args.chunk, kl_weight=args.kl_weight)
    model = ACT(cfg, pretrained=not args.no_pretrained).to(device)
    model.set_stats(data.state, data.action)
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr_backbone}, {"params": rest, "lr": args.lr}],
        weight_decay=1e-4,
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = 0
    last = out / "last.pt"
    if args.resume and last.exists():
        blob = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        start = blob["step"]
        print(f"再開 step={start}")

    def save(path: Path, step: int) -> None:
        torch.save(
            {
                "cfg": asdict(cfg),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
            },
            path,
        )

    rng = np.random.default_rng(args.seed + start)
    log = out / "log.csv"
    t0 = time.time()
    for step in range(start, args.steps):
        model.train()
        batch = data.batch(rng.integers(0, len(data), args.batch), cfg.chunk, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, parts = model.loss(*batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            v = validate(model, val, cfg.chunk, device)
            row = {"step": done, **parts, "val_l1": v, "sec": round(time.time() - t0)}
            print(row, flush=True)
            new = not log.exists()
            with open(log, "a", encoding="utf-8") as f:
                if new:
                    f.write(",".join(row) + "\n")
                f.write(",".join(str(x) for x in row.values()) + "\n")
        if done % args.save_every == 0 or done == args.steps:
            save(last, done)
    save(out / "final.pt", args.steps)


@torch.no_grad()
def validate(model: ACT, val: Demos, chunk: int, device, n: int = 256) -> float:
    """潜在変数 0（推論と同じ条件）での L1。検証用エピソードから等間隔に n 枚。"""
    model.eval()
    idx = np.linspace(0, len(val) - 1, min(n, len(val))).astype(int)
    err = count = 0.0
    for i in range(0, len(idx), 64):
        image, state, actions, pad = val.batch(idx[i : i + 64], chunk, device)
        pred = model.predict(image, state)
        valid = (~pad)[..., None].float()
        err += (((pred - actions) / model.action_std).abs() * valid).sum().item()
        count += valid.sum().item() * pred.shape[-1]
    return round(err / count, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="out/last.pt から続ける")
    ap.add_argument(
        "--no-pretrained", action="store_true", help="ImageNet の重みを落とさない（試験用）"
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
