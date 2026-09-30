"""Deep Q-network with ReLU or FTA in the last hidden layer.

Network: input -> 64 (ReLU) -> 64 -> [activation] -> linear Q-values, as in the LunarLander setting of
Pan et al. (2021) reproduced by the paper. Variants:
  * relu:  ReLU in the last hidden layer ("DQN")
  * large: ReLU, with the last hidden layer widened to 64 * k units, the size of DQN-FTA's features ("DQN-Large")
  * fta:   FTA in the last hidden layer ("DQN-FTA"), optionally preceded by tanh, BatchNorm or RangeNorm
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict, field

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .layers import FTA, RangeNorm


@dataclass
class DQNConfig:
    env: str = "LunarLander-v3"
    net: str = "fta"                 # relu | large | fta
    norm: str = "none"               # none | tanh | batchnorm | rangenorm (before FTA)
    bound: float = 20.0              # FTA tiling bound u, with l = -u
    tiles: int = 20                  # k
    eta: float | None = None         # defaults to delta = 2u / k
    hidden: int = 64
    target_network: bool = False
    target_update: int = 1000        # steps between target-network copies
    steps: int = 500_000
    lr: float = 1e-4
    gamma: float = 0.99
    batch: int = 64
    buffer: int = 100_000
    warmup: int = 1_000              # random steps before learning starts
    epsilon: float = 0.1
    eval_every: int = 10_000         # log the mean return of recent episodes this often
    eval_window: int = 10            # episodes averaged at each log point
    seed: int = 0
    label: str = field(default="", compare=False)


def build_network(cfg: DQNConfig, obs_dim: int, n_actions: int) -> nn.Module:
    h = cfg.hidden
    layers: list[nn.Module] = [nn.Linear(obs_dim, h), nn.ReLU()]
    if cfg.net == "relu":
        layers += [nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions)]
    elif cfg.net == "large":
        layers += [nn.Linear(h, h * cfg.tiles), nn.ReLU(), nn.Linear(h * cfg.tiles, n_actions)]
    elif cfg.net == "fta":
        layers.append(nn.Linear(h, h))
        layers += {"none": [], "tanh": [nn.Tanh()], "batchnorm": [nn.BatchNorm1d(h)], "rangenorm": [RangeNorm()]}[cfg.norm]
        layers += [FTA(-cfg.bound, cfg.bound, cfg.tiles, cfg.eta), nn.Linear(h * cfg.tiles, n_actions)]
    else:
        raise ValueError(f"unknown net {cfg.net!r}")
    return nn.Sequential(*layers)


class Replay:
    def __init__(self, size: int, obs_dim: int):
        self.s = np.zeros((size, obs_dim), np.float32); self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(size, np.int64); self.r = np.zeros(size, np.float32); self.d = np.zeros(size, np.float32)
        self.size, self.i, self.n = size, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, k: int, rng: np.random.Generator):
        j = rng.integers(0, self.n, k)
        return (torch.from_numpy(self.s[j]), torch.from_numpy(self.a[j]), torch.from_numpy(self.r[j]),
                torch.from_numpy(self.s2[j]), torch.from_numpy(self.d[j]))


def train(cfg: DQNConfig) -> dict:
    """Train one agent; returns the config and the learning curve (mean return of recent episodes by step)."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed)
    env = gym.make(cfg.env)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q = build_network(cfg, obs_dim, n_actions)
    target = build_network(cfg, obs_dim, n_actions) if cfg.target_network else q
    if cfg.target_network:
        target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    buf = Replay(cfg.buffer, obs_dim)

    s, _ = env.reset(seed=cfg.seed)
    ep_ret, returns, curve = 0.0, [], []
    t0 = time.time()
    for t in range(1, cfg.steps + 1):
        if t <= cfg.warmup or rng.random() < cfg.epsilon:
            a = int(rng.integers(n_actions))
        else:
            q.eval()
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)).argmax())
            q.train()
        s2, r, term, trunc, _ = env.step(a)
        buf.add(s, a, r, s2, float(term))
        ep_ret += r
        s = s2
        if term or trunc:
            returns.append(ep_ret); ep_ret = 0.0
            s, _ = env.reset()

        if t > cfg.warmup:
            bs, ba, br, bs2, bd = buf.sample(cfg.batch, rng)
            with torch.no_grad():
                target.eval()
                y = br + cfg.gamma * (1 - bd) * target(bs2).max(1).values
                target.train()
            qsa = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(qsa, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if cfg.target_network and t % cfg.target_update == 0:
                target.load_state_dict(q.state_dict())

        if t % cfg.eval_every == 0:
            recent = returns[-cfg.eval_window:]
            curve.append((t, float(np.mean(recent)) if recent else float("nan")))
    env.close()
    return {"config": asdict(cfg), "curve": curve, "episodes": len(returns), "seconds": round(time.time() - t0, 1)}
