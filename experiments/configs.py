"""Every experiment in Raj and Shaik (ETTIS 2024), as run configurations.

PAPER lists the full-scale settings described in the paper (500k steps; 10 runs, or 50 for the
distribution study). QUICK is a reduced version that fits on a laptop CPU in about an hour; its results are
the ones published in results/.
"""
from fta import DQNConfig

LL, CP = "LunarLander-v3", "CartPole-v1"


def _fta(u, k=20, norm="none", **kw):
    tag = {"none": "", "tanh": "-tanh", "batchnorm": "-batchnorm", "rangenorm": "-rangenorm"}[norm]
    return DQNConfig(net="fta", bound=u, tiles=k, norm=norm, label=f"DQN-FTA{tag} (u={u:g}, k={k})", **kw)


def paper():
    """Experiment name -> list of configs (one per agent); each is run for several seeds."""
    exp = {}
    # Fig. 1: reproducibility, u = 20, delta = eta = 2 -> k = 20; with and without a target network.
    exp["reproduce"] = [c for tn in (False, True) for c in (
        DQNConfig(env=LL, net="relu", target_network=tn, label=f"DQN{' + target' if tn else ''}"),
        DQNConfig(env=LL, net="large", tiles=20, target_network=tn, label=f"DQN-Large{' + target' if tn else ''}"),
        _fta(20, 20, env=LL, target_network=tn) if not tn else DQNConfig(env=LL, net="fta", bound=20, tiles=20, target_network=True, label="DQN-FTA (u=20, k=20) + target"),
    )]
    # Fig. 2: tiling-bound sweep, u in {0.1, 1, 10, 50, 100}, k in {16, 64, 128}.
    exp["bound_sweep"] = [_fta(u, k, env=LL) for k in (16, 64, 128) for u in (0.1, 1, 10, 50, 100)]
    # Fig. 3: best FTA (u = 1, k = 64) against untuned FTA, DQN and DQN-Large.
    exp["best_fta"] = [_fta(1, 64, env=LL), _fta(20, 20, env=LL), DQNConfig(env=LL, net="relu", label="DQN"),
                       DQNConfig(env=LL, net="large", tiles=20, label="DQN-Large")]
    # Fig. 4: normalizing before FTA, with the best (u = 1) and worst (u = 0.01, 100) bounds.
    exp["normalize"] = [_fta(u, 20, norm=n, env=LL) for n in ("none", "tanh", "batchnorm", "rangenorm") for u in (0.01, 1, 100)]
    # Fig. 5: CartPole, u in {0.01, 1, 100}, k = 20, no target network, with and without tanh.
    exp["cartpole"] = [_fta(u, 20, norm=n, env=CP) for n in ("none", "tanh") for u in (0.01, 1, 100)]
    # Fig. 6: distribution of returns, 50 runs each of DQN, untuned DQN-FTA and DQN-FTA-tanh, k = 20.
    exp["distribution"] = [DQNConfig(env=LL, net="relu", label="DQN"), _fta(20, 20, env=LL), _fta(20, 20, norm="tanh", env=LL)]
    return exp


PAPER_SEEDS = {"distribution": 50}      # others: 10
PAPER_STEPS = 500_000


def quick():
    """Reduced-scale version run for this repository: fewer steps and seeds, key agents only."""
    ll = dict(env=LL, steps=100_000, eval_every=5_000)
    cp = dict(env=CP, steps=50_000, eval_every=2_500)
    return {
        "lunarlander": [DQNConfig(net="relu", label="DQN", **ll), DQNConfig(net="large", tiles=20, label="DQN-Large", **ll)]
                       + [_fta(u, 20, **ll) for u in (0.01, 1, 20, 100)]
                       + [_fta(u, 20, norm="tanh", **ll) for u in (0.01, 1, 100)],
        "cartpole": [_fta(u, 20, norm=n, **cp) for n in ("none", "tanh") for u in (0.01, 1, 100)],
    }


QUICK_SEEDS = {"lunarlander": 3, "cartpole": 5}
