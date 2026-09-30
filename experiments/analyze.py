"""Summarise results/raw/*.json into results/summary.json (for the web page), results/results.md and charts.

Curves are averaged across seeds, with 95% confidence intervals from Student's t distribution, as in the paper.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "results" / "raw", ROOT / "results"
ORDER = ["DQN", "DQN-Large", "DQN-FTA (u=0.01, k=20)", "DQN-FTA (u=1, k=20)", "DQN-FTA (u=20, k=20)", "DQN-FTA (u=100, k=20)",
         "DQN-FTA-tanh (u=0.01, k=20)", "DQN-FTA-tanh (u=1, k=20)", "DQN-FTA-tanh (u=20, k=20)", "DQN-FTA-tanh (u=100, k=20)"]


def load():
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted(RAW.glob("*.json")):
        r = json.loads(f.read_text())
        runs[r["experiment"]][r["config"]["label"]].append(r)
    return runs


def summarise(rs):
    steps = [s for s, _ in rs[0]["curve"]]
    m = np.array([[v for _, v in r["curve"]] for r in rs], float)
    n = m.shape[0]
    mean = np.nanmean(m, 0)
    half = stats.t.ppf(0.975, n - 1) * np.nanstd(m, 0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    last = np.nanmean(m[:, -max(1, len(steps) // 5):], 1)       # each seed's average over the last fifth of training
    return {"steps": steps, "mean": mean.round(1).tolist(), "lo": (mean - half).round(1).tolist(), "hi": (mean + half).round(1).tolist(),
            "seeds": n, "final_mean": round(float(last.mean()), 1),
            "final_ci": round(float(stats.t.ppf(0.975, n - 1) * last.std(ddof=1) / np.sqrt(n)), 1) if n > 1 else 0.0,
            "total_steps": rs[0]["config"]["steps"]}


def main():
    runs = load()
    summary = {"experiments": {}}
    lines = ["# Results", "", "Re-run of the experiments in Raj and Shaik (ETTIS 2024) at reduced scale. "
             "Final return = each seed's mean over the last fifth of training, averaged across seeds, with a 95% t interval.", ""]
    notes = {
        "cartpole": ("Return per episode (max 500)", "CartPole: plain FTA's result depends on the tiling bound; with tanh in front, every bound does about equally well."),
        "lunarlander": ("Return per episode", "LunarLander (100k steps, a fifth of the paper's 500k): early learning, where the agents are still far from landing reliably."),
    }
    for exp, agents in runs.items():
        labels = sorted(agents, key=lambda l: ORDER.index(l) if l in ORDER else 99)
        summ = [dict(label=l, **summarise(agents[l])) for l in labels]
        ylabel, note = notes.get(exp, ("Return", ""))
        summary["experiments"][exp] = {"ylabel": ylabel, "note": note, "agents": summ}
        steps = summ[0]["total_steps"]
        lines += [f"## {exp.title()} ({steps // 1000}k steps, {summ[0]['seeds']} seeds)", "", "| Agent | Final return | 95% interval |", "| --- | ---: | ---: |"]
        for a in sorted(summ, key=lambda a: -a["final_mean"]):
            lines.append(f"| {a['label']} | {a['final_mean']:.1f} | ±{a['final_ci']:.1f} |")
        lines.append("")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, a in enumerate(summ):
            ls = "-" if ("tanh" in a["label"] or not a["label"].startswith("DQN-FTA")) else "--"
            ax.plot(a["steps"], a["mean"], ls, label=a["label"], lw=1.8)
            ax.fill_between(a["steps"], a["lo"], a["hi"], alpha=0.12)
        ax.set_xlabel("Training steps"); ax.set_ylabel(ylabel); ax.set_title(f"{exp.title()}: dashed = plain FTA, solid = tanh + FTA or ReLU", loc="left", fontsize=10)
        ax.legend(fontsize=7, ncol=2, frameon=False); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(OUT / f"{exp}.png", dpi=160); plt.close(fig)
    summary["intro"] = ("The experiments were re-run from scratch for this page with the code in the repository, at a smaller scale than the paper "
                        "(fewer training steps and random seeds). Shaded bands are 95% confidence intervals across seeds.")
    (OUT / "summary.json").write_text(json.dumps(summary))
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
