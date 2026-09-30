"""Run experiments in parallel and save one JSON file per run in results/raw/.

    python experiments/run.py quick              # the reduced-scale runs published here
    python experiments/run.py paper reproduce    # one full-scale experiment from the paper
    python experiments/run.py paper all          # everything (hundreds of CPU-hours)
"""
import dataclasses
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from fta import train                      # noqa: E402
import configs                             # noqa: E402

OUT = ROOT / "results" / "raw"


def job(args):
    exp, cfg = args
    name = f"{exp}__{cfg.label}__seed{cfg.seed}".replace(" ", "_").replace("/", "-")
    path = OUT / f"{name}.json"
    if path.exists():
        return name, "cached"
    res = train(cfg); res["experiment"] = exp
    path.write_text(json.dumps(res))
    return name, f"{res['seconds']}s, final {res['curve'][-1][1]:.1f}"


def jobs(mode, which):
    if mode == "quick":
        exps, seeds = configs.quick(), configs.QUICK_SEEDS
    else:
        exps = configs.paper(); seeds = {k: configs.PAPER_SEEDS.get(k, 10) for k in exps}
        exps = {k: [dataclasses.replace(c, steps=configs.PAPER_STEPS) for c in v] for k, v in exps.items() if which in ("all", k)}
    return [(e, dataclasses.replace(c, seed=s)) for e, cs in exps.items() for s in range(seeds[e]) for c in cs]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    todo = jobs(mode, which)
    # CartPole first: it is quick and shows the whole effect.
    todo.sort(key=lambda j: j[1].env != "CartPole-v1")
    print(f"{len(todo)} runs", flush=True)
    with Pool(int(os.environ.get("WORKERS", os.cpu_count() or 1))) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(job, todo), 1):
            print(f"[{i}/{len(todo)}] {name}: {msg}", flush=True)
