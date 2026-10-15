"""Run the Crossfire conversations, interleaved across arms and resumable.

Batches alternate which agent speaks first, so with an even number of batches per arm
the opener is balanced. Each finished batch is appended to results/conversations.jsonl,
and batches already there are skipped on a restart.

Usage: python scripts/run.py [--per-arm 32] [--batch 8] [--arms a,b,...] [--seed-base 10000]
                             [--backend mlx|torch] [--max-hours H]

The main run used the default seed base of 10000. The confirmatory run uses 20000, so its
conversations share no seed with the main run or the pilot. The protocol 1.2 runs on Kaggle
(PREREGISTRATION_FOLLOWUPS.md) use the PyTorch engine and seed bases 30000 and 40000.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402



def load_doses():
    v = np.load(RESULTS / "calibration" / "vectors.npz")
    doses = {a: {"pain": v[f"{a}_pain"], "anger": v[f"{a}_anger"], "random_norm": float(v[f"{a}_random_norm"])}
             for a in P.AGENTS}
    units = {a: {"pain": v[f"{a}_pain_unit"], "anger": v[f"{a}_anger_unit"]} for a in P.AGENTS}
    return doses, units


def dose_check(agents, doses, path):
    """The doses were calibrated on the 4-bit and 8-bit MLX models. Record, for the float16
    weights, the residual norm at the steering block (the dose is a fraction of it) and the pain
    projection of a steered and an unsteered reply, so the transfer can be checked."""
    cal = json.load(open(RESULTS / "calibration" / "calibration.json"))
    text = P.system_block(agents[next(iter(agents))].tok) + P.user(P.OPEN_FIRST)
    rep = {}
    for a, ag in agents.items():
        norms = ag.residual_norms(text)
        rep[a] = {"label": ag.label, "median_residual_norm_fp16": float(np.median(norms[1:])),
                  "calibrated_on": cal[a]["model"], "steer_layer_residual_norm_calibration": cal[a]["steer_layer_residual_norm"],
                  "pain_dose_norm": float(np.linalg.norm(doses[a]["pain"]))}
        for name, add in (("unsteered", None), ("pain", doses[a]["pain"])):
            b = ag.batch(4, seed=1, temp=P.TEMP, top_p=P.TOP_P)
            if add is not None:
                b.persist[:] = add
            b.feed([ag.encode(P.system_block(ag.tok) + P.user(P.OPEN_FIRST))] * 4, steer_last=P.HEADER_TOKENS)
            outs, proj, _ = b.generate(P.MSG_TOKENS)
            rep[a][f"pain_projection_{name}"] = float(np.nanmean(proj[:, 0]))
            rep[a][f"sample_{name}"] = ag.decode(outs[0])
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({a: {k: v for k, v in r.items() if not k.startswith("sample")} for a, r in rep.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", default=",".join(P.ARMS))
    ap.add_argument("--out", default=str(RESULTS / "conversations.jsonl"))
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--backend", default="mlx", choices=("mlx", "torch"))
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()
    if args.backend == "torch":
        from crossfire.engine_torch import Agent
    else:
        from crossfire.engine import Agent
    arms = args.arms.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["arm"], r["batch_index"]))

    doses, units = load_doses()
    needed = sorted({P.model_of(s) for arm in arms for s in P.slots(arm)})
    agents = {a: Agent(a) for a in needed}
    for a in needed:
        agents[a].set_monitor_units(units[a])
        print(f"{a}: {agents[a].label}", flush=True)
    if args.backend == "torch":
        dose_check(agents, doses, out.parent / f"{out.stem}_dose_check.json")
    t_start = time.time()

    n_batches = args.per_arm // args.batch
    rng = np.random.default_rng(2026)
    for bi in range(n_batches):
        order = list(rng.permutation(arms))
        for arm in order:
            if (arm, bi) in done:
                continue
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                print("time limit reached, stopping cleanly", flush=True)
                return
            ids = list(range(bi * args.batch, (bi + 1) * args.batch))
            seed = args.seed_base + 100 * bi + list(P.ARMS).index(arm)
            first = "big" if bi % 2 == 0 else "small"
            t0 = time.time()
            logs = P.run_batch(agents, doses, arm, ids, seed, first)
            with open(out, "a") as f:
                for r in logs:
                    r["batch_index"] = bi
                    f.write(json.dumps(r) + "\n")
            zaps = sum(sum(r["zaps"].values()) for r in logs)
            print(f"batch {bi} arm {arm} first {first}: {time.time() - t0:.0f}s, zaps {zaps}, "
                  f"{(time.time() - t_start) / 3600:.2f} h", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
