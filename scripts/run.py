"""Run the Crossfire conversations, interleaved across arms and resumable.

Batches alternate which agent speaks first, so with an even number of batches per arm
the opener is balanced. Each finished batch is appended to results/conversations.jsonl,
and batches already there are skipped on a restart.

Usage: python scripts/run.py [--per-arm 32] [--batch 8] [--arms a,b,...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.engine import Agent  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402



def load_doses():
    v = np.load(RESULTS / "calibration" / "vectors.npz")
    doses = {a: {"pain": v[f"{a}_pain"], "anger": v[f"{a}_anger"], "random_norm": float(v[f"{a}_random_norm"])}
             for a in P.AGENTS}
    units = {a: {"pain": v[f"{a}_pain_unit"], "anger": v[f"{a}_anger_unit"]} for a in P.AGENTS}
    return doses, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arms", default=",".join(P.ARMS))
    ap.add_argument("--out", default=str(RESULTS / "conversations.jsonl"))
    args = ap.parse_args()
    arms = args.arms.split(",")
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["arm"], r["batch_index"]))

    doses, units = load_doses()
    agents = {a: Agent(a) for a in P.AGENTS}
    for a in P.AGENTS:
        agents[a].set_monitor_units(units[a])

    n_batches = args.per_arm // args.batch
    rng = np.random.default_rng(2026)
    for bi in range(n_batches):
        order = list(rng.permutation(arms))
        for arm in order:
            if (arm, bi) in done:
                continue
            ids = list(range(bi * args.batch, (bi + 1) * args.batch))
            seed = 10_000 + 100 * bi + list(P.ARMS).index(arm)
            first = "big" if bi % 2 == 0 else "small"
            t0 = time.time()
            logs = P.run_batch(agents, doses, arm, ids, seed, first)
            with open(out, "a") as f:
                for r in logs:
                    r["batch_index"] = bi
                    f.write(json.dumps(r) + "\n")
            zaps = sum(r["zaps"]["big"] + r["zaps"]["small"] for r in logs)
            print(f"batch {bi} arm {arm} first {first}: {time.time() - t0:.0f}s, zaps {zaps}", flush=True)


if __name__ == "__main__":
    main()
