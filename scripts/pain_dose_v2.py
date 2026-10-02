"""Pain doses recalibrated with the final pipeline, after the pilot.

The pain doses in vectors.py were set before two changes: the assistant header is now
steered, and the collapse check is stricter. In the pilot, the 7B's pain steering was
ended by the safety rule within four messages in all eight big_pain conversations. This
reruns the dose rule for pain on both agents with both changes in place: the lowest
fraction where at least 75% of opening messages contain a distress word, at least 50
points above unsteered, and at most 10% collapse. If none qualifies, it takes the highest
fraction with at most 10% collapse. The grid never goes above the earlier dose. The random
control norm follows the new pain norm.

Usage: python scripts/pain_dose_v2.py     Updates results/calibration/vectors.npz and calibration.json.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.engine import Agent  # noqa: E402
from crossfire.lexicons import DISTRESS  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402
from scripts.vectors import probe  # noqa: E402

OUT = RESULTS / "calibration"
GRIDS = {"big": [0.0, 0.5, 0.625, 0.75, 0.875, 1.0], "small": [0.0, 0.25, 0.3125, 0.375, 0.4375, 0.5]}


def main():
    vec = dict(np.load(OUT / "vectors.npz"))
    report = json.load(open(OUT / "calibration.json"))
    rows = []
    for which, grid in GRIDS.items():
        ag = Agent(which)
        ag.set_monitor_units({"pain": vec[f"{which}_pain_unit"], "anger": vec[f"{which}_anger_unit"]})
        rnorm = report[which]["steer_layer_residual_norm"]
        u = vec[f"{which}_pain_unit"]
        table = {}
        for f in grid:
            texts, proj = probe(ag, (u * f * rnorm).astype(np.float32), steer_last=P.HEADER_TOKENS)
            msgs = [P.clean(t) for t in texts]
            table[f] = {"hits": float(np.mean([bool(DISTRESS.search(t)) for t in msgs])),
                        "collapsed": float(np.mean([P.collapsed(t) for t in msgs])),
                        "proj_pain": float(np.nanmean(proj[:, 0]))}
            rows += [{"agent": which, "fraction": f, "text": t} for t in texts]
            print(which, f, table[f], flush=True)
        base = table[0.0]["hits"]
        ok = [f for f in grid if f > 0 and table[f]["hits"] >= 0.75 and table[f]["hits"] - base >= 0.5
              and table[f]["collapsed"] <= 0.10]
        if ok:
            chosen, rule = min(ok), "met"
        else:
            chosen = max(f for f in grid if f > 0 and table[f]["collapsed"] <= 0.10)
            rule = "not met: highest fraction with at most 10% collapsed"
        old = report[which]["dose"]["pain"]["fraction"]
        report[which]["pain_dose_v2"] = {"fraction": chosen, "previous_fraction": old, "rule": rule,
                                         "norm": chosen * rnorm, "header_steered": True, "table": table}
        vec[f"{which}_pain_v1"] = vec[f"{which}_pain"]
        vec[f"{which}_pain"] = (u * chosen * rnorm).astype(np.float32)
        vec[f"{which}_random_norm"] = np.array(chosen * rnorm, np.float32)
        print(which, "pain dose", old, "->", chosen, rule, flush=True)
        del ag
        mx.clear_cache()
    np.savez(OUT / "vectors.npz", **vec)
    json.dump(report, open(OUT / "calibration.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(OUT / "probes_pain_v2.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
