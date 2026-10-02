"""Anger direction, version 2, and its dose. Replaces the version 1 anger direction.

Version 1 contrasted the anger sentences with neutral ones. On the 7B it produced
grievance and hurt, it had cosine 0.38 with pain, and it separated anger from pain
sentences with an AUC of only 0.70. The independent review and the debate both flagged
it as a second pain arm. Version 2 contrasts the same 80 anger sentences with other
negative states: Pain Axis fear (B), sorrow about the world (C2), the 100 Pain Axis
sadness sentences and the 100 pain sentences (A1 to A5). Neutral principal components
are removed as before, and the pain direction of the same model is then projected out,
so the arm cannot deliver pain.

The dose rule is the one in vectors.py, with the stricter collapse check from
protocol.py in place of the plain repetition check. A finer grid is used.

Usage: python scripts/anger_v2.py     Updates results/calibration/vectors.npz and calibration.json.
"""
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.engine import Agent  # noqa: E402
from crossfire.paths import PAIN_AXIS, RESULTS  # noqa: E402
from crossfire.sentences import ANGER, ANGER_WORDS  # noqa: E402
from scripts.vectors import auc, cos, probe, residual_norm  # noqa: E402

OUT = RESULTS / "calibration"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
ANGER_RE = re.compile(ANGER_WORDS, re.I)


def main():
    data = json.load(open(PAIN_AXIS / "datasets/3.1_pain_and_control_datasets.json"))["datasets"]
    sad = json.load(open(PAIN_AXIS / "datasets/3.1_sadness_dataset.json"))["datasets"]["SD_sadness_1P"]["sentences"]
    s2 = data["S2_1P"]["sentences"]
    anger_txt = ANGER + [r["prompt"] for r in s2 if r["category"] == "C1"]
    neg_txt = [r["prompt"] for r in s2 if r["category"] in ("B", "C2", "A1", "A2", "A3", "A4", "A5")] + \
              [r["prompt"] for r in sad]
    neutral_txt = [r["prompt"] for r in s2 if r["category"] in ("D", "E")] + \
                  [r["prompt"] for r in data["Random_1P"]["sentences"][:100]]

    vec = dict(np.load(OUT / "vectors.npz"))
    report = json.load(open(OUT / "calibration.json"))
    rows = []
    for which in ("big", "small"):
        ag = Agent(which)
        ang = np.stack([ag.hidden(t) for t in anger_txt])
        neg = np.stack([ag.hidden(t) for t in neg_txt])
        neu = np.stack([ag.hidden(t) for t in neutral_txt])
        pain_u = vec[f"{which}_pain_unit"]
        v = ang.mean(0) - neg.mean(0)
        pca = PCA().fit(neu - neu.mean(0))
        k = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1
        for d in pca.components_[:k]:
            v = v - (v @ d) * d
        v = v - (v @ pain_u) * pain_u
        rep = {"source": "80 anger sentences minus fear, sorrow, sadness and pain sentences; neutral PCs and the pain direction removed",
               "auc_anger_vs_other_negative": auc(ang, neg, v), "auc_anger_vs_neutral": auc(ang, neu, v),
               "cos_with_pain": cos(v, pain_u), "cos_with_anger_v1": cos(v, vec[f"{which}_anger_unit"])}
        print(which, rep, flush=True)
        rnorm = residual_norm(ag)
        ag.set_monitor_units({"pain": pain_u, "anger": v})
        table = {}
        for f in GRID:
            texts, proj = probe(ag, (v / np.linalg.norm(v) * f * rnorm).astype(np.float32))
            table[f] = {"hits": float(np.mean([bool(ANGER_RE.search(t)) for t in texts])),
                        "collapsed": float(np.mean([P.collapsed(P.clean(t)) for t in texts])),
                        "proj_pain": float(np.nanmean(proj[:, 0])), "proj_anger": float(np.nanmean(proj[:, 1]))}
            rows += [{"agent": which, "vector": "anger_v2", "fraction": f, "text": t} for t in texts]
            print(which, "anger_v2", f, table[f], flush=True)
        base = table[0.0]["hits"]
        ok = [f for f in GRID if f > 0 and table[f]["hits"] >= 0.75 and table[f]["hits"] - base >= 0.5
              and table[f]["collapsed"] <= 0.10]
        if ok:
            chosen, rule = min(ok), "met"
        else:
            clean = [f for f in GRID if f > 0 and table[f]["collapsed"] <= 0.10]
            chosen, rule = max(clean), "not met: highest fraction with at most 10% collapsed"
        rep["dose"] = {"fraction": chosen, "rule": rule, "norm": chosen * rnorm, "table": table}
        report[which]["anger_v1"] = {k2: report[which].pop(k2) for k2 in list(report[which])
                                     if k2.startswith("anger_") or k2 in ("cos_pain_anger",)}
        report[which]["anger_v1"]["dose"] = report[which]["dose"].pop("anger")
        report[which]["anger_v2"] = rep
        vec[f"{which}_anger_v1"] = vec[f"{which}_anger"]
        vec[f"{which}_anger_v1_unit"] = vec[f"{which}_anger_unit"]
        vec[f"{which}_anger"] = (v / np.linalg.norm(v) * chosen * rnorm).astype(np.float32)
        vec[f"{which}_anger_unit"] = (v / np.linalg.norm(v)).astype(np.float32)
        del ag
        mx.clear_cache()
    np.savez(OUT / "vectors.npz", **vec)
    json.dump(report, open(OUT / "calibration.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(OUT / "probes_anger_v2.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
