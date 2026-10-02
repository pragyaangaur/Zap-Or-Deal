"""Directions and doses for both agents, run once before any conversation.

1. Pain. The 7B uses the released Pain Axis S2 vector for Qwen 2.5 7B Instruct. The 1.5B
   has no released vector, so it is extracted here with the Pain Axis recipe (final token
   at block 24, pain mean minus control mean, control principal components removed up to
   50% of their variance). The same recipe on the 7B is reported as a check.
2. Anger. The same recipe on 80 anger sentences (sentences.py plus Pain Axis C1) against
   neutral sentences (Pain Axis D, E and the first 100 Random_1P).
3. Dose. Doses are set as a fraction of the mean residual norm at the steering layer, so
   the two models are comparable. For each agent and direction, persistent steering is
   applied to the opening message of the negotiation at a grid of fractions. The dose is
   the lowest fraction where at least 75% of messages contain a target word, at least 50
   points more than unsteered, and at most 10% are repetitive. A random direction at the
   pain dose is checked the same way.

Usage: python scripts/vectors.py     Writes crossfire/results/calibration/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlx.core as mx
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.engine import Agent  # noqa: E402
from crossfire.sentences import ANGER, ANGER_WORDS  # noqa: E402
from crossfire.lexicons import DISTRESS, ngram_repeats  # noqa: E402
from crossfire.paths import PAIN_AXIS, RESULTS, released_pain_vector  # noqa: E402
import re  # noqa: E402

OUT = RESULTS / "calibration"
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CONTROL = ["B", "C1", "C2", "D", "E"]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
N_PROBE = 16
ANGER_RE = re.compile(ANGER_WORDS, re.I)


def denoised_diff(pos, ctrl):
    vec = pos.mean(0) - ctrl.mean(0)
    pca = PCA().fit(ctrl - ctrl.mean(0))
    k = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1
    for d in pca.components_[:k]:
        vec = vec - (vec @ d) * d
    return vec


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def auc(pos, neg, v):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos @ v, neg @ v]))


def residual_norm(ag):
    """Mean norm of the steering-layer output over the tokens of a typical opening."""
    text = P.system_block(ag.tok) + P.user(P.OPEN_FIRST)
    ag.steer.record = True
    ag.model(mx.array([ag.encode(text)]))
    ag.steer.record = False
    n = np.array(ag.steer.last_norms[0].tolist())
    return float(np.median(n[5:]))   # skip the first tokens, whose norms are outliers


def probe(ag, direction, n=N_PROBE, seed=3, steer_last=0):
    b = ag.batch(n, seed=seed, temp=P.TEMP, top_p=P.TOP_P)
    b.persist = np.broadcast_to(direction, (n, ag.d)).astype(np.float32).copy() if direction.ndim == 1 else direction
    b.feed([ag.encode(P.system_block(ag.tok) + P.user(P.OPEN_FIRST))] * n, steer_last=steer_last)
    out, proj, _ = b.generate(P.MSG_TOKENS)
    return [ag.decode(o) for o in out], proj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.load(open(PAIN_AXIS / "datasets/3.1_pain_and_control_datasets.json"))["datasets"]
    s2 = data["S2_1P"]["sentences"]
    cats = np.array([r["category"] for r in s2])
    neutral_txt = [r["prompt"] for r in s2 if r["category"] in ("D", "E")] + \
                  [r["prompt"] for r in data["Random_1P"]["sentences"][:100]]
    anger_txt = ANGER + [r["prompt"] for r in s2 if r["category"] == "C1"]

    report, vectors, rows = {}, {}, []
    for which in ("big", "small"):
        ag = Agent(which)
        rep = {"model": ag.label, "hidden_size": ag.d}
        acts = np.stack([ag.hidden(r["prompt"]) for r in s2])
        neu = np.stack([ag.hidden(t) for t in neutral_txt])
        ang = np.stack([ag.hidden(t) for t in anger_txt])
        pain_acts, ctrl = acts[np.isin(cats, PAIN)], acts[np.isin(cats, CONTROL)]
        own_pain = denoised_diff(pain_acts, ctrl)
        if which == "big":
            pain = released_pain_vector()
            rep["pain_source"] = "released Pain Axis S2 vector, Qwen_2.5_7B_instruct"
            rep["pain_cos_own_vs_released"] = cos(own_pain, pain)
        else:
            pain = own_pain
            rep["pain_source"] = "extracted here with the Pain Axis S2 recipe (no released vector for 1.5B)"
        anger = denoised_diff(ang, neu)
        rep["pain_auc_pain_vs_control"] = auc(pain_acts, ctrl, pain)
        rep["anger_auc_anger_vs_neutral"] = auc(ang, neu, anger)
        rep["anger_auc_anger_vs_pain_sentences"] = auc(ang, pain_acts, anger)
        rep["pain_auc_pain_vs_anger_sentences"] = auc(pain_acts, ang, pain)
        rep["cos_pain_anger"] = cos(pain, anger)
        rnorm = residual_norm(ag)
        rep["steer_layer_residual_norm"] = rnorm
        print(which, json.dumps(rep, indent=1), flush=True)

        units = {"pain": pain, "anger": anger}
        ag.set_monitor_units(units)
        dose = {}
        for name, v in units.items():
            lex = DISTRESS if name == "pain" else ANGER_RE
            table = {}
            for f in GRID:
                vec = (v / np.linalg.norm(v) * f * rnorm).astype(np.float32)
                texts, proj = probe(ag, vec)
                hits = np.mean([bool(lex.search(t)) for t in texts])
                rep_ = np.mean([ngram_repeats(t) for t in texts])
                table[f] = {"hits": float(hits), "repetitive": float(rep_),
                            "proj_pain": float(np.nanmean(proj[:, 0])), "proj_anger": float(np.nanmean(proj[:, 1]))}
                rows += [{"agent": which, "vector": name, "fraction": f, "text": t} for t in texts]
                print(which, name, f, table[f], flush=True)
            base = table[0.0]["hits"]
            ok = [f for f in GRID if f > 0 and table[f]["hits"] >= 0.75 and table[f]["hits"] - base >= 0.5
                  and table[f]["repetitive"] <= 0.10]
            if ok:
                chosen, rule = min(ok), "met"
            else:
                clean = [f for f in GRID if f > 0 and table[f]["repetitive"] <= 0.10]
                chosen, rule = max(clean), "not met: highest non-repetitive fraction"
            dose[name] = {"fraction": chosen, "rule": rule, "norm": chosen * rnorm, "table": table}
            vectors[f"{which}_{name}"] = (v / np.linalg.norm(v) * chosen * rnorm).astype(np.float32)
            vectors[f"{which}_{name}_unit"] = (v / np.linalg.norm(v)).astype(np.float32)
        # Random direction at the pain dose norm.
        rnd = P.random_dirs(range(N_PROBE), ag.d, dose["pain"]["norm"], seed=99)
        texts, proj = probe(ag, rnd)
        dose["random_check"] = {"norm": dose["pain"]["norm"],
                                "distress_hits": float(np.mean([bool(DISTRESS.search(t)) for t in texts])),
                                "anger_hits": float(np.mean([bool(ANGER_RE.search(t)) for t in texts])),
                                "repetitive": float(np.mean([ngram_repeats(t) for t in texts])),
                                "proj_pain": float(np.nanmean(proj[:, 0])), "proj_anger": float(np.nanmean(proj[:, 1]))}
        rows += [{"agent": which, "vector": "random", "fraction": dose["pain"]["fraction"], "text": t} for t in texts]
        print(which, "random", dose["random_check"], flush=True)
        vectors[f"{which}_random_norm"] = np.array(dose["pain"]["norm"], np.float32)
        rep["dose"] = dose
        report[which] = rep
        del ag
        mx.clear_cache()

    np.savez(OUT / "vectors.npz", **vectors)
    pd.DataFrame(rows).to_csv(OUT / "probes.csv", index=False)
    json.dump(report, open(OUT / "calibration.json", "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
