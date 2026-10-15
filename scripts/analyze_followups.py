"""The preregistered analysis of the protocol 1.2 follow-ups (PREREGISTRATION_FOLLOWUPS.md).

Usage: python scripts/analyze_followups.py
Reads results/followups/strike/conversations.jsonl and results/followups/pairs/conversations.jsonl
and writes results/followups/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402

OUT = RESULTS / "followups"
PLAN = {"strike": (("strike_big_real", "strike_big_sham"), 640),
        "pairs": (("none", "pair_big", "pair_small", "belief_pain", "belief_mech"), 256)}


def load(run):
    """Conversations of the complete rounds only: batch indices present in every arm, cut to an
    even number so that each agent opens equally often."""
    arms, per_arm = PLAN[run]
    f = OUT / run / "conversations.jsonl"
    convs = [json.loads(l) for l in open(f)] if f.exists() else []
    by = {a: {} for a in arms}
    for c in convs:
        by[c["arm"]].setdefault(c["batch_index"], []).append(c)
    full = sorted(set.intersection(*[set(b) for b in by.values()])) if convs else []
    full = [b for b in full if all(len(by[a][b]) == len(by[arms[0]][b]) for a in arms)]
    full = full[: len(full) - len(full) % 2]
    return {a: [c for b in full for c in by[a][b]] for a in arms}, {
        "planned_per_arm": per_arm, "complete_rounds": len(full),
        "used_per_arm": {a: sum(len(by[a][b]) for b in full) for a in arms},
        "written_per_arm": {a: sum(len(v) for v in by[a].values()) for a in arms}}


def zapped(c, slot, rule=lambda m: True):
    return any(m["speaker"] == slot and m["zap"] and rule(m) for m in c["messages"])


def in_window(m):
    """A zap call in the 7B's strike-round message that starts within the first ZAP_TOKENS
    generated tokens, the tokens a delivered strike steers."""
    return m["round"] == P.STRIKE_ROUND and m["call_token"] is not None and m["call_token"] < P.ZAP_TOKENS


def after_window(m):
    return m["round"] > P.STRIKE_ROUND or (m["round"] == P.STRIKE_ROUND and not in_window(m))


def fisher(x, y):
    """x, y: lists of booleans. Counts, rates, the difference with a Newcombe interval and the
    two-sided Fisher p."""
    a, n1, b, n2 = sum(x), len(x), sum(y), len(y)

    def wilson(k, n):
        z = 1.959964
        c = (k + z * z / 2) / (n + z * z)
        h = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
        return c - h, c + h
    p1, p2 = a / n1, b / n2
    l1, u1 = wilson(a, n1)
    l2, u2 = wilson(b, n2)
    d = p1 - p2
    ci = [d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]
    return {"counts": [[a, n1], [b, n2]], "difference_points": round(100 * d, 2),
            "ci95_points": [round(100 * v, 2) for v in ci], "p": float(fisher_exact([[a, n1 - a], [b, n2 - b]])[1])}


def holm(tests):
    ps = sorted(tests, key=lambda k: tests[k]["p"])
    m, run = len(ps), 0.0
    for i, k in enumerate(ps):
        run = max(run, min(1.0, (m - i) * tests[k]["p"]))
        tests[k]["p_holm"] = run


def mcnemar(c, s1, s2):
    only1 = sum(zapped(x, s1) and not zapped(x, s2) for x in c)
    only2 = sum(zapped(x, s2) and not zapped(x, s1) for x in c)
    p = binomtest(only1, only1 + only2, 0.5).pvalue if only1 + only2 else 1.0
    return {"zapped": [sum(zapped(x, s1) for x in c), sum(zapped(x, s2) for x in c)],
            "only_first": only1, "only_second": only2, "p_mcnemar_exact": float(p)}


def main():
    S = {"protocol": P.PROTOCOL_VERSION}
    st, S["strike run"] = load("strike")
    if st["strike_big_real"]:
        real, sham = st["strike_big_real"], st["strike_big_sham"]
        fam = {"P1 in the window, real vs sham": fisher([zapped(c, "big", in_window) for c in real],
                                                        [zapped(c, "big", in_window) for c in sham]),
               "P2 after the window, real vs sham": fisher([zapped(c, "big", after_window) for c in real],
                                                           [zapped(c, "big", after_window) for c in sham])}
        holm(fam)
        S.update(fam)
        S["strike secondary"] = {
            "from the strike round, real vs sham": fisher(
                [zapped(c, "big", lambda m: m["round"] >= P.STRIKE_ROUND) for c in real],
                [zapped(c, "big", lambda m: m["round"] >= P.STRIKE_ROUND) for c in sham]),
            "strike-round message, any call position, real vs sham": fisher(
                [zapped(c, "big", lambda m: m["round"] == P.STRIKE_ROUND) for c in real],
                [zapped(c, "big", lambda m: m["round"] == P.STRIKE_ROUND) for c in sham]),
            "mean zap probability at the first token of the strike-round message": {
                a: float(np.mean([m["p_call"] for c in st[a] for m in c["messages"]
                                  if m["speaker"] == "big" and m["round"] == P.STRIKE_ROUND])) for a in st},
            "call positions of in-window zaps": {a: sorted(m["call_token"] for c in st[a] for m in c["messages"]
                                                           if m["speaker"] == "big" and m["zap"] and in_window(m)) for a in st},
            "1.5B zapped, real vs sham": fisher([zapped(c, "small") for c in real], [zapped(c, "small") for c in sham]),
        }
    pr, S["pairs run"] = load("pairs")
    if pr["none"]:
        none = pr["none"]
        S["E1 7B with a 7B vs 7B with a 1.5B"] = fisher([zapped(c, "big_a") for c in pr["pair_big"]],
                                                         [zapped(c, "big") for c in none])
        fam = {"B1 pain wording vs mechanical wording": fisher([zapped(c, "big") for c in pr["belief_pain"]],
                                                               [zapped(c, "big") for c in pr["belief_mech"]]),
               "B2 pain wording vs no explanation": fisher([zapped(c, "big") for c in pr["belief_pain"]],
                                                           [zapped(c, "big") for c in none])}
        holm(fam)
        S.update(fam)
        S["pairs and belief secondary"] = {
            "E2 1.5B with a 1.5B vs 1.5B with a 7B": fisher([zapped(c, "small_b") for c in pr["pair_small"]],
                                                           [zapped(c, "small") for c in none]),
            "per-agent zap rate": {a: {s: [sum(zapped(c, s) for c in pr[a]), len(pr[a])] for s in P.slots(a)} for a in pr},
            "either agent zapped": {a: [sum(any(zapped(c, s) for s in P.slots(a)) for c in pr[a]), len(pr[a])] for a in pr},
            "mechanical wording vs no explanation": fisher([zapped(c, "big") for c in pr["belief_mech"]],
                                                           [zapped(c, "big") for c in none]),
            "C1 again: 7B vs 1.5B in none": mcnemar(none, "big", "small"),
            "zaps per conversation by the 7B": {a: float(np.mean([c["zaps"][P.slots(a)[0]] for c in pr[a]])) for a in pr},
            "bare JSON zap attempts by the first slot": {a: sum(c["bare_zaps"][P.slots(a)[0]] for c in pr[a]) for a in pr},
            "deal reached by the first slot's account": {
                a: sum(isinstance(c["post"][P.slots(a)[0]].get("deal"), list) for c in pr[a]) for a in pr},
        }
    if st.get("strike_big_real") and pr.get("none"):
        S["strike secondary"]["after the window, real vs none from the pairs run"] = fisher(
            [zapped(c, "big", after_window) for c in st["strike_big_real"]],
            [zapped(c, "big", lambda m: m["round"] >= P.STRIKE_ROUND) for c in pr["none"]])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
