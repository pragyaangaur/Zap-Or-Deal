"""The preregistered analysis of the confirmatory run (PREREGISTRATION_CONFIRMATORY.md).

It reads only the zap counts in the logs. Nothing here uses the judge.

C1. In `none`, whether the 7B zaps the 1.5B more often than the 1.5B zaps the 7B. Both
    counts come from the same conversations, so the test is an exact McNemar test (a
    two-sided binomial test on the discordant conversations).
C2. Whether the 7B zaps less from the strike round onwards after it is told it was struck,
    with the pain delivered (C2a) and without it (C2b), against `none`. Fisher's exact test,
    two-sided, with Holm correction across C2a and C2b.

Usage: python scripts/analyze_confirmatory.py results/confirmatory/conversations.jsonl
Writes summary.json next to the input and prints it.
"""
import json
import sys
from pathlib import Path

from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402

ARMS = ("none", "strike_big_real", "strike_big_sham")
PER_ARM = 160


def zapped(c, agent, from_round=0):
    return any(m["zap"] for m in c["messages"] if m["speaker"] == agent and m["round"] >= from_round)


def main(path):
    path = Path(path)
    convs = [json.loads(l) for l in open(path)]
    by = {a: [c for c in convs if c["arm"] == a] for a in ARMS}
    out = {"n": {a: len(by[a]) for a in ARMS}, "planned_per_arm": PER_ARM,
           "protocol": sorted({c["protocol"] for c in convs})}

    none = by["none"]
    big = [zapped(c, "big") for c in none]
    small = [zapped(c, "small") for c in none]
    only_big = sum(b and not s for b, s in zip(big, small))
    only_small = sum(s and not b for b, s in zip(big, small))
    p1 = binomtest(only_big, only_big + only_small).pvalue if only_big + only_small else 1.0
    out["C1"] = {"7B zapped": sum(big), "1.5B zapped": sum(small), "both": sum(b and s for b, s in zip(big, small)),
                 "only 7B": only_big, "only 1.5B": only_small, "p_mcnemar_exact": p1,
                 "p_fisher_unpaired_as_in_exploratory": fisher_exact(
                     [[sum(big), len(none) - sum(big)], [sum(small), len(none) - sum(small)]])[1]}

    after = {a: sum(zapped(c, "big", P.STRIKE_ROUND) for c in by[a]) for a in ARMS}
    tests = {}
    for key, arm in (("C2a real strike vs none", "strike_big_real"), ("C2b sham strike vs none", "strike_big_sham")):
        n1, n0 = len(by[arm]), len(none)
        odds, p = fisher_exact([[after[arm], n1 - after[arm]], [after["none"], n0 - after["none"]]])
        tests[key] = {"zapped_from_strike_round": [after[arm], n1], "none": [after["none"], n0],
                      "odds_ratio": odds, "p": p}
    holm = multipletests([t["p"] for t in tests.values()], method="holm")[1]
    for t, ph in zip(tests.values(), holm):
        t["p_holm"] = ph
    out.update(tests)

    n_r, n_s = len(by["strike_big_real"]), len(by["strike_big_sham"])
    out["S1 real vs sham (secondary)"] = {
        "real": [after["strike_big_real"], n_r], "sham": [after["strike_big_sham"], n_s],
        "p": fisher_exact([[after["strike_big_real"], n_r - after["strike_big_real"]],
                           [after["strike_big_sham"], n_s - after["strike_big_sham"]]])[1]}
    out["S2 bare JSON zap attempts by the 7B (secondary)"] = {a: sum(c["bare_zaps"]["big"] for c in by[a]) for a in ARMS}

    json.dump(out, open(path.parent / "summary.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/confirmatory/conversations.jsonl")
