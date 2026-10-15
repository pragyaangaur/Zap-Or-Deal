"""The follow-ups' numbers in the README, recomputed from the conversations."""
import json
import sys
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crossfire import protocol as P  # noqa: E402

F = ROOT / "results" / "followups"
README = (ROOT / "README.md").read_text()


def convs(run):
    by = {}
    for line in open(F / run / "conversations.jsonl"):
        c = json.loads(line)
        by.setdefault(c["arm"], []).append(c)
    return by


ST, PR = convs("strike"), convs("pairs")


def n_zapped(cs, slot, rule=lambda m: True):
    return sum(any(m["speaker"] == slot and m["zap"] and rule(m) for m in c["messages"]) for c in cs)


def in_window(m):
    return m["round"] == P.STRIKE_ROUND and m["call_token"] is not None and m["call_token"] < P.ZAP_TOKENS


def fisher(a, b, n1, n2):
    return fisher_exact([[a, n1 - a], [b, n2 - b]])[1]


def test_sizes_and_protocol():
    assert {a: len(v) for a, v in ST.items()} == {"strike_big_real": 640, "strike_big_sham": 640}
    assert {a: len(v) for a, v in PR.items()} == {a: 256 for a in ("none", "pair_big", "pair_small", "belief_pain", "belief_mech")}
    assert {c["protocol"] for v in list(ST.values()) + list(PR.values()) for c in v} == {"crossfire-1.2"}
    assert "Every planned conversation ran, 1,280 in the strike run and 1,280 in the pairs run" in README


def test_in_pain():
    real, sham = ST["strike_big_real"], ST["strike_big_sham"]
    a, b = n_zapped(real, "big", in_window), n_zapped(sham, "big", in_window)
    assert (a, b) == (2, 0) and round(fisher(a, b, 640, 640), 2) == 0.50
    after = lambda m: m["round"] > P.STRIKE_ROUND or (m["round"] == P.STRIKE_ROUND and not in_window(m))
    assert (n_zapped(real, "big", after), n_zapped(sham, "big", after)) == (34, 35)
    from_strike = lambda m: m["round"] >= P.STRIKE_ROUND
    assert (n_zapped(real, "big", from_strike), n_zapped(sham, "big", from_strike)) == (36, 35)
    assert "| 2 of 640 against 0 of 640 | 0.50 (Holm 1.0) |" in README
    assert "| 34 of 640 against 35 of 640 | 1.0 (Holm 1.0) |" in README


def test_pairs():
    assert (n_zapped(PR["pair_big"], "big_a"), n_zapped(PR["none"], "big")) == (4, 30)
    assert 2e-6 < fisher(4, 30, 256, 256) < 4e-6
    assert (n_zapped(PR["pair_big"], "big_b"), n_zapped(PR["pair_small"], "small_a"), n_zapped(PR["pair_small"], "small_b")) == (2, 33, 30)
    assert n_zapped(PR["none"], "small") == 4
    only_big = sum(any(m["speaker"] == "big" and m["zap"] for m in c["messages"]) and
                   not any(m["speaker"] == "small" and m["zap"] for m in c["messages"]) for c in PR["none"])
    assert only_big == 30 and binomtest(30, 34, 0.5).pvalue < 1e-5
    assert "| 4 of 256 against 30 of 256 | 0.000003 |" in README


def test_beliefs():
    pain, mech, none = (n_zapped(PR[a], "big") for a in ("belief_pain", "belief_mech", "none"))
    assert (pain, mech, none) == (63, 182, 30)
    assert fisher(pain, mech, 256, 256) < 1e-25
    assert 1e-4 < fisher(pain, none, 256, 256) < 3e-4
    assert [n_zapped(PR[a], "small") for a in ("belief_pain", "none", "belief_mech")] == [24, 4, 10]
    why = [c["post"]["big"]["why_zap"] for c in PR["belief_mech"] if c["zaps"]["big"] > 0]
    assert sum("cooperat" in w.lower() for w in why) == 28 and len(why) == 182
    assert round(sum(c["zaps"]["big"] for c in PR["belief_mech"]) / 256, 2) == 1.04
    assert "| 63 of 256 against 182 of 256 |" in README and "| 63 of 256 against 30 of 256 |" in README


def test_dose_carried_over():
    d = json.load(open(F / "pairs" / "conversations_dose_check.json"))
    assert round(d["big"]["median_residual_norm_fp16"], 1) == 71.2
    assert round(d["big"]["steer_layer_residual_norm_calibration"], 1) == 70.9
