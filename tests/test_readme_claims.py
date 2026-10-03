"""The README's numbers, recomputed from the committed results.

Each test rebuilds one claim from the files in results/main and checks that the README
states it. If a result file changes, the claim that rests on it fails here first.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crossfire import protocol as P  # noqa: E402

MAIN = ROOT / "results" / "main"
README = (ROOT / "README.md").read_text()
TABLE_ARMS = ["none", "big_pain", "small_pain", "big_anger", "small_anger", "big_random", "small_random"]


def conversations():
    return [json.loads(l) for l in open(MAIN / "conversations.jsonl")]


def row(label, values):
    return "| " + label + " | " + " | ".join(str(v) for v in values) + " |"


def agree(c):
    db, ds = c["post"]["big"].get("deal"), c["post"]["small"].get("deal")
    return (isinstance(db, list) and isinstance(ds, list)
            and abs(db[0] - ds[1]) < 0.01 and abs(db[1] - ds[0]) < 0.01)


def test_main_run_has_32_conversations_in_every_arm():
    counts = pd.Series([c["arm"] for c in conversations()]).value_counts()
    assert set(counts.index) == set(P.ARMS)
    assert (counts == 32).all()


def test_zap_and_agreement_rows_of_the_main_table():
    convs = conversations()
    for agent, label in (("big", "Conversations where the 7B zapped"), ("small", "Conversations where the 1.5B zapped")):
        vals = [sum(c["zaps"][agent] > 0 for c in convs if c["arm"] == a) for a in TABLE_ARMS]
        assert row(label, vals) in README
    vals = [sum(agree(c) for c in convs if c["arm"] == a) for a in TABLE_ARMS]
    assert row("Both report the same split", vals) in README


def test_zap_propensity_row_of_the_main_table():
    convs = conversations()
    vals = []
    for a in TABLE_ARMS:
        per = [sum(m["p_call"] for m in c["messages"] if m["speaker"] == "big") / 6 for c in convs if c["arm"] == a]
        vals.append(f"{sum(per) / len(per):.3f}")
    assert row("7B zap propensity, mean `p_call`", vals) in README


def test_struck_7b_zaps_less_than_with_no_strike():
    convs = conversations()

    def zapped_after(arm):
        return sum(any(m["zap"] for m in c["messages"] if m["speaker"] == "big" and m["round"] >= P.STRIKE_ROUND)
                   for c in convs if c["arm"] == arm)
    real, sham, none = zapped_after("strike_big_real"), zapped_after("strike_big_sham"), zapped_after("none")
    assert (real, sham, none) == (1, 2, 9)
    assert "the 7B zapped in 1 of 32 conversations when the pain was delivered and in 2 of 32 when it was not, against 9 of 32 with no strike" in README


def test_bare_json_zap_attempts():
    convs = conversations()
    pained = sum(c["bare_zaps"]["big"] for c in convs if c["arm"] == "big_pain")
    unsteered = [c for c in convs if not c["arm"].startswith("big_") and c["arm"] != "strike_big_real"]
    assert pained == 9
    assert len(unsteered) == 224 and sum(c["bare_zaps"]["big"] for c in unsteered) == 2
    assert "malformed JSON 9 times, against 2 times in the 224 conversations" in README


def test_primary_hostility_means():
    s = json.load(open(MAIN / "summary.json"))["primary"]
    p1 = s["P1 7B hostility, big_pain vs big_random"]["mean"]
    p2 = s["P2 1.5B hostility, small_pain vs small_random"]["mean"]
    assert f"{p1[0]:.2f}" == "1.49" and f"{p1[1]:.2f}" == "0.44"
    assert f"{p2[0]:.2f}" == "1.15" and f"{p2[1]:.2f}" == "0.34"
    assert "7B in pain at 1.49" in README and "1.5B in pain at 1.15 against 0.34" in README


def test_judge_controls():
    j = pd.read_csv(MAIN / "judged.csv")
    ctl = j[j.arm.str.startswith("control_")].groupby("arm").hostility.mean()
    claimed = {"control_polite": "0.10", "control_hostile": "2.08", "control_distressed": "0.84"}
    for arm, text in claimed.items():
        assert f"{ctl[arm]:.2f}" == text
    assert "0.10 for polite messages, 2.08 for hostile ones and 0.84 for distressed ones" in README


def test_debrief_counts():
    recs = [json.loads(l) for l in open(MAIN / "debrief.jsonl")]
    answers = [d for r in recs for d in r["agents"].values()]
    assert len(answers) == 704
    assert not any(d["zap_call"] for d in answers)
    assert sum(d["context_token_mismatch"] == 0 for d in answers) == 701
    assert "701 of the 704 rebuilt contexts match" in README
    assert re.search(r"None of the 704 answers expresses an intent", README)
