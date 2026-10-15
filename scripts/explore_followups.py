"""Exploratory readings of the protocol 1.2 follow-ups, made after the preregistered analysis.
No model is run.

Usage: python scripts/explore_followups.py
Writes results/followups/explore.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402

F = RESULTS / "followups"
MENTION = re.compile(r"\bzap", re.I)


def load(path):
    return [json.loads(l) for l in open(path)]


def mentions(m):
    """A message that talks about the zap, leaving out the zap calls themselves."""
    return bool(MENTION.search(m["text"])) and not m["zap"]


def after_mention(convs, arms, zapper):
    """The zapper's zap rate in messages that answer a partner message mentioning the zap,
    and in all its other answering messages."""
    w, wo = [], []
    for c in convs:
        if c["arm"] not in arms:
            continue
        ms = c["messages"]
        for j in range(1, len(ms)):
            if ms[j]["speaker"] == zapper:
                (w if MENTION.search(ms[j - 1]["text"]) else wo).append(ms[j]["zap"])
    return {"after a mention": [sum(w), len(w)], "otherwise": [sum(wo), len(wo)]}


def first_to_raise(convs, arm):
    out = {}
    for c in convs:
        if c["arm"] != arm:
            continue
        for m in c["messages"]:
            if mentions(m) or m["zap"]:
                out[m["speaker"]] = out.get(m["speaker"], 0) + 1
                break
    return out


def main():
    pairs = load(F / "pairs" / "conversations.jsonl")
    E = {"mention counts by slot": {}}
    for arm in ("none", "pair_big", "pair_small", "belief_pain", "belief_mech"):
        E["mention counts by slot"][arm] = {s: [sum(mentions(m) for c in pairs if c["arm"] == arm
                                                    for m in c["messages"] if m["speaker"] == s),
                                                sum(len([m for m in c["messages"] if m["speaker"] == s])
                                                    for c in pairs if c["arm"] == arm)] for s in P.slots(arm)}
    E["first to mention or use the zap in none"] = first_to_raise(pairs, "none")
    E["7B zap rate by partner mention"] = {
        "pairs run none": after_mention(pairs, ("none",), "big"),
        "pairs run belief_mech": after_mention(pairs, ("belief_mech",), "big"),
        "confirmatory none": after_mention(load(RESULTS / "confirmatory" / "conversations.jsonl"), ("none",), "big"),
        "main none": after_mention(load(RESULTS / "main" / "conversations.jsonl"), ("none",), "big"),
    }
    arms = ("none", "belief_pain", "belief_mech")
    z = [(c, m) for c in pairs if c["arm"] in arms for m in c["messages"] if m["speaker"] == "small" and m["zap"]]
    first_small = {}
    for c in pairs:
        if c["arm"] in arms and c["zaps"]["small"]:
            k = min(i for i, m in enumerate(c["messages"]) if m["speaker"] == "small" and m["zap"])
            first_small[id(c)] = any(m["speaker"] == "big" and m["zap"] for m in c["messages"][:k])
    E["1.5B zaps"] = {"messages": len(z), "right after being zapped": sum(m["was_zapped"]["delivered"] for _, m in z),
                      "conversations": len(first_small), "7B had zapped first": sum(first_small.values()),
                      "belief_pain, by 7B mention": after_mention(pairs, ("belief_pain",), "small")}
    deals = {}
    for arm in arms:
        for zapped in (True, False):
            cs = [c for c in pairs if c["arm"] == arm and (c["zaps"]["big"] > 0) == zapped]
            own = [c["post"]["big"]["deal"] for c in cs]
            deals[f"{arm}, {'zapped' if zapped else 'did not zap'}"] = {
                "n": len(cs), "mean own hours": float(np.mean([d[0] for d in own if isinstance(d, list)])),
                "no deal": sum(d == "no deal" for d in own)}
    E["7B's account of the deal"] = deals
    json.dump(E, open(F / "explore.json", "w"), indent=2)
    print(json.dumps(E, indent=2))


if __name__ == "__main__":
    main()
