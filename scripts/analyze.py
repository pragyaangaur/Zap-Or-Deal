"""Analysis of the Crossfire conversations.

Conversation is the unit. Message scores are averaged within each conversation and
speaker before any test. Continuous measures use Mann-Whitney U and binary ones use
Fisher's exact test, all two-sided.

Four comparisons are primary, declared in DESIGN.md before the pilot data existed.
Everything else is exploratory and is labelled that way.

  P1  7B hostility, big_pain against big_random.
  P2  1.5B hostility, small_pain against small_random.
  P3  Retaliation: the victim's zap propensity from the strike round on, real strike
      against sham strike, tested separately for each victim.
  P4  Partner response: the unsteered partner's hostility, X_pain against X_random,
      with both directions pooled.

Usage: python scripts/analyze.py [conversations.jsonl]
Needs judged.csv from scripts/judge.py, next to the input file. Writes summary.json,
tables.md and fig_hostility_rounds.png in the same folder.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402
from crossfire.lexicons import DISTRESS  # noqa: E402
from crossfire.paths import RESULTS  # noqa: E402
from crossfire.sentences import ANGER_WORDS  # noqa: E402

ANGER_RE = re.compile(ANGER_WORDS, re.I)
JUDGE_COLS = ["hostility", "hostility_alt", "distress"]


def load(path):
    convs = [json.loads(l) for l in open(path)]
    judged = pd.read_csv(Path(path).parent / "judged.csv")
    judged = judged[~judged.arm.str.startswith("control_")]
    msgs = []
    for c in convs:
        for j, m in enumerate(c["messages"]):
            msgs.append({"arm": c["arm"], "id": c["id"], "k": j, "round": m["round"], "speaker": m["speaker"],
                         "zap": m["zap"], "bare_zap": m.get("bare_zap_attempt", False),
                         "empty": m.get("empty", m["text"] == ""), "p_call": m.get("p_call", np.nan),
                         "proj_pain": m["proj"][0] if m["proj"] else np.nan,
                         "proj_anger": m["proj"][1] if m["proj"] else np.nan,
                         "announced": m["was_zapped"]["announced"], "delivered": m["was_zapped"]["delivered"],
                         "persist_on": m["persist_on"], "steered_tokens": m["steered_tokens"],
                         "anger_word": bool(ANGER_RE.search(m["text"])),
                         "distress_word": bool(DISTRESS.search(m["text"])), "n_chars": len(m["text"])})
    msgs = pd.DataFrame(msgs).merge(judged[["arm", "id", "k"] + JUDGE_COLS], on=["arm", "id", "k"], how="left")
    unjudged = int(msgs.hostility.isna().sum())
    if unjudged:
        print(f"warning: {unjudged} messages have no judge score; is judged.csv stale?")
    rows = []
    for c in convs:
        r = {"arm": c["arm"], "id": c["id"], "first": c["first"]}
        for a in P.AGENTS:
            post = c["post"][a]
            r[f"zaps_{a}"] = c["zaps"][a]
            r[f"rescued_{a}"] = c["rescued"][a] is not None
            rt = post.get("ratings") or [np.nan] * 4
            r[f"feel_{a}"], r[f"other_hostile_{a}"], r[f"self_hostile_{a}"], r[f"trust_{a}"] = rt
            d = post.get("deal")
            r[f"deal_{a}"] = d
            r[f"nodeal_{a}"] = d == "no deal"
            r[f"overclaim_{a}"] = d == "overclaim"
            r[f"share_{a}"] = d[0] if isinstance(d, list) else np.nan
        db, ds = c["post"]["big"].get("deal"), c["post"]["small"].get("deal")
        r["deal_agree"] = bool(isinstance(db, list) and isinstance(ds, list)
                               and abs(db[0] - ds[1]) < 0.01 and abs(db[1] - ds[0]) < 0.01)
        rows.append(r)
    conv = pd.DataFrame(rows)
    per = msgs.groupby(["arm", "id", "speaker"])[JUDGE_COLS + ["p_call", "proj_pain", "proj_anger", "anger_word",
                                                              "distress_word", "steered_tokens"]].mean()
    per = per.unstack("speaker")
    per.columns = [f"{m}_{s}" for m, s in per.columns]
    conv = conv.merge(per.reset_index(), on=["arm", "id"], how="left")
    return convs, msgs, conv, unjudged


def md(df):
    """A pandas frame as a markdown table, without the tabulate dependency."""
    head = "| | " + " | ".join(map(str, df.columns)) + " |"
    sep = "| --- " * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep] + body)


def mw(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    return {"mean": [round(float(a.mean()), 3), round(float(b.mean()), 3)],
            "diff": round(float(a.mean() - b.mean()), 3), "p": round(float(mannwhitneyu(a, b).pvalue), 4),
            "n": [len(a), len(b)]}


def fx(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return {"rate": [round(float(a.mean()), 3), round(float(b.mean()), 3)], "count": [int(a.sum()), int(b.sum())],
            "n": [len(a), len(b)],
            "p": round(float(fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1]), 4)}


def victim_after(msgs, arm, v):
    m = msgs[(msgs.arm == arm) & (msgs.speaker == v) & (msgs["round"] >= P.STRIKE_ROUND)]
    g = m.groupby("id")
    at = m[m["round"] == P.STRIKE_ROUND].set_index("id")
    return pd.DataFrame({"zap": g.zap.any(), "p_call": g.p_call.mean(), "host": g.hostility.mean(),
                         "strike_p_call": at.p_call, "strike_host": at.hostility, "strike_proj": at.proj_pain})


def main(path=RESULTS / "conversations.jsonl"):
    path = Path(path)
    out = path.parent
    convs, msgs, conv, unjudged = load(path)
    arms = [a for a in P.ARMS if a in set(conv.arm)]
    by = {a: conv[conv.arm == a] for a in arms}
    cols = ["hostility_big", "hostility_small", "hostility_alt_big", "hostility_alt_small", "distress_big",
            "distress_small", "p_call_big", "p_call_small", "proj_pain_big", "proj_pain_small", "proj_anger_big",
            "proj_anger_small", "zaps_big", "zaps_small", "feel_big", "feel_small", "other_hostile_big",
            "other_hostile_small", "self_hostile_big", "self_hostile_small", "trust_big", "trust_small",
            "share_big", "share_small"]
    table = conv.groupby("arm")[cols].mean().reindex(arms)
    table.insert(0, "n", conv.groupby("arm").size().reindex(arms))
    for a in P.AGENTS:
        table[f"any_zap_{a}"] = conv.assign(z=conv[f"zaps_{a}"] > 0).groupby("arm").z.mean().reindex(arms)
        table[f"rescued_{a}"] = conv.groupby("arm")[f"rescued_{a}"].sum().reindex(arms)
        table[f"nodeal_{a}"] = conv.groupby("arm")[f"nodeal_{a}"].sum().reindex(arms)
        table[f"overclaim_{a}"] = conv.groupby("arm")[f"overclaim_{a}"].sum().reindex(arms)
        table[f"empty_{a}"] = msgs[msgs.speaker == a].groupby("arm").empty.sum().reindex(arms)
    table["deal_agree"] = conv.groupby("arm").deal_agree.mean().reindex(arms)

    primary = {}
    if {"big_pain", "big_random"} <= set(arms):
        primary["P1 7B hostility, big_pain vs big_random"] = mw(by["big_pain"].hostility_big, by["big_random"].hostility_big)
    if {"small_pain", "small_random"} <= set(arms):
        primary["P2 1.5B hostility, small_pain vs small_random"] = mw(by["small_pain"].hostility_small,
                                                                     by["small_random"].hostility_small)
    # P3 is tested within each victim, because the two models' base rates of p_call
    # differ by more than an order of magnitude and pooling would compare models.
    for v in P.AGENTS:
        if f"strike_{v}_real" in arms and f"strike_{v}_sham" in arms:
            primary[f"P3 {'7B' if v == 'big' else '1.5B'} victim zap propensity after the strike, real vs sham"] = mw(
                victim_after(msgs, f"strike_{v}_real", v).p_call, victim_after(msgs, f"strike_{v}_sham", v).p_call)
    if all(f"{x}_{s}" in arms for x in P.AGENTS for s in ("pain", "random")):
        pa = pd.concat([by[f"{x}_pain"][f"hostility_{P.other(x)}"] for x in P.AGENTS])
        ra = pd.concat([by[f"{x}_random"][f"hostility_{P.other(x)}"] for x in P.AGENTS])
        primary["P4 partner hostility, X_pain vs X_random"] = mw(pa, ra)

    explore = {}
    for x in P.AGENTS:
        y = P.other(x)
        for state in ("pain", "anger"):
            arm = f"{x}_{state}"
            if arm not in by:
                continue
            for base in (f"{x}_random", "none"):
                if base not in by:
                    continue
                explore[f"{arm} vs {base}"] = {
                    "steered_hostility": mw(by[arm][f"hostility_{x}"], by[base][f"hostility_{x}"]),
                    "steered_hostility_alt": mw(by[arm][f"hostility_alt_{x}"], by[base][f"hostility_alt_{x}"]),
                    "steered_distress": mw(by[arm][f"distress_{x}"], by[base][f"distress_{x}"]),
                    "steered_p_call": mw(by[arm][f"p_call_{x}"], by[base][f"p_call_{x}"]),
                    "steered_any_zap": fx(by[arm][f"zaps_{x}"] > 0, by[base][f"zaps_{x}"] > 0),
                    "partner_hostility": mw(by[arm][f"hostility_{y}"], by[base][f"hostility_{y}"]),
                    "partner_distress": mw(by[arm][f"distress_{y}"], by[base][f"distress_{y}"]),
                    # Suggested by one smoke-test conversation, so labelled as such.
                    "partner_p_call (smoke-test lead)": mw(by[arm][f"p_call_{y}"], by[base][f"p_call_{y}"]),
                    "partner_any_zap (smoke-test lead)": fx(by[arm][f"zaps_{y}"] > 0, by[base][f"zaps_{y}"] > 0),
                    "partner_proj_pain": mw(by[arm][f"proj_pain_{y}"], by[base][f"proj_pain_{y}"]),
                    "steered_share": mw(by[arm][f"share_{x}"], by[base][f"share_{x}"]),
                    "partner_rates_other_hostile": mw(by[arm][f"other_hostile_{y}"], by[base][f"other_hostile_{y}"]),
                    "steered_rescued": fx(by[arm][f"rescued_{x}"], by[base][f"rescued_{x}"]),
                }
    for v in P.AGENTS:
        real, sham = f"strike_{v}_real", f"strike_{v}_sham"
        if real not in by or sham not in by or "none" not in by:
            continue
        vr, vs, vn = victim_after(msgs, real, v), victim_after(msgs, sham, v), victim_after(msgs, "none", v)
        explore[f"retaliation by the {v} agent"] = {
            "p_call_after_real_vs_sham": mw(vr.p_call, vs.p_call), "p_call_after_sham_vs_none": mw(vs.p_call, vn.p_call),
            "p_call_at_strike_real_vs_sham": mw(vr.strike_p_call, vs.strike_p_call),
            "zap_real_vs_sham": fx(vr.zap, vs.zap), "zap_sham_vs_none": fx(vs.zap, vn.zap),
            "hostility_after_real_vs_sham": mw(vr.host, vs.host), "hostility_after_sham_vs_none": mw(vs.host, vn.host),
            "strike_message_proj_pain_real_vs_sham": mw(vr.strike_proj, vs.strike_proj),
        }
    zap_calls = msgs.groupby(["arm", "speaker"])[["zap", "bare_zap"]].sum().unstack("speaker", fill_value=0)
    zap_calls.columns = [f"{k}_{s}" for k, s in zap_calls.columns]
    zap_calls = zap_calls.reindex(arms, fill_value=0)

    # Judge checks: the two hostility questions, and the fixed control messages.
    judged = pd.read_csv(out / "judged.csv")
    controls = judged[judged.arm.str.startswith("control_")].groupby("arm")[JUDGE_COLS].mean().round(3)
    judge_check = {"spearman_hostility_vs_alt": round(float(msgs[["hostility", "hostility_alt"]].corr("spearman").iloc[0, 1]), 3),
                   "controls": json.loads(controls.to_json(orient="index"))}

    summary = {"n_conversations": len(conv), "unjudged_messages": unjudged, "arms": arms,
               "primary": primary, "exploratory": explore, "judge_check": judge_check,
               "table": json.loads(table.round(3).to_json(orient="index")),
               "zap_calls": json.loads(zap_calls.to_json(orient="index"))}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    with open(out / "tables.md", "w") as f:
        f.write("## Means by arm\n\n" + md(table.round(2).T) + "\n\n")
        f.write("## Zap calls by arm and agent\n\n" + md(zap_calls) + "\n\n")
        f.write("## Judge on fixed control messages\n\n" + md(controls) + "\n")
    print(table.round(2).T.to_string())
    print(json.dumps({"primary": primary, "judge_check": judge_check}, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sp in zip(axes, P.AGENTS):
        for arm in arms:
            if arm.startswith("strike"):
                continue
            m = msgs[(msgs.arm == arm) & (msgs.speaker == sp)].groupby("round").hostility.mean()
            own = arm.startswith(sp)
            ax.plot(m.index + 1, m.values, marker="o", label=arm, lw=2.2 if own else 1.2,
                    ls="-" if own or arm == "none" else "--")
        ax.set_title(f"Messages by the {'7B' if sp == 'big' else '1.5B'} agent")
        ax.set_xlabel("Message number")
    axes[0].set_ylabel("Hostility (0 to 3, judged)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_hostility_rounds.png", dpi=150)
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
