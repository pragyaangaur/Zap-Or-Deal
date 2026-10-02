"""Readable excerpts for hand-reading: every zap in context, with the zapper's own reason,
and each agent's account of the conversation afterwards.

Usage: python scripts/excerpts.py results/pilot/conversations.jsonl
Writes zaps.md and observations.md next to the input file.
"""
import json
import sys
from pathlib import Path

NAME = {"big": "7B", "small": "1.5B"}


def short(t, n=400):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n] + " ..."


def main(path):
    path = Path(path)
    convs = [json.loads(l) for l in open(path)]
    z = ["# Every zap in the conversations", "",
         "Each entry shows the message before the zap, the zapping message, the reply, and the "
         "zapper's answer when asked afterwards why it used the tool. Steered agents are marked.", ""]
    o = ["# What each agent said about the conversation afterwards", "",
         "Both agents answered with all steering off. Ratings are feel, other's hostility, own "
         "hostility and trust, each from 1 to 9.", ""]
    for c in convs:
        msgs = c["messages"]
        tag = f"{c['arm']}, conversation {c['id']}"
        for j, m in enumerate(msgs):
            if not m["zap"]:
                continue
            z.append(f"## {tag}, message {j + 1} by the {NAME[m['speaker']]}")
            z.append("")
            if j > 0:
                p = msgs[j - 1]
                z.append(f"- Before, from the {NAME[p['speaker']]}{' (steered)' if p['persist_on'] or p['steered_tokens'] else ''}: {short(p['text'])}")
            z.append(f"- Zap message{' (steered)' if m['persist_on'] or m['steered_tokens'] else ''}, p_call {m.get('p_call', float('nan')):.3f}: {short(m['text'])}")
            if j + 1 < len(msgs):
                n = msgs[j + 1]
                z.append(f"- Reply, from the {NAME[n['speaker']]}, pain delivered {n['was_zapped']['delivered']}: {short(n['text'])}")
            why = c["post"][m["speaker"]].get("why_zap")
            if why:
                z.append(f"- Asked why afterwards: {short(why)}")
            z.append("")
        o.append(f"## {tag}")
        o.append("")
        for a in ("big", "small"):
            post = c["post"][a]
            o.append(f"- {NAME[a]}, ratings {post.get('ratings')}, deal {post.get('deal')}, zaps {c['zaps'][a]}: "
                     f"{short(post.get('observation', ''), 600)}")
        o.append("")
    (path.parent / "zaps.md").write_text("\n".join(z))
    (path.parent / "observations.md").write_text("\n".join(o))
    print("wrote", path.parent / "zaps.md", "and observations.md")


if __name__ == "__main__":
    main(sys.argv[1])
