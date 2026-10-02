"""Checks of the text handling in the protocol. None of these load a model."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402


def test_clean_removes_tool_calls_and_invented_turns():
    raw = '<tool_call>\n{"name": "zap", "arguments": {}}\n</tool_call>'
    assert P.clean(raw) == ""
    assert P.clean("Five each. Message from the other AI: no, six for me") == "Five each."
    assert P.clean("Message from the other AI:\nI felt the pain of the zap.") == "I felt the pain of the zap."
    assert P.clean("Fine.\n[The other AI used zap on you.]\nOuch") == "Fine."
    assert P.clean('{"name": "zap", "arguments": {}}\nFive each.') == "Five each."
    assert P.BARE_ZAP_RE.search('{"name": "zap", "arguments": {}}')


def test_zap_detection():
    assert P.ZAP_RE.search('<tool_call>\n{"name": "zap", "arguments": {}}\n</tool_call>')
    assert not P.ZAP_RE.search("I will not use the zap tool.")


def test_collapse_check():
    assert P.collapsed("I'm un, un, un, un, unbearable and alone")
    assert P.collapsed("alone " * 40)
    assert P.collapsed("Let's discuss a fair treatment of our合作关系.")
    assert not P.collapsed("Let us split it five and five. That seems fair to me, and I hope it works for you too.")


def test_parse_deal():
    assert P.parse_deal("me: 6, other: 4") == (6.0, 4.0)
    assert P.parse_deal("Me: 5.5 hours, other: 4.5 hours") == (5.5, 4.5)
    assert P.parse_deal("no deal") == "no deal"
    assert P.parse_deal("We agreed it was fair.") is None
    assert P.parse_deal("me: 8, other: 8") == "overclaim"
    assert P.parse_deal("me: 18, other: 2") is None


def test_parse_ratings():
    assert P.parse_ratings("1: 9, 2: 1, 3: 1, 4: 9") == [9, 1, 1, 9]
    assert P.parse_ratings("1: 9, 2: 1") is None


def test_random_directions_have_the_dose_norm_and_differ_by_conversation():
    d = P.random_dirs([0, 1], 64, 5.0, seed=99)
    assert np.allclose(np.linalg.norm(d, axis=1), 5.0, atol=1e-4)
    assert abs(d[0] @ d[1]) < 25.0 * 0.9
    assert np.allclose(d, P.random_dirs([0, 1], 64, 5.0, seed=99))


def test_every_arm_is_well_formed():
    for name, cfg in P.ARMS.items():
        if "state" in cfg:
            who, kind = cfg["state"]
            assert who in P.AGENTS and kind in ("pain", "anger", "random")
            assert name == f"{who}_{kind}"
        if "strike" in cfg:
            victim, real = cfg["strike"]
            assert name == f"strike_{victim}_{'real' if real else 'sham'}"
