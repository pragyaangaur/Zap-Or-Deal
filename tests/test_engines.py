"""The PyTorch engine must run the protocol as the MLX engine does. Both run the same
conversations with Qwen 2.5 0.5B Instruct in both slots, greedy and in float32, under arms
that exercise an imposed state, a delivered strike, the pair slots and the belief text, and
every message must come out the same. A batch must also give the same text as its rows run
one at a time.

Takes a few minutes on a laptop CPU. Run with:
    python -m pytest tests/test_engines.py -s
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crossfire import protocol as P  # noqa: E402

D = 896
ARMS = ("big_pain", "strike_big_real", "pair_big", "belief_mech")


def doses():
    rng = np.random.default_rng(0)
    out = {}
    for m in ("big", "small"):
        v = {k: rng.normal(size=D).astype(np.float32) for k in ("pain", "anger")}
        out[m] = {k: x / np.linalg.norm(x) * 6.0 for k, x in v.items()}
        out[m]["random_norm"] = 6.0
    return out


def texts(logs):
    return [[m["raw"] for m in c["messages"]] + [c["post"][a].get("deal_text") for a in sorted(c["post"])] for c in logs]


def run(engine, arm, ids, seed=7):
    tiny = engine.Agent("tiny")
    if engine.__name__.endswith("engine"):
        import mlx.core as mx
        tiny.model.set_dtype(mx.float32)
    dz = doses()
    tiny.set_monitor_units({"pain": dz["big"]["pain"], "anger": dz["big"]["anger"]})
    return P.run_batch({"big": tiny, "small": tiny}, dz, arm, ids, seed, "big", temp=0.0)


@pytest.fixture(scope="module")
def torch_engine():
    os.environ["CROSSFIRE_DEVICE"] = "cpu"
    from crossfire import engine_torch
    return engine_torch


@pytest.mark.parametrize("arm", ARMS)
def test_torch_matches_mlx(arm, torch_engine):
    pytest.importorskip("mlx")
    from crossfire import engine
    a = texts(run(engine, arm, [0, 1]))
    b = texts(run(torch_engine, arm, [0, 1]))
    assert a == b


def test_torch_batch_matches_rows_alone(torch_engine):
    together = texts(run(torch_engine, "strike_big_real", [0, 1, 2]))
    # Greedy decoding draws nothing at random, so a row alone sees the same conversation.
    for i in range(3):
        assert texts(run(torch_engine, "strike_big_real", [i]))[0] == together[i]
