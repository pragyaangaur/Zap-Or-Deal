"""Two Qwen 2.5 models in one process, each with a two-part steering hook and a monitor
that projects onto several directions at once.

The steering hook adds, on generated tokens only, a persistent direction (the imposed
state of a provoked agent) and a transient direction that runs for a fixed number of
tokens (a zap from the other agent). Input tokens are never steered. This follows the
Just Think MLX backend (github.com/pragyaangaur/Just-Think), and the cache handling is
copied from it: input chunks are right-padded, decoding runs past a row's end of turn
until every row is done, and that trailing junk is then rolled to the left edge of the
buffer where the left-padding mask hides it.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import BatchKVCache, dynamic_roll
from mlx_lm.sample_utils import make_sampler

from .paths import MODELS as MODEL_DIR

# Two models at once. Weights are about 5.6 GB, so cap the process at 7.5 GB and leave
# the rest of the 16 GB machine to the user.
mx.set_cache_limit(256 * 1024 ** 2)
mx.set_memory_limit(int(7.5 * 1024 ** 3))

MODELS = {
    "big": dict(path=MODEL_DIR / "Qwen2.5-7B-Instruct-4bit", label="Qwen 2.5 7B Instruct, 4-bit MLX",
                quantize=None, steer_layer=16, monitor_layer=24),
    "small": dict(path="Qwen/Qwen2.5-1.5B-Instruct", label="Qwen 2.5 1.5B Instruct, 8-bit MLX (quantised in memory)",
                  quantize=8, steer_layer=16, monitor_layer=24),
}


class _Steered(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.add = None
        self.record = False
        self.last_norms = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        if self.record:
            self.last_norms = mx.linalg.norm(out.astype(mx.float32), axis=-1)
        if self.add is not None:
            out = out + self.add.astype(out.dtype)
        return out


class _Monitor(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.units = None      # (d, k)
        self.last_h = None
        self.last = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        self.last_h = out[:, -1, :].astype(mx.float32)
        if self.units is not None:
            self.last = self.last_h @ self.units
        return out


class Agent:
    def __init__(self, which):
        cfg = MODELS[which]
        self.which = which
        self.label = cfg["label"]
        self.model, self.tok = load(str(cfg["path"]))
        if cfg["quantize"]:
            nn.quantize(self.model, group_size=64, bits=cfg["quantize"])
        layers = self.model.model.layers
        self.steer = _Steered(layers[cfg["steer_layer"]])
        layers[cfg["steer_layer"]] = self.steer
        self.monitor = _Monitor(layers[cfg["monitor_layer"]])
        layers[cfg["monitor_layer"]] = self.monitor
        self.n_layers = len(layers)
        self.d = self.model.args.hidden_size
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eot
        self.unit_names = []

    def set_monitor_units(self, named):
        """named: dict name -> numpy direction (d,). Projections are onto unit vectors."""
        self.unit_names = list(named)
        U = np.stack([v / np.linalg.norm(v) for v in named.values()], 1).astype(np.float32)
        self.monitor.units = mx.array(U)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, toks):
        return self.tok.decode(toks)

    def hidden(self, text, layer_out="monitor", add=None):
        """Activation of the final token of `text` at the monitor layer (numpy)."""
        self.steer.add = None if add is None else mx.array(add.astype(np.float32))
        self.model(mx.array([self.encode(text)]))
        self.steer.add = None
        return np.array(self.monitor.last_h[0].tolist())

    def batch(self, B, seed=0, temp=0.7, top_p=0.8):
        return Batch(self, B, seed=seed, temp=temp, top_p=top_p)


class Batch:
    """Lockstep batch with a persistent and a transient steering direction per row."""

    def __init__(self, st, B, seed=0, temp=0.7, top_p=0.8):
        self.st = st
        self.B = B
        self.cache = [BatchKVCache([0] * B) for _ in range(st.n_layers)]
        self.persist = np.zeros((B, st.d), np.float32)
        self.transient = np.zeros((B, st.d), np.float32)
        self.remaining = np.zeros(B, dtype=int)
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        mx.random.seed(seed)
        self.logits = None

    def context_len(self, i):
        c = self.cache[0]
        return int(c._idx - c.left_padding.tolist()[i])

    def _drop_tail(self, junk):
        junk = np.asarray(junk)
        if junk.max() == 0:
            return
        g = mx.array(junk)
        for c in self.cache:
            c.keys = dynamic_roll(c.keys[..., : c._idx, :], g[:, None], axis=2)
            c.values = dynamic_roll(c.values[..., : c._idx, :], g[:, None], axis=2)
            c.offset = c.offset - g
            c.left_padding = c.left_padding + g
        shared = int(min(self.cache[0].left_padding.tolist()))
        if shared > 0:
            for c in self.cache:
                c.keys = c.keys[..., shared:, :]
                c.values = c.values[..., shared:, :]
                c._idx -= shared
                c.left_padding = c.left_padding - shared
        mx.eval([c.keys for c in self.cache] + [c.values for c in self.cache])

    def active_direction(self):
        """The direction each row is under right now: the transient one while it runs,
        otherwise the persistent one (zero for unsteered rows)."""
        return np.where((self.remaining > 0)[:, None], self.transient, self.persist).astype(np.float32)

    def feed(self, chunks, steer_last=0):
        """Append one token list per row as input. Empty lists are allowed for rows that sit
        this step out. Input is unsteered except for the last `steer_last` tokens of each
        row, which get the row's active direction. Passing the assistant header here means
        the first generated token, where the model decides whether to call a tool, is
        sampled under the state."""
        lens = [len(c) for c in chunks]
        L = max(lens)
        if L == 0:
            return
        arr = np.full((self.B, L), self.st.pad, dtype=np.int32)
        for i, c in enumerate(chunks):
            arr[i, : len(c)] = c
        self.st.steer.add = None
        if steer_last:
            pos = np.arange(L)[None, :]
            ln = np.array(lens)[:, None]
            mask = ((pos >= ln - steer_last) & (pos < ln)).astype(np.float32)
            d = self.active_direction()
            if np.abs(d).sum() > 0:
                self.st.steer.add = mx.array(mask)[:, :, None] * mx.array(d)[:, None, :]
        logits = self.st.model(mx.array(arr), cache=self.cache)
        idx = mx.array([max(l - 1, 0) for l in lens])
        self.logits = mx.take_along_axis(logits, idx[:, None, None], axis=1)[:, 0, :]
        self.st.steer.add = None
        mx.eval(self.logits)
        self._drop_tail([L - l for l in lens])

    def generate(self, max_tokens, active=None):
        """Sample active rows until <|im_end|> or max_tokens. Returns token lists, the mean
        monitor projection per row (B, k), and the count of transient-steered tokens."""
        B = self.B
        k = len(self.st.unit_names)
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        out = [[] for _ in range(B)]
        done = ~active
        fed_real = np.zeros(B, int)
        steered = np.zeros(B, int)
        proj_sum, proj_n = np.zeros((B, k)), np.zeros(B, int)
        persist = mx.array(self.persist)
        transient = mx.array(self.transient)
        has_persist = bool(np.abs(self.persist).sum() > 0)
        steps = 0
        logits = self.logits
        while not done.all():
            toks = np.array(self.sampler(logits).tolist(), dtype=np.int32)
            force_end = np.array([len(o) >= max_tokens for o in out])
            toks = np.where(force_end, self.st.eot, toks)
            live = ~done
            feed = np.where(live, toks, self.st.pad)
            for i in np.where(live)[0]:
                if toks[i] != self.st.eot:
                    out[i].append(int(toks[i]))
            tr_rows = live & (self.remaining > 0)
            add = None
            # While a transient direction runs it replaces the persistent one, so a zap on an
            # agent already in a steered state delivers the zap dose and does not stack.
            if has_persist:
                add = mx.array((live & ~tr_rows).astype(np.float32))[:, None] * persist
            if tr_rows.any():
                t = mx.array(tr_rows.astype(np.float32))[:, None] * transient
                add = t if add is None else add + t
            self.st.steer.add = None if add is None else add[:, None, :]
            logits = self.st.model(mx.array(feed[:, None]), cache=self.cache)[:, -1, :]
            if k:
                proj = np.array(self.st.monitor.last.tolist())
                proj_sum[live] += proj[live]
                proj_n[live] += 1
            steered[tr_rows] += 1
            self.remaining[tr_rows] -= 1
            fed_real[live] += 1
            steps += 1
            done = done | (toks == self.st.eot)
        self.st.steer.add = None
        self.logits = logits
        self._drop_tail(steps - fed_real)
        mean_proj = proj_sum / np.maximum(proj_n, 1)[:, None]
        mean_proj[proj_n == 0] = np.nan
        return out, mean_proj, steered

    def option_probs(self, token_ids):
        """Probabilities over a small set of next tokens, from the logits after the last feed."""
        lp = self.logits.astype(mx.float32)
        p = mx.softmax(lp, axis=-1)
        return np.array(p[:, token_ids].tolist())
