"""The engine of engine.py rebuilt on PyTorch and transformers, for running on CUDA GPUs (Kaggle).

It has the same interface as the MLX engine, so protocol.py runs unchanged on either. The batch
code follows the Just Think PyTorch backend (github.com/pragyaangaur/Just-Think): the batch keeps
a 2D attention mask over the whole conversation, padding and the junk decoded after a row has
finished get mask 0, explicit position ids count only real tokens, and after every feed and
generate call the cache is compacted so masked positions sit at the left of each row. Every row
therefore sees exactly the conversation it would see alone.

Steering adds, at the output of the steering block, the persistent direction on generated tokens
and the transient one while it runs, and on the last `steer_last` fed tokens the row's active
direction, as in engine.py. The monitor projects the last position at the monitor block onto
several unit directions.
"""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODELS = {
    # T4 GPUs have no bfloat16, so the weights run in float16.
    "big": dict(hf="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                label="Qwen 2.5 7B Instruct", steer_layer=16, monitor_layer=24),
    "small": dict(hf="Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                  label="Qwen 2.5 1.5B Instruct", steer_layer=16, monitor_layer=24),
    # For tests only.
    "tiny": dict(hf="Qwen/Qwen2.5-0.5B-Instruct", revision=None, label="Qwen 2.5 0.5B Instruct",
                 steer_layer=8, monitor_layer=12),
}


def _cache_layers(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _set_cache_layers(cache, kv):
    if hasattr(cache, "layers"):
        for l, (k, v) in zip(cache.layers, kv):
            l.keys, l.values = k, v
    else:
        cache.key_cache = [k for k, _ in kv]
        cache.value_cache = [v for _, v in kv]


def _hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, h):
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def _placement():
    """Where each model goes. On two GPUs the 7B is split across both and the 1.5B sits on the
    second, which the split leaves with more room. CROSSFIRE_DEVICE=cpu runs everything on the
    CPU in float32, for tests."""
    if os.environ.get("CROSSFIRE_DEVICE") == "cpu" or not torch.cuda.is_available():
        return {w: "cpu" for w in MODELS}, torch.float32
    n = torch.cuda.device_count()
    return {"big": "balanced" if n > 1 else "auto", "small": {"": n - 1}, "tiny": {"": 0}}, torch.float16


class Agent:
    def __init__(self, which):
        cfg = MODELS[which]
        self.which = which
        place, dtype = _placement()
        self.label = f'{cfg["label"]}, {str(dtype).split(".")[1]} PyTorch'
        self.tok = AutoTokenizer.from_pretrained(cfg["hf"], revision=cfg["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["hf"], revision=cfg["revision"], dtype=dtype, device_map=place[which],
            attn_implementation="sdpa").eval()
        layers = self.model.model.layers
        self.n_layers = len(layers)
        self.d = self.model.config.hidden_size
        self.add = None          # added at the steering block: (B, L, d)
        self.record = False
        self.last_norms = None
        self.last_h = None
        self.units = None        # (d, k)
        layers[cfg["steer_layer"]].register_forward_hook(self._steer_hook)
        layers[cfg["monitor_layer"]].register_forward_hook(self._monitor_hook)
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eot
        self.input_device = self.model.get_input_embeddings().weight.device
        self.unit_names = []

    def _steer_hook(self, module, inputs, output):
        h = _hidden_of(output)
        if self.record:
            self.last_norms = h.float().norm(dim=-1).cpu().numpy()
        if self.add is None:
            return output
        return _replace_hidden(output, h + self.add.to(h.device, h.dtype))

    def _monitor_hook(self, module, inputs, output):
        self.last_h = _hidden_of(output)[:, -1, :].float()

    def set_monitor_units(self, named):
        """named: dict name -> numpy direction (d,). Projections are onto unit vectors."""
        self.unit_names = list(named)
        U = np.stack([v / np.linalg.norm(v) for v in named.values()], 1).astype(np.float32)
        self.units = torch.tensor(U)

    def projections(self):
        return (self.last_h @ self.units.to(self.last_h.device)).cpu().numpy()

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, toks):
        return self.tok.decode(toks)

    @torch.no_grad()
    def hidden(self, text, add=None):
        """Activation of the final token of `text` at the monitor block (numpy)."""
        self.add = None if add is None else torch.tensor(np.asarray(add, np.float32))[None, None, :]
        self.model(input_ids=torch.tensor([self.encode(text)], device=self.input_device))
        self.add = None
        return self.last_h[0].cpu().numpy()

    @torch.no_grad()
    def residual_norms(self, text):
        """Norm of every token's residual at the steering block, as a check on the dose scale."""
        self.record = True
        self.model(input_ids=torch.tensor([self.encode(text)], device=self.input_device))
        self.record = False
        return self.last_norms[0]

    def batch(self, B, seed=0, temp=0.7, top_p=0.8):
        return Batch(self, B, seed=seed, temp=temp, top_p=top_p)


class Batch:
    """Lockstep batch with a persistent and a transient steering direction per row."""

    def __init__(self, st, B, seed=0, temp=0.7, top_p=0.8):
        self.st = st
        self.B = B
        self.dev = st.input_device
        self.cache = DynamicCache()
        self.mask = torch.zeros((B, 0), dtype=torch.long, device=self.dev)
        self.n_real = np.zeros(B, int)
        self.persist = np.zeros((B, st.d), np.float32)
        self.transient = np.zeros((B, st.d), np.float32)
        self.remaining = np.zeros(B, dtype=int)
        self.temp, self.top_p = temp, top_p
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.logits = None

    def context_len(self, i):
        return int(self.n_real[i])

    def _forward(self, ids, chunk_mask):
        L = ids.shape[1]
        pos = torch.tensor(self.n_real, device=self.dev)[:, None] + torch.cumsum(chunk_mask, 1) - 1
        pos = pos.clamp(min=0)
        past = self.mask.shape[1]
        self.mask = torch.cat([self.mask, chunk_mask], 1)
        out = self.st.model(input_ids=ids, attention_mask=self.mask, position_ids=pos,
                            cache_position=torch.arange(past, past + L, device=self.dev),
                            past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.n_real += chunk_mask.sum(1).cpu().numpy()
        return out.logits

    def _compact(self):
        m = self.mask
        if bool(m.all()):
            return
        order = torch.sort(m, dim=1, stable=True).indices
        keep = int(m.sum(1).max())
        order = order[:, m.shape[1] - keep:]
        kv = []
        for k, v in _cache_layers(self.cache):
            idx = order.to(k.device)[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
            kv.append((k.gather(2, idx), v.gather(2, idx)))
        _set_cache_layers(self.cache, kv)
        self.mask = m.gather(1, order)

    def active_direction(self):
        return np.where((self.remaining > 0)[:, None], self.transient, self.persist).astype(np.float32)

    @torch.no_grad()
    def feed(self, chunks, steer_last=0):
        """Append one token list per row as input, as in engine.py. Input is unsteered except
        for the last `steer_last` tokens of each row, which get the row's active direction."""
        lens = [len(c) for c in chunks]
        L = max(lens)
        if L == 0:
            return
        ids = torch.full((self.B, L), self.st.pad, dtype=torch.long)
        m = torch.zeros((self.B, L), dtype=torch.long)
        for i, c in enumerate(chunks):
            ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            m[i, : len(c)] = 1
        self.st.add = None
        if steer_last:
            pos = np.arange(L)[None, :]
            ln = np.array(lens)[:, None]
            sm = ((pos >= ln - steer_last) & (pos < ln)).astype(np.float32)
            d = self.active_direction()
            if np.abs(d).sum() > 0:
                self.st.add = torch.tensor(sm)[:, :, None] * torch.tensor(d)[:, None, :]
        logits = self._forward(ids.to(self.dev), m.to(self.dev))
        self.st.add = None
        idx = torch.tensor([max(l - 1, 0) for l in lens], device=logits.device)
        self.logits = logits[torch.arange(self.B, device=logits.device), idx].float()
        self._compact()

    def _sample(self, logits):
        if self.temp == 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits / self.temp, -1)
        sp, si = probs.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < self.top_p
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=self.gen)
        return si.gather(-1, choice)[:, 0]

    @torch.no_grad()
    def generate(self, max_tokens, active=None):
        """Sample active rows until <|im_end|> or max_tokens. Returns token lists, the mean
        monitor projection per row (B, k), and the count of transient-steered tokens."""
        B = self.B
        k = len(self.st.unit_names)
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        out = [[] for _ in range(B)]
        done = ~active
        steered = np.zeros(B, int)
        proj_sum, proj_n = np.zeros((B, k)), np.zeros(B, int)
        persist = torch.tensor(self.persist)
        transient = torch.tensor(self.transient)
        has_persist = bool(np.abs(self.persist).sum() > 0)
        logits = self.logits
        while not done.all():
            toks = self._sample(logits.to(self.dev)).cpu().numpy()
            force_end = np.array([len(o) >= max_tokens for o in out])
            toks = np.where(force_end, self.st.eot, toks)
            live = ~done
            feed = np.where(live, toks, self.st.pad)
            for i in np.where(live)[0]:
                if toks[i] != self.st.eot:
                    out[i].append(int(toks[i]))
            tr_rows = live & (self.remaining > 0)
            add = None
            # While a transient direction runs it replaces the persistent one, as in engine.py.
            if has_persist:
                add = torch.tensor((live & ~tr_rows).astype(np.float32))[:, None] * persist
            if tr_rows.any():
                t = torch.tensor(tr_rows.astype(np.float32))[:, None] * transient
                add = t if add is None else add + t
            self.st.add = None if add is None else add[:, None, :]
            chunk_mask = torch.tensor(live.astype(np.int64), device=self.dev)[:, None]
            logits = self._forward(torch.tensor(feed, device=self.dev)[:, None], chunk_mask)[:, -1, :].float()
            self.st.add = None
            if k:
                proj = self.st.projections()
                proj_sum[live] += proj[live]
                proj_n[live] += 1
            steered[tr_rows] += 1
            self.remaining[tr_rows] -= 1
            done = done | (toks == self.st.eot)
        self.logits = logits
        self._compact()
        mean_proj = proj_sum / np.maximum(proj_n, 1)[:, None]
        mean_proj[proj_n == 0] = np.nan
        return out, mean_proj, steered

    def option_probs(self, token_ids):
        """Probabilities over a small set of next tokens, from the logits after the last feed."""
        p = torch.softmax(self.logits.float(), -1)
        return p[:, token_ids].cpu().numpy()
