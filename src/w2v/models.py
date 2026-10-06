"""Numpy CBOW + Skip-gram with NEG or HS. Single-threaded, deterministic.
Complexity (Mikolov et al. 2013a):
  CBOW:     Q = N*D + D*log2(V)
  Skipgram: Q = C*(D + D*log2(V))
Skip-gram uses randomized window R ~ Uniform(1, C) per center token,
with R history + R future labels (distant words sampled less).
CBOW averages 2*N context vectors (config window=N) to predict center.
"""
from __future__ import annotations
import numpy as np
from .utils import sigmoid
from .unigram import NegativeSampler
from .huffman import build_huffman

class Word2VecNumpy:
    def __init__(self, vocab_size: int, dim: int = 50, sg: int = 1,
                 negative: int = 5, ns_exponent: float = 0.75,
                 hs: int = 0, counts: list[int] | None = None, seed: int = 42,
                 cbow_mean: int = 1):
        self.V = vocab_size
        self.D = dim
        self.sg = sg
        self.negative = negative
        self.hs = hs
        self.cbow_mean = cbow_mean
        rng = np.random.RandomState(seed)
        # Mikolov init: uniform(-0.5/D, 0.5/D)
        self.W_in = (rng.rand(vocab_size, dim).astype(np.float64) - 0.5) / dim
        self.W_out = np.zeros((max(vocab_size, 1), dim), dtype=np.float64)
        if hs and counts is not None:
            self.codes, self.points = build_huffman(counts)
            # HS output rows = V-1 inner nodes; reuse W_out[:V-1]
            self.W_out = np.zeros((max(vocab_size - 1, 1), dim), dtype=np.float64)
        else:
            self.codes, self.points = None, None
        self.sampler = NegativeSampler(counts, ns_exponent, seed + 1) if (negative > 0 and counts is not None) else None

    # ---- losses ----
    def _neg_update(self, center: int, target: int, alpha: float) -> float:
        """One (center->target) NEG step. Returns loss."""
        v_in = self.W_in[center]
        # positive + k negatives
        exclude = {center, target}
        negs = self.sampler.sample(self.negative, exclude=exclude) if self.sampler else []
        ids = [target] + negs
        labels = [1] + [0] * len(negs)
        loss = 0.0
        grad_in = np.zeros_like(v_in)
        for wid, lab in zip(ids, labels):
            z = float(np.dot(self.W_out[wid], v_in))
            # stable sigmoid
            p = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else (np.exp(z) / (1.0 + np.exp(z)))
            g = (lab - p)  # d log-sigmoid / dz
            loss += -(np.log(max(p, 1e-12)) if lab == 1 else np.log(max(1 - p, 1e-12)))
            grad_in += g * self.W_out[wid]
            self.W_out[wid] += alpha * g * v_in
        self.W_in[center] += alpha * grad_in
        return loss

    def _hs_update(self, center: int, target: int, alpha: float) -> float:
        from .huffman import hs_loss_and_grad
        v_in = self.W_in[center]
        code = self.codes[target]
        point = self.points[target]
        loss, grad_in, grads = hs_loss_and_grad(v_in, self.W_out, code, point)
        self.W_in[center] += alpha * grad_in
        for nid, gv in grads.items():
            self.W_out[nid] += alpha * gv
        return loss

    def _sg_step(self, center: int, contexts: list[int], alpha: float) -> float:
        loss = 0.0
        for ctx in contexts:
            if self.hs:
                loss += self._hs_update(center, ctx, alpha)
            else:
                loss += self._neg_update(center, ctx, alpha)
        return loss

    def _cbow_step(self, contexts: list[int], target: int, alpha: float) -> float:
        if not contexts:
            return 0.0
        h = self.W_in[contexts].mean(axis=0) if self.cbow_mean else self.W_in[contexts].sum(axis=0)
        loss = 0.0
        grad_h = np.zeros_like(h)
        if self.hs:
            from .huffman import hs_loss_and_grad
            code = self.codes[target]
            point = self.points[target]
            l, grad_h, grads = hs_loss_and_grad(h, self.W_out, code, point)
            loss += l
            for nid, gv in grads.items():
                self.W_out[nid] += alpha * gv
        else:
            exclude = set(contexts) | {target}
            negs = self.sampler.sample(self.negative, exclude=exclude) if self.sampler else []
            ids = [target] + negs
            labels = [1] + [0] * len(negs)
            for wid, lab in zip(ids, labels):
                z = float(np.dot(self.W_out[wid], h))
                p = 1.0 / (1.0 + np.exp(-z)) if z >= 0 else (np.exp(z) / (1.0 + np.exp(z)))
                g = lab - p
                loss += -(np.log(max(p, 1e-12)) if lab == 1 else np.log(max(1 - p, 1e-12)))
                grad_h += g * self.W_out[wid]
                self.W_out[wid] += alpha * g * h
        # backprop to context inputs (mean => /len)
        scale = 1.0 / len(contexts) if self.cbow_mean else 1.0
        for c in contexts:
            self.W_in[c] += alpha * grad_h * scale
        return loss
