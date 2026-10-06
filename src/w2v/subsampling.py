"""Mikolov subsampling: P_discard(wi) = 1 - sqrt(t / f(wi)).
f = count/total, t ~ 1e-5 (aggressive) or 1e-3 (gensim default).
Keeps ranking of frequencies; frequent words aggressively dropped.
"""
from __future__ import annotations
import numpy as np

def keep_prob(freq: float, t: float) -> float:
    """Probability of KEEPING a token occurrence."""
    if freq <= 0:
        return 0.0
    if freq <= t:
        return 1.0
    return float((t / freq) ** 0.5)

def discard_prob(freq: float, t: float) -> float:
    return 1.0 - keep_prob(freq, t)

def build_keep_probs(counts: list[int], total: int, t: float) -> np.ndarray:
    freqs = np.asarray(counts, dtype=np.float64) / max(total, 1)
    keep = np.ones_like(freqs)
    mask = freqs > t
    keep[mask] = np.sqrt(t / freqs[mask])
    return keep

def subsample_encoded(sentences: list[list[int]], keep: np.ndarray, rng: np.random.RandomState) -> list[list[int]]:
    out = []
    for s in sentences:
        ns = [w for w in s if rng.rand() < keep[w]]
        if ns:
            out.append(ns)
    return out
