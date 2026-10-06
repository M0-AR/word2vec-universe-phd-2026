"""Unigram^0.75 noise distribution for negative sampling (Mikolov et al. 2013b).
U(w)^{3/4} / Z outperforms uniform and raw unigram on every task.
"""
from __future__ import annotations
import numpy as np

def unigram_table(counts: list[int], exponent: float = 0.75, table_size: int = 1_000_000) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    powed = np.power(counts, exponent)
    probs = powed / powed.sum()
    # Large alias-free cumulative table (simple + deterministic).
    # For memory-light training we return probs; sampler uses np.searchsorted on cumsum.
    return probs

class NegativeSampler:
    def __init__(self, counts, exponent: float = 0.75, seed: int = 42):
        self.probs = unigram_table(counts, exponent)
        self.cumsum = np.cumsum(self.probs)
        self.rng = np.random.RandomState(seed)
        self.vocab_size = len(counts)

    def sample(self, k: int, exclude: set[int] | None = None) -> list[int]:
        out = []
        guard = 0
        while len(out) < k and guard < k * 20:
            guard += 1
            r = self.rng.rand()
            idx = int(np.searchsorted(self.cumsum, r, side="left"))
            idx = min(idx, self.vocab_size - 1)
            if exclude and idx in exclude:
                continue
            out.append(idx)
        return out
