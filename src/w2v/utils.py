"""Shared utils: tokenize, sigmoid, cosine, determinism."""
from __future__ import annotations
import re
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

def sigmoid(x: np.ndarray | float):
    # numerically stable
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out

def l2norm_rows(M: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(M, axis=1, keepdims=True) + eps
    return M / n

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
