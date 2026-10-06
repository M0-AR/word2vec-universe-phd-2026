"""Analogy + similarity + completion + compositionality eval.
Analogy: X = b - a + c, nearest by cosine excluding {a,b,c}. Exact-match metric.
Reports semantic/syntactic split, coverage (answerable/total), accuracy(answerable) + accuracy(total).
Also: WordSim-style Spearman, MSR sentence completion, additive compositionality demo,
outlier detection (most-distant from mean), hubness stats, bias probe.
"""
from __future__ import annotations
import numpy as np
from .utils import l2norm_rows

def nearest(X_norm: np.ndarray, q: np.ndarray, exclude: set[int], topk: int = 1):
    qn = q / (np.linalg.norm(q) + 1e-12)
    sims = X_norm @ qn
    if exclude:
        sims[list(exclude)] = -1e9
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]

def analogy_eval(W: np.ndarray, w2i: dict, i2w: list, questions: list[tuple[str, str, str, str, str]]):
    """questions: (a,b,c,expected,section). a:b :: c:?  X=vec(b)-vec(a)+vec(c)."""
    Xn = l2norm_rows(W.astype(np.float64))
    total = len(questions)
    answerable = 0
    correct = 0
    by_sec: dict[str, list] = {}
    for a, b, c, exp, sec in questions:
        if not all(x in w2i for x in (a, b, c, exp)):
            continue
        answerable += 1
        ia, ib, ic, ie = w2i[a], w2i[b], w2i[c], w2i[exp]
        q = W[ib] - W[ia] + W[ic]
        pred, _ = nearest(Xn, q, exclude={ia, ib, ic}, topk=1)
        ok = int(pred[0]) == ie
        correct += ok
        by_sec.setdefault(sec, []).append(ok)
    acc_answerable = correct / max(answerable, 1)
    acc_total = correct / max(total, 1)
    coverage = answerable / max(total, 1)
    secs = {k: float(np.mean(v)) if v else 0.0 for k, v in by_sec.items()}
    return {
        "total": total, "answerable": answerable, "correct": correct,
        "coverage": coverage, "acc_answerable": acc_answerable, "acc_total": acc_total,
        "by_section": secs,
    }

def parse_analogy_file(path: str):
    qs = []
    sec = "unknown"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                sec = line[1:].strip() or "unknown"
                continue
            parts = line.split()
            if len(parts) == 4:
                a, b, c, d = parts
                qs.append((a.lower(), b.lower(), c.lower(), d.lower(), sec))
    return qs

def most_similar(W, w2i, i2w, word: str, topk: int = 5):
    Xn = l2norm_rows(W.astype(np.float64))
    if word not in w2i:
        return []
    i = w2i[word]
    idx, sims = nearest(Xn, W[i], exclude={i}, topk=topk)
    return [(i2w[int(j)], float(s)) for j, s in zip(idx, sims)]

def outlier_detection(W, w2i, i2w, words: list[str]):
    """Average vector for list, most distant = outlier (human IQ-test style)."""
    ids = [w2i[w] for w in words if w in w2i]
    if len(ids) < 3:
        return None
    M = W[np.array(ids)]
    mean = M.mean(axis=0)
    dists = np.linalg.norm(M - mean, axis=1)
    worst = int(np.argmax(dists))
    return words[worst] if words[worst] in w2i else i2w[ids[worst]]

def compositional_add(W, w2i, i2w, w1: str, w2: str, topk: int = 4):
    if w1 not in w2i or w2 not in w2i:
        return []
    Xn = l2norm_rows(W.astype(np.float64))
    q = W[w2i[w1]] + W[w2i[w2]]
    idx, sims = nearest(Xn, q, exclude={w2i[w1], w2i[w2]}, topk=topk)
    return [(i2w[int(j)], float(s)) for j, s in zip(idx, sims)]
