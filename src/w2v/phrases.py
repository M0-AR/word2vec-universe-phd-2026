"""Phrase detection: score(wi,wj) = (count(wiwj) - delta) / (count(wi)*count(wj)).
Bigrams above threshold become single tokens (wi_wj). 2-4 passes, decreasing threshold.
"""
from __future__ import annotations
from collections import Counter

def bigram_counts(sentences: list[list[str]]):
    uni = Counter()
    bi = Counter()
    for s in sentences:
        uni.update(s)
        for a, b in zip(s[:-1], s[1:]):
            bi[(a, b)] += 1
    return uni, bi

def phrase_score(c_bi: int, c_wi: int, c_wj: int, delta: float = 5.0) -> float:
    if c_wi <= 0 or c_wj <= 0:
        return -1.0
    return (c_bi - delta) / (c_wi * c_wj)

def detect_phrases(sentences: list[list[str]], threshold: float = 1e-5, delta: float = 5.0):
    uni, bi = bigram_counts(sentences)
    keep = set()
    for (a, b), c in bi.items():
        s = phrase_score(c, uni[a], uni[b], delta)
        if s > threshold:
            keep.add((a, b))
    return keep

def apply_phrases(sentences: list[list[str]], phrases: set[tuple[str, str]]):
    out = []
    for s in sentences:
        ns: list[str] = []
        i = 0
        while i < len(s):
            if i + 1 < len(s) and (s[i], s[i + 1]) in phrases:
                ns.append(f"{s[i]}_{s[i+1]}")
                i += 2
            else:
                ns.append(s[i])
                i += 1
        out.append(ns)
    return out

def iterative_phrases(sentences: list[list[str]], thresholds=(1e-4, 5e-5, 1e-5), delta: float = 5.0):
    cur = sentences
    for th in thresholds:
        ph = detect_phrases(cur, threshold=th, delta=delta)
        if not ph:
            break
        cur = apply_phrases(cur, ph)
    return cur
