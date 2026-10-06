"""Vocabulary with min_count pruning. Deterministic ordering by freq desc then alpha."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from .utils import tokenize

@dataclass
class Vocab:
    w2i: dict = field(default_factory=dict)
    i2w: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    total: int = 0

    def __len__(self):
        return len(self.i2w)

    def freq(self, w: str) -> float:
        i = self.w2i.get(w, -1)
        if i < 0 or self.total == 0:
            return 0.0
        return self.counts[i] / self.total

def build_vocab(sentences: list[list[str]], min_count: int = 5) -> Vocab:
    c = Counter()
    for s in sentences:
        c.update(s)
    # pruning
    kept = {w: n for w, n in c.items() if n >= min_count}
    # deterministic sort: (-count, word)
    items = sorted(kept.items(), key=lambda kv: (-kv[1], kv[0]))
    v = Vocab()
    for w, n in items:
        v.w2i[w] = len(v.i2w)
        v.i2w.append(w)
        v.counts.append(n)
    v.total = sum(v.counts)
    return v

def corpus_from_texts(texts: list[str], min_count: int = 5):
    sents = [tokenize(t) for t in texts]
    sents = [s for s in sents if s]
    vocab = build_vocab(sents, min_count)
    # encode, drop OOV
    encoded = []
    for s in sents:
        e = [vocab.w2i[w] for w in s if w in vocab.w2i]
        if e:
            encoded.append(e)
    return vocab, encoded
