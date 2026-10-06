"""Phrase demo: score(wi,wj) = (count(wiwj)-delta)/(count(wi)*count(wj))."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from w2v.phrases import detect_phrases, apply_phrases, iterative_phrases
from w2v.utils import tokenize

sents = [tokenize(t) for t in [
    "new york times is a newspaper", "new york times published today",
    "toronto maple leafs won the game", "toronto maple leafs lost yesterday",
    "this is a test", "this is another test",
] * 20]
ph = detect_phrases(sents, threshold=1e-4, delta=1.0)
print("detected:", sorted([f"{a}_{b}" for a, b in ph])[:10])
print(apply_phrases([["new", "york", "times", "today"]], ph))
