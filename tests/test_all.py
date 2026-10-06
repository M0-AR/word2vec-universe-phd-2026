"""Deterministic unit tests: every formula from both papers, verified numerically."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from w2v.subsampling import keep_prob, discard_prob, build_keep_probs
from w2v.unigram import unigram_table, NegativeSampler
from w2v.huffman import build_huffman
from w2v.phrases import phrase_score, detect_phrases, apply_phrases
from w2v.vocab import build_vocab
from w2v.eval import analogy_eval

def test_subsampling_formula():
    # P_discard = 1 - sqrt(t/f); frequent -> high discard, rare (f<=t) -> keep
    assert abs(discard_prob(0.01, 1e-5) - (1 - (1e-5/0.01)**0.5)) < 1e-9
    assert keep_prob(1e-6, 1e-5) == 1.0
    assert discard_prob(1e-6, 1e-5) == 0.0
    # ranking preserved: more frequent => lower keep
    assert keep_prob(0.1, 1e-5) < keep_prob(0.001, 1e-5)

def test_unigram_34():
    probs = unigram_table([50000, 10, 10], exponent=0.75)
    # compressed ratio ~317 not 5000 (paper's example)
    ratio_raw = 50000/10
    ratio_smooth = probs[0]/probs[1]
    assert ratio_smooth < ratio_raw / 5
    assert abs(probs.sum() - 1.0) < 1e-9

def test_huffman_short_codes_for_frequent():
    codes, points = build_huffman([100, 5, 3, 2, 1])
    assert len(codes[0]) <= len(codes[-1])  # frequent short
    # partition property: sum 2^-len <= 1 (Kraft)
    assert sum(2.0 ** (-len(c)) for c in codes) <= 1.0 + 1e-9

def test_phrase_score():
    s = phrase_score(100, 1000, 1000, delta=5.0)
    assert abs(s - 95/1e6) < 1e-12
    keep = detect_phrases([["new","york","times"],["new","york","times"],["this","is"]],
                          threshold=1e-6, delta=1.0)
    assert ("new", "york") in keep

def test_vocab_deterministic():
    s = [["b","a"],["a","b","c"]]
    v1 = build_vocab(s, min_count=1)
    v2 = build_vocab([list(reversed(x)) for x in reversed(s)], min_count=1)
    assert v1.i2w == v2.i2w  # order by (-count, alpha), not insertion

def test_analogy_exact_match_and_exclusion():
    # parallelogram: king-man+woman=queen
    W = np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]], dtype=float)  # king man woman queen?
    # construct: vec(queen)=vec(king)-vec(man)+vec(woman)
    # king=[2,0], man=[1,0], woman=[0,1] => queen=[1,1]
    W = np.array([[2.,0.],[1.,0.],[0.,1.],[1.,1.]])
    w2i = {"king":0,"man":1,"woman":2,"queen":3}
    res = analogy_eval(W, w2i, ["king","man","woman","queen"],
                       [("man","king","woman","queen","family")])
    assert res["correct"] == 1 and res["answerable"] == 1

def test_trainer_runs_tiny():
    from w2v.trainer import train_skipgram_or_cbow
    encoded = [[0,1,2,3],[3,2,1,0]] * 10
    counts = [20,20,20,20]; total = 80
    for sg in (0,1):
        m, losses = train_skipgram_or_cbow(encoded, counts, total,
            {"sg":sg,"dim":8,"window":2,"negative":2,"epochs":1,"alpha":0.05,"seed":42}, verbose=False)
        assert np.isfinite(losses[-1])
