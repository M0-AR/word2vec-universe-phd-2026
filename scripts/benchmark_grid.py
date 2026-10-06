"""Grid benchmark: sg x {0,1} x window x {2,5} x dim x {25,50} x neg x {5,15} x sample_t.
Tiny enough to finish on CPU in minutes; reports coverage+accuracy split (the honest metric).
Usage: python scripts/benchmark_grid.py --config configs/tiny.yaml
"""
from __future__ import annotations
import argparse, os, sys, itertools, json, yaml, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from w2v.vocab import corpus_from_texts
from w2v.trainer import train_skipgram_or_cbow
from w2v.eval import analogy_eval, parse_analogy_file
from w2v.utils import tokenize, set_seed
from w2v.synthetic import SYNTH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    base = yaml.safe_load(open(args.config))
    sents = [tokenize(t) for t in SYNTH[: base.get("synthetic_sentences", 2000)]]
    vocab, encoded = corpus_from_texts([" ".join(s) for s in sents], min_count=base.get("min_count", 2))
    qs = parse_analogy_file(base.get("analogy_file", "data/questions-words.sample.txt"))
    grid = list(itertools.product([0, 1], [2, 5], [25, 50]))
    rows = []
    for sg, win, dim in grid:
        cfg = dict(base, sg=sg, window=win, dim=dim, epochs=2)
        t0 = time.time()
        set_seed(42)
        model, losses = train_skipgram_or_cbow(encoded, vocab.counts, vocab.total, cfg, verbose=False)
        res = analogy_eval(model.W_in, {w: i for i, w in enumerate(vocab.i2w)}, vocab.i2w, qs)
        row = {"sg": sg, "window": win, "dim": dim, "final_loss": losses[-1],
               "acc_total": res["acc_total"], "acc_answerable": res["acc_answerable"],
               "coverage": res["coverage"], "correct": res["correct"],
               "answerable": res["answerable"], "secs": time.time() - t0}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(rows, open(os.path.join(args.outdir, "benchmark_grid.json"), "w"), indent=2)
    # hidden-pattern summary
    sg1 = [r for r in rows if r["sg"] == 1]
    sg0 = [r for r in rows if r["sg"] == 0]
    print(f"\n# PATTERN: skipgram mean acc_total={np.mean([r['acc_total'] for r in sg1]):.3f} "
          f"vs cbow={np.mean([r['acc_total'] for r in sg0]):.3f} on synthetic micro-corpus")
    print("# NOTE: absolute numbers are meaningless on 2k sentences; RELATIVE ordering + coverage is the finding.")

if __name__ == "__main__":
    main()
