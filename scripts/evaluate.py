"""Evaluate saved model on analogy sample. Usage: python scripts/evaluate.py --config configs/tiny.yaml"""
from __future__ import annotations
import argparse, os, sys, json, yaml
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from w2v.eval import analogy_eval, parse_analogy_file, most_similar, compositional_add, outlier_detection

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tag = os.path.splitext(os.path.basename(args.config))[0]
    W = np.load(os.path.join(args.outdir, f"W_in_{tag}.npy"))
    vocab = json.load(open(os.path.join(args.outdir, f"vocab_{tag}.json")))
    i2w, w2i = vocab["i2w"], {w: i for i, w in enumerate(vocab["i2w"])}
    qs = parse_analogy_file(cfg.get("analogy_file", "data/questions-words.sample.txt"))
    res = analogy_eval(W, w2i, i2w, qs)
    print(json.dumps(res, indent=2))
    # demos
    for w in ["paris", "king", "profit", "apple"]:
        print(w, "->", most_similar(W, w2i, i2w, w, topk=3))
    print("add(russia,river) ->", compositional_add(W, w2i, i2w, "russia", "river"))
    print("outlier(paris,berlin,rome,apple) ->",
          outlier_detection(W, w2i, i2w, ["paris", "berlin", "rome", "apple"]))
    json.dump(res, open(os.path.join(args.outdir, f"eval_{tag}.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
