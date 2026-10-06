"""Train entry-point. Usage: python scripts/train.py --config configs/tiny.yaml"""
from __future__ import annotations
import argparse, os, sys, yaml, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from w2v.vocab import corpus_from_texts
from w2v.trainer import train_skipgram_or_cbow
from w2v.phrases import iterative_phrases
from w2v.utils import tokenize, set_seed
from w2v.synthetic import SYNTH

def load_corpus(cfg):
    kind = cfg.get("corpus", "synthetic")
    if kind == "synthetic":
        texts = SYNTH[: cfg.get("synthetic_sentences", 2000)]
        sents = [tokenize(t) for t in texts]
        return sents
    if kind == "text8":
        from scripts.download_data import ensure_text8
        path = ensure_text8()
        with open(path, encoding="utf-8") as f:
            raw = f.read().lower().split()
        # chunk into sentences of 40 tokens
        return [raw[i:i+40] for i in range(0, min(len(raw), 200000), 40)]
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    sents = load_corpus(cfg)
    # phrase pass (light; threshold high for tiny corpora)
    if cfg.get("phrases", False):
        sents = iterative_phrases(sents)
    vocab, encoded = corpus_from_texts([" ".join(s) for s in sents], min_count=cfg.get("min_count", 2))
    print(f"vocab={len(vocab)} sents={len(encoded)} total_tokens={vocab.total}")
    model, losses = train_skipgram_or_cbow(encoded, vocab.counts, vocab.total, cfg)
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.config))[0]
    np.save(os.path.join(args.outdir, f"W_in_{tag}.npy"), model.W_in)
    np.save(os.path.join(args.outdir, f"W_out_{tag}.npy"), model.W_out)
    json.dump({"i2w": vocab.i2w, "counts": vocab.counts},
              open(os.path.join(args.outdir, f"vocab_{tag}.json"), "w"))
    json.dump({"losses": losses, "config": cfg},
              open(os.path.join(args.outdir, f"train_{tag}.json"), "w"), indent=2)
    print(f"saved to {args.outdir}/ (tag={tag}) final_loss={losses[-1]:.4f}")

if __name__ == "__main__":
    main()
