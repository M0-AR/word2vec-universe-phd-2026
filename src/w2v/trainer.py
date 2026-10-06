"""SGD trainer with linear LR decay alpha -> min_alpha, dynamic windows, subsampling."""
from __future__ import annotations
import numpy as np
from .models import Word2VecNumpy
from .subsampling import build_keep_probs, subsample_encoded

def train_skipgram_or_cbow(encoded: list[list[int]], counts: list[int], total: int, cfg: dict, verbose: bool = True):
    from .utils import set_seed
    set_seed(cfg.get("seed", 42))
    V = len(counts)
    model = Word2VecNumpy(
        vocab_size=V, dim=cfg.get("dim", 50), sg=cfg.get("sg", 1),
        negative=cfg.get("negative", 5), ns_exponent=cfg.get("ns_exponent", 0.75),
        hs=cfg.get("hs", 0), counts=counts, seed=cfg.get("seed", 42),
        cbow_mean=cfg.get("cbow_mean", 1),
    )
    epochs = cfg.get("epochs", 3)
    alpha0 = cfg.get("alpha", 0.025)
    min_alpha = cfg.get("min_alpha", 0.0001)
    window = cfg.get("window", 5)
    t = cfg.get("sample_t", 1e-3)
    rng = np.random.RandomState(cfg.get("seed", 42) + 999)
    keep = build_keep_probs(counts, total, t) if t and t < 1 else None
    alpha = alpha0
    # total steps estimate for linear decay
    total_tokens = sum(len(s) for s in encoded) * max(epochs, 1)
    step = 0
    losses = []
    for ep in range(epochs):
        data = encoded
        if keep is not None:
            data = subsample_encoded(encoded, keep, rng)
        # shuffle sentence order deterministically
        order = rng.permutation(len(data))
        ep_loss = 0.0
        ep_pairs = 0
        for oi in order:
            sent = data[int(oi)]
            for pos, center in enumerate(sent):
                # linear decay
                frac = step / max(total_tokens, 1)
                alpha = max(min_alpha, alpha0 * (1.0 - frac))
                step += 1
                if model.sg == 1:
                    R = rng.randint(1, window + 1)  # dynamic window
                    lo = max(0, pos - R)
                    hi = min(len(sent), pos + R + 1)
                    ctx = [sent[i] for i in range(lo, hi) if i != pos]
                    if not ctx:
                        continue
                    ep_loss += model._sg_step(center, ctx, alpha)
                    ep_pairs += len(ctx)
                else:
                    N = window
                    lo = max(0, pos - N)
                    hi = min(len(sent), pos + N + 1)
                    ctx = [sent[i] for i in range(lo, hi) if i != pos]
                    if not ctx:
                        continue
                    ep_loss += model._cbow_step(ctx, center, alpha)
                    ep_pairs += 1
        avg = ep_loss / max(ep_pairs, 1)
        losses.append(avg)
        if verbose:
            print(f"[epoch {ep+1}/{epochs}] alpha={alpha:.5f} avg_loss={avg:.4f} pairs={ep_pairs}")
    return model, losses
