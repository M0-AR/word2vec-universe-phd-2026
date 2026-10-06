# Zero → Hero: learn word2vec by breaking it (15 min)

1. `python scripts/train.py --config configs/tiny.yaml` — watch loss 4.1→2.5. Change `sg:1→0`. CBOW loss higher? Why? (Hint: CBOW averages contexts; SG gets R×2 labels per token.)
2. `python scripts/evaluate.py --config configs/tiny.yaml` — `add(russia,river)→volga 0.98`. Now try `king-man+woman`. Fails on tiny? That's coverage, not intelligence.
3. `python scripts/benchmark_grid.py --config configs/tiny.yaml` — window 2 beats 5 on syntax. Dim 50 beats 25 only for sg=1/win=2. Interaction > main effect.
4. Open `src/w2v/subsampling.py`: set `sample_t: 1.0` (off) vs `1e-5`. Rare-word neighbours degrade. That's the subsidy.
5. `python scripts/phrase_detect.py` — lower threshold to `1e-6`: garbage phrases (`is_a`) appear. Threshold is precision/recall.
6. `python scripts/live_verify.py --config configs/finance_live.yaml` — generic cosines 0.99 (collapsed) vs finance 0.02–0.28 (spread). Hubness is visible to the naked eye.
Next: text8 (`configs/small.yaml`), full questions-words.txt (19544), Gensim baseline, walk-forward SPY backtest.
