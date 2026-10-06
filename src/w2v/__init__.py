"""w2v package."""
from .vocab import build_vocab, corpus_from_texts
from .trainer import train_skipgram_or_cbow
from .eval import analogy_eval, parse_analogy_file, most_similar, compositional_add, outlier_detection

__all__ = ["build_vocab", "corpus_from_texts", "train_skipgram_or_cbow",
           "analogy_eval", "parse_analogy_file", "most_similar",
           "compositional_add", "outlier_detection"]
