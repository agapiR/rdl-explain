"""Text column encoders for building the relational graph.

``GloveTextEmbedding`` is from:
https://github.com/snap-stanford/relbench/blob/main/examples/text_embedder.py
commit SHA 74d4c37acb721659d266b3aa8dbdf23bbf841620

``DistilBertTextEmbedding`` is the encoder used for the paper's experiments
(see the paper, Section 6, "Experiment details"). The two produce different
embedding dimensions (300 vs 768), which changes the feature-encoder input
dimension -- so a graph must be built with the SAME embedder as the checkpoint
that will be loaded onto it, or the state dict will not load.
"""
from typing import List, Optional

import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


def _clean(sentences: List[str]) -> List[str]:
    """RelBench text columns can contain None/NaN; tokenizers expect strings."""
    cleaned: List[str] = []
    for s in sentences:
        if s is None:
            cleaned.append("")
        elif isinstance(s, str):
            cleaned.append(s)
        else:
            # Handle NaN/float/other scalars.
            try:
                if isinstance(s, float) and s != s:  # NaN
                    cleaned.append("")
                else:
                    cleaned.append(str(s))
            except Exception:
                cleaned.append("")
    return cleaned


class DistilBertTextEmbedding:
    """DistilBERT sentence embeddings (768-d), as used in the paper.

    The paper encodes text attributes with
    ``sentence-transformers/distilbert-base-nli-mean-tokens``. Use this when
    rebuilding a graph for a checkpoint trained under that setting.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/distilbert-base-nli-mean-tokens",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(_clean(sentences), convert_to_tensor=True)


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(_clean(sentences), convert_to_tensor=True)
