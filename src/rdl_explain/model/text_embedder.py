"""
From: https://github.com/snap-stanford/relbench/blob/main/examples/text_embedder.py
commit SHA 74d4c37acb721659d266b3aa8dbdf23bbf841620
"""
from typing import List, Optional

import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        # Some RelBench datasets can contain missing values in text columns.
        # SentenceTransformers tokenizers expect strings.
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
        return self.model.encode(cleaned, convert_to_tensor=True)
