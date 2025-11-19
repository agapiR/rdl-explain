from pydantic import BaseModel, Field
from typing import List, Literal


class ModelConfig(BaseModel):
    gnn_layers: int
    channels: int
    out_channels: int = 1
    aggr: Literal["mean", "sum", "max"] = "mean" 
    norm: Literal["batch_norm", "none"] = "batch_norm"
    shallow_list: List[str] = []    # List of node types to add shallow embeddings to input
    id_awareness: bool = False      # ID awareness