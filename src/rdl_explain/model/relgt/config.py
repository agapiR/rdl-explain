from pydantic import BaseModel
from typing import Literal


class RelGTConfig(BaseModel):
    local_num_layers: int
    channels: int
    out_channels: int = 1
    global_dim: int = 64
    heads: int = 4
    ff_dropout: float = 0.0
    attn_dropout: float = 0.0
    conv_type: Literal["local", "global", "full"] = "full"
    ablate: Literal["none", "type", "hop", "time", "tfs", "gnn"] = "none"
    gnn_pe_dim: int = 0
    num_centroids: int = 4096
    sample_node_len: int = 100
