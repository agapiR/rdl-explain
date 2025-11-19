from pydantic import BaseModel, field_validator, model_validator
from typing import List, Literal
import torch

class ExplainerConfig(BaseModel):
    device: str = "cpu"
    gnn_layers: int
    base_fanout: int = 64  # default

    # Inference
    num_neighbors: List[int] | None = None
    inference_batch_size: int = 128
    temporal_strategy: Literal["uniform", "last", "none"] = "uniform"
    num_workers: int = 0

    @field_validator('device', mode='before')
    @classmethod
    def validate_device(cls, v):
        """Accept both string and torch.device, convert to string"""
        if isinstance(v, torch.device):
            return str(v)
        return v
    
    def get_torch_device(self) -> torch.device:
        """Get the actual torch.device object"""
        return torch.device(self.device)

    @model_validator(mode='after')
    def set_num_neighbors(self):
        if self.num_neighbors is None:
            self.num_neighbors = [
                self.base_fanout // (2**i) for i in range(self.gnn_layers)
            ]
        return self