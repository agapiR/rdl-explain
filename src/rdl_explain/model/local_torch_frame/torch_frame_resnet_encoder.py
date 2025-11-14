"""
ResNet encoder for tabular data using TorchFrame.

Adapted from:
    PyTorch Frame library
    GitHub: https://github.com/pyg-team/pytorch-frame
    Commit: c1f0cb8ffe1352db8e80c09b2c59a06d00c899b9
    File: torch_frame/nn/models/resnet.py
    Original authors: PyG team
    MIT License

Modifications by:
    Agapi Rissaki, 2025

Description of changes:
- Added support for explainability via masking.
- Added support for intermediate encoding extraction.

Notes:
- Code is based on the above commit to ensure reproducibility.
- All original functionality retained unless explicitly modified.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

from torch_geometric.typing import NodeType

from rdl_explain.explain.explain_utils import eliminate


class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`layer_norm`, :obj:`batch_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        self.norm1: BatchNorm1d | LayerNorm | None
        self.norm2: BatchNorm1d | LayerNorm | None
        if normalization == "batch_norm":
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == "layer_norm":
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        self.shortcut: Linear | None
        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            x = self.shortcut(x)

        out = out + x

        return out


class ResNet(Module):
    r"""The ResNet model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using ResNet, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(dict[str,Dict[:class:`torch_frame.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batch_norm`, :obj:`layer_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoding_channels = channels

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])
        in_channels = channels * num_cols

        if num_layers:
            self.backbone = Sequential(*[
                FCResidualBlock(
                    in_channels if i == 0 else channels,
                    channels,
                    normalization=normalization,
                    dropout_prob=dropout_prob,
                ) for i in range(num_layers)
            ])

            self.decoder = Sequential(
                LayerNorm(channels),
                ReLU(),
                Linear(channels, out_channels),
            )
        else:
            self.backbone = Sequential()
            self.decoder = Sequential(
                LayerNorm(in_channels),
                ReLU(),
                Linear(in_channels, out_channels),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        x = self.backbone(x)
        out = self.decoder(x)
        return out

    def forward_to_explain(
        self, 
        tf: TensorFrame, 
        mask: Union[Dict[str, Tensor], Tensor],
        mask_type: str = 'column',
        elimination_strategy: str = 'zero',
        uninformative_feat_vector: Optional[Tensor] = None,
    ) -> Tensor:

        x, col_names = self.encoder(tf)

        batch_size, num_cols, _ = x.shape

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        if mask_type == 'column':
            # Concatenate mask values in a single tensor, following the order of the columns in col_names
            mask_values = torch.cat([mask[col_name] for col_name in col_names], dim=0)
            # Repeat mask values for each row in the batch
            mask_values = mask_values.repeat(batch_size).reshape(batch_size, num_cols)
            # Repeat mask values for each encoding channel
            mask_values = mask_values.repeat_interleave(self.encoding_channels, dim=1)
        elif mask_type == 'table':
            # In case of table masks, there is a single mask value for all columns and all rows
            mask_values = mask
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")

        # Apply mask to the encoding, by element-wise multiplication
        x = eliminate(x, mask_values, strategy=elimination_strategy, x_default=uninformative_feat_vector)

        # Decoding
        x = self.backbone(x)
        out = self.decoder(x)
        return out

    def get_intermediate_encoding(self, tf: TensorFrame) -> Tuple[Tuple[Tensor, List[str]], Tensor]:
        r"""Transforming :class:`TensorFrame` object into intermediate encoding.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Intermediate encoding of shape [batch_size, channels, num_cols].
        """
        x, col_names = self.encoder(tf)

        batch_size, num_cols, _ = x.shape

        # Flattening the encoder output
        intermediate_enc = x.view(x.size(0), math.prod(x.shape[1:]))

        # Decoding
        x = self.backbone(intermediate_enc)
        fused_enc = self.decoder(x)
        
        return (intermediate_enc, col_names), fused_enc