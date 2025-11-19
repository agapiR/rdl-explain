"""
Model definition for relational deep learning on heterogeneous temporal graphs.

Adapted from:
    RelBench
    GitHub: https://github.com/snap-stanford/relbench
    Commit: f4fdd49f8348e8e83fa9a498567611928ea36528
    File: relbench/examples/model.py
    Original authors: Stanford SNAP Lab
    MIT License

Modifications by:
    Agapi Rissaki, 2025

Description of changes:
- Modified imports to use local neural network components.
- Added method `forward_to_explain` for explanation via masking.
- Added method `get_intermediate_encoding` to retrieve intermediate encodings.

Notes:
- Code is based on the above commit to ensure reproducibility.
- All original functionality retained unless explicitly modified.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

# Replace with local imports
# from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from rdl_explain.model.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE

class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_kwargs={"channels": channels, "num_layers": 2}
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])

    def forward_to_explain(
        self,
        explanation_type: str,
        mask_dict: Dict,
        batch: HeteroData,
        entity_table: NodeType,
        elimination_strategy: str = 'zero',
        uninformative_feat_vector: Optional[Dict[NodeType, Tensor]] = None,
    ) -> Tensor:
        """
        Forward method used for explanations via masking.

        Args:
            explanation_type (str): Type of explanation ('table', 'column', etc.).
            mask_dict (Dict): Dictionary containing masks.
            batch (HeteroData): Input batch data.
            entity_table (NodeType): The main entity node type.
            elimination_strategy (str): Strategy for feature elimination via masking.
            uninformative_feat_vector (Optional[Dict[NodeType, Tensor]]): Predefined uninformative feature vectors.

        Returns:
            Tensor: Model output after applying explanation masks.
        """

        if explanation_type not in ['table', 'column']:
            raise NotImplementedError(
                f"Explanation type {explanation_type} not implemented."
            )

        seed_time = batch[entity_table].seed_time

        # Apply activation to mask
        mask = {key: value.sigmoid() for key, value in mask_dict.items()}

        # Apply masks to features, for 'table' and 'column' explanations
        if explanation_type == 'table' or explanation_type == 'column':
            # Get initial node features from encoder, applying table / column masking
            x_dict = self.encoder.forward_to_explain(batch.tf_dict, mask, mask_type=explanation_type, elimination_strategy=elimination_strategy, uninformative_feat_vector=uninformative_feat_vector)
        else:
            # Fallback to standard encoding otherwise
            x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def get_intermediate_encoding(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tuple[Tuple[Tensor, List[str]], Tensor]:
        """
        Get intermediate column-wise and fused encoding from the encoder, for a given node type.

        Args:
            batch: Heterogeneous graph batch data
            entity_table: The entity node type to get encoding for

        Returns:
            A tuple containing:
                - A tuple of (intermediate_enc, col_names):
                    - intermediate_enc: Tensor of shape [num_nodes, num_columns * channels]
                    - col_names: List of column names corresponding to the intermediate_enc
                - fused_enc: Tensor of shape [num_nodes, channels] representing fused encoding
        """

        (intermediate_enc, col_names), fused_enc = self.encoder.get_intermediate_encoding(batch.tf_dict, entity_table)
        return (intermediate_enc, col_names), fused_enc