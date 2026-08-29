import os
import time
import copy
import math
import numpy as np
from typing import Any, Dict, NamedTuple, Optional, Tuple, List, Union
from collections import defaultdict

import torch
from torch import Tensor
import torch_frame
from torch_frame.data import MultiNestedTensor, MultiEmbeddingTensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType, EdgeType

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Relbench imports
from relbench.base import Database

def node_type_to_col_names_by_stype(graph):
    return {node_type: graph[node_type].tf.col_names_dict for node_type in graph.node_types}

def node_type_to_col_names(graph):
    return {node_type: [col for cols_per_stype in list(graph[node_type].tf.col_names_dict.values()) for col in cols_per_stype] for node_type in graph.node_types}

def perturb_instance(
    instance: HeteroData,
    mask: Dict,
    mask_type: Union[str, List[str]],
    perturbation_type: Union[str, List[str]],
    ignore_blocked_edges: bool = True,
    verbose: bool = False,
) -> HeteroData:
    """Perturb a HeteroData graph in-place by randomly shuffling the values of
    masked elements, leaving retained elements unchanged.

    **In-place modification:** this function modifies `instance` directly. The
    caller is responsible for saving and restoring any state that must be
    preserved across calls.

    Supports three mask types, which can be combined by passing lists.
    When multiple mask types are provided they are applied in fixed order:
    fkpk → column → row.

    Mask type 'column' — shuffle column values across rows:
        mask: {(node_type, col_name): bool_tensor}
            True  = retained, column values are kept as-is.
            False = perturbed, column values are randomly permuted across rows.
        IMPORTANT: mask values must be **bool tensors** (torch.bool), not Python
        booleans. Python's ~ operator on True/False returns non-zero integers
        (both truthy), so ~True and ~False would cause every column to be
        permuted regardless of the intended mask value.
        The mask must contain an entry for every (node_type, col_name) pair in
        the graph; missing keys raise a KeyError at runtime.
        Perturbation types:
            'permutation_joint'       — all perturbed columns in a table share the
                                        same random row permutation.
            'permutation_independent' — each perturbed column gets its own
                                        independent random row permutation.

    Mask type 'row' — shuffle entire rows for masked nodes:
        mask: {node_type: bool_tensor of shape (n_rows,)}
            True = retained, False = perturbed (row permuted among other perturbed rows).
        Perturbation types: 'permutation_joint', 'permutation_independent'.

    Mask type 'fkpk' — shuffle foreign-key assignments for masked join edges:
        mask: {edge_type: bool_tensor}
            True = retained, False = perturbed (FK reassigned).
        Perturbation types: 'foreign_key_permutation', 'foreign_key_exchange',
            'foreign_key_uniform_random', 'foreign_key_hist_random'.

    When combining multiple mask types, pass a dict of dicts keyed by mask_type
    string, e.g. {'column': col_mask, 'fkpk': fk_mask}.

    Args:
        instance: HeteroData graph to perturb (modified in-place).
        mask: Mask dictionary (format depends on mask_type, see above).
        mask_type: 'column', 'row', 'fkpk', or a list of these.
        perturbation_type: Perturbation strategy string, or a list matching mask_type.
        ignore_blocked_edges: (fkpk only) Skip edge types where neither endpoint
            participates in any retained edge; avoids perturbing signal-less paths.
        verbose: If True, print progress and timing information.

    Returns:
        The perturbed instance (same object as input).
    """
    if isinstance(mask_type, str):
        mask_type_list = [mask_type]
        perturbation_type_list = [perturbation_type]
        mask_dict = {mask_type: mask}
    elif isinstance(mask_type, list):
        mask_type_list = mask_type
        perturbation_type_list = perturbation_type
        mask_dict = mask
    else:
        raise ValueError(f"Invalid mask_type ({mask_type}). Expected str or list of str, got {type(mask_type)}.")

    # Enforce application order: fkpk → column → row
    mask_type_order = {'fkpk': 0, 'column': 1, 'row': 2}
    order = sorted(range(len(mask_type_list)), key=lambda i: mask_type_order.get(mask_type_list[i], float('inf')))
    mask_type_list = [mask_type_list[i] for i in order]
    perturbation_type_list = [perturbation_type_list[i] for i in order]

    # The function operates directly on `instance` (no copy is made).
    perturbed_instance = instance

    supported_stypes = [torch_frame.numerical, torch_frame.categorical, torch_frame.embedding, torch_frame.timestamp]
    node_type_to_col_names_by_stype_dict = node_type_to_col_names_by_stype(instance)

    if verbose:
        start_time = time.time()

    for mask_type, perturbation_type in zip(mask_type_list, perturbation_type_list):
        if mask_type not in ('column', 'row', 'fkpk'):
            raise ValueError(f"Invalid mask type '{mask_type}'. Supported: 'column', 'row', 'fkpk'.")
        if mask_type not in mask_dict:
            raise ValueError(f"Mask type '{mask_type}' not found in mask dict. Available: {list(mask_dict.keys())}.")
        mask = mask_dict[mask_type]

        for node_type, col_names_by_stype in node_type_to_col_names_by_stype_dict.items():
            for st, _ in col_names_by_stype.items():
                if st not in supported_stypes:
                    raise ValueError(f"Unsupported stype {st}. Supported: {supported_stypes}.")

        if mask_type == 'fkpk' and perturbation_type not in (
            'foreign_key_permutation', 'foreign_key_exchange',
            'foreign_key_uniform_random', 'foreign_key_hist_random',
        ):
            raise ValueError(f"Invalid perturbation type '{perturbation_type}' for 'fkpk'.")
        if mask_type in ('row', 'column') and perturbation_type not in ('permutation_joint', 'permutation_independent'):
            raise ValueError(f"Invalid perturbation type '{perturbation_type}' for '{mask_type}'.")

        if mask_type == 'column':
            if perturbation_type == 'permutation_joint':
                for tab in instance.node_types:
                    perm = torch.randperm(instance[tab].tf.num_rows)
                    for st, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if st in (torch_frame.numerical, torch_frame.categorical, torch_frame.timestamp):
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    perturbed_instance[tab].tf.feat_dict[st][:, col_idx] = \
                                        instance[tab].tf.feat_dict[st][:, col_idx][perm]
                        elif st == torch_frame.embedding:
                            emb_list = []
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    emb_list.append(instance[tab].tf.feat_dict[st][:, col_idx].values[perm])
                                else:
                                    emb_list.append(perturbed_instance[tab].tf.feat_dict[st][:, col_idx].values)
                            perturbed_instance[tab].tf.feat_dict[st] = MultiEmbeddingTensor.from_tensor_list(emb_list)

            elif perturbation_type == 'permutation_independent':
                for tab in instance.node_types:
                    n_rows = instance[tab].tf.num_rows
                    for st, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if st in (torch_frame.numerical, torch_frame.categorical, torch_frame.timestamp):
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    perm = torch.randperm(n_rows)
                                    perturbed_instance[tab].tf.feat_dict[st][:, col_idx] = \
                                        instance[tab].tf.feat_dict[st][:, col_idx][perm]
                        elif st == torch_frame.embedding:
                            emb_list = []
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    perm = torch.randperm(n_rows)
                                    emb_list.append(instance[tab].tf.feat_dict[st][:, col_idx].values[perm])
                                else:
                                    emb_list.append(perturbed_instance[tab].tf.feat_dict[st][:, col_idx].values)
                            perturbed_instance[tab].tf.feat_dict[st] = MultiEmbeddingTensor.from_tensor_list(emb_list)
            else:
                raise ValueError(f"Invalid perturbation type '{perturbation_type}' for 'column'.")

        elif mask_type == 'row':

            def get_row_permutation_for_masked_rows(n_rows, row_mask):
                """Permute only the perturbed rows among themselves; retained rows stay in place."""
                rows_to_permute = torch.where(~row_mask)[0]
                permutation = rows_to_permute[torch.randperm(len(rows_to_permute))]
                global_indices = torch.arange(n_rows)
                global_indices[rows_to_permute] = global_indices[permutation]
                return global_indices

            if perturbation_type == 'permutation_joint':
                for tab in instance.node_types:
                    perm = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                    for st, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if st in (torch_frame.numerical, torch_frame.categorical, torch_frame.timestamp):
                            perturbed_instance[tab].tf.feat_dict[st] = instance[tab].tf.feat_dict[st][perm]
                        elif st == torch_frame.embedding:
                            emb_list = [instance[tab].tf.feat_dict[st][:, col_idx].values[perm]
                                        for col_idx in range(len(col_names))]
                            perturbed_instance[tab].tf.feat_dict[st] = MultiEmbeddingTensor.from_tensor_list(emb_list)

            elif perturbation_type == 'permutation_independent':
                for tab in instance.node_types:
                    for st, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if st in (torch_frame.numerical, torch_frame.categorical, torch_frame.timestamp):
                            for col_idx in range(len(col_names)):
                                perm = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                                perturbed_instance[tab].tf.feat_dict[st][:, col_idx] = \
                                    instance[tab].tf.feat_dict[st][:, col_idx][perm]
                        elif st == torch_frame.embedding:
                            emb_list = []
                            for col_idx in range(len(col_names)):
                                perm = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                                emb_list.append(instance[tab].tf.feat_dict[st][:, col_idx].values[perm])
                            perturbed_instance[tab].tf.feat_dict[st] = MultiEmbeddingTensor.from_tensor_list(emb_list)
            else:
                raise ValueError(f"Invalid perturbation type '{perturbation_type}' for 'row'.")

        elif mask_type == 'fkpk':
            # Identify node types adjacent to at least one retained edge.
            # Edge types where neither endpoint is in this set carry no model signal
            # and can be skipped to avoid unnecessary perturbation.
            unmasked_nodes = set()
            for edge_type in instance.edge_types:
                if mask[edge_type]:
                    src, _, dst = edge_type
                    unmasked_nodes.update([src, dst])
            if not unmasked_nodes:
                ignore_blocked_edges = False
            elif verbose:
                print(f"Retained nodes (adjacent to retained edges): {unmasked_nodes}.")

            # Edges and their reverses are always perturbed symmetrically.
            # Convention: (src, 'name', dst) ↔ (dst, 'rev_name', src).
            perturbed_fk_pairs = set()
            for edge_type in instance.edge_types:
                src, edge_name, dst = edge_type
                if ignore_blocked_edges and src not in unmasked_nodes and dst not in unmasked_nodes:
                    if verbose:
                        print(f"Skipping {edge_type}: neither endpoint is adjacent to a retained edge.")
                    continue
                rev_edge_type = (dst, edge_name.replace('rev_', ''), src) if 'rev_' in edge_name \
                                else (dst, 'rev_' + edge_name, src)
                assert rev_edge_type in instance.edge_types, \
                    f"Reverse edge type {rev_edge_type} not found. Available: {instance.edge_types}."
                if edge_type in perturbed_fk_pairs or rev_edge_type in perturbed_fk_pairs:
                    continue
                if ~mask[edge_type] and ~mask[rev_edge_type]:
                    num_edges = instance.edge_index_dict[edge_type].shape[1]
                    num_unique_dst = instance.edge_index_dict[edge_type][1].unique().shape[0]
                    fk = 0 if num_unique_dst < num_edges else 1
                    rev_fk = 1 - fk
                    perturbed_instance[edge_type].edge_index = instance.edge_index_dict[edge_type]
                    perturbed_instance[rev_edge_type].edge_index = instance.edge_index_dict[edge_type][[1, 0], :]
                    assert torch.all(
                        perturbed_instance.edge_index_dict[edge_type][fk] ==
                        perturbed_instance.edge_index_dict[rev_edge_type][rev_fk]
                    ), f"Edge {edge_type} and its reverse are not stored symmetrically."

                    if perturbation_type == 'foreign_key_permutation':
                        perm = torch.randperm(num_edges)
                        perturbed_instance[edge_type].edge_index[fk] = \
                            perturbed_instance.edge_index_dict[edge_type][fk][perm]
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = \
                            perturbed_instance.edge_index_dict[rev_edge_type][rev_fk][perm]

                    elif perturbation_type == 'foreign_key_exchange':
                        # Permute the set of unique FK values; each original FK is replaced
                        # by a different FK, preserving the per-entity degree distribution.
                        unique_fks = perturbed_instance.edge_index_dict[edge_type][fk].unique().numpy()
                        unique_fks_permuted = unique_fks[np.random.permutation(len(unique_fks))]
                        for i, unique_fk in enumerate(unique_fks):
                            fk_indices = (perturbed_instance.edge_index_dict[edge_type][fk] == unique_fk).nonzero(as_tuple=True)[0]
                            if len(fk_indices) > 0:
                                perturbed_instance[edge_type].edge_index[fk][fk_indices] = unique_fks_permuted[i]
                                perturbed_instance[rev_edge_type].edge_index[rev_fk][fk_indices] = unique_fks_permuted[i]

                    elif perturbation_type == 'foreign_key_uniform_random':
                        unique_fks = perturbed_instance.edge_index_dict[edge_type][fk].unique().numpy()
                        random_fks = np.random.choice(unique_fks, size=num_edges, replace=True)
                        perturbed_instance[edge_type].edge_index[fk] = torch.tensor(random_fks, dtype=torch.long)
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = torch.tensor(random_fks, dtype=torch.long)

                    elif perturbation_type == 'foreign_key_hist_random':
                        # Preserve each FK's frequency but shuffle which FK gets which count.
                        unique_fks, counts = perturbed_instance.edge_index_dict[edge_type][fk].unique(return_counts=True)
                        shuffled_counts = np.random.permutation(counts.numpy())
                        random_fks = np.random.choice(unique_fks, size=num_edges, replace=True,
                                                       p=shuffled_counts / shuffled_counts.sum())
                        perturbed_instance[edge_type].edge_index[fk] = torch.tensor(random_fks, dtype=torch.long)
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = torch.tensor(random_fks, dtype=torch.long)
                    else:
                        raise ValueError(f"Invalid perturbation type '{perturbation_type}' for 'fkpk'.")

                    perturbed_fk_pairs.update([edge_type, rev_edge_type])
        else:
            raise ValueError(f"Invalid mask type '{mask_type}'. Supported: 'column', 'row', 'fkpk'.")

    if verbose:
        elapsed = time.time() - start_time
        print(f"Perturbation complete in {elapsed:.2f}s "
              f"(mask_types={mask_type_list}, perturbation_types={perturbation_type_list}).")

    return perturbed_instance

def eliminate(
    x_input: Tensor, 
    mask: Tensor,
    strategy: str = 'zero',
    x_default: Tensor = None,
) -> Tensor:
    """
    Eliminate features based on the given mask.
    Args:
        x_input (Tensor): Input tensor of shape (n_rows, n_feat_input).
        mask (Tensor): Mask tensor of shape (n_rows, n_feat_input).
        strategy (str): Elimination strategy. Options are:
            - 'zero':                       Set masked features to zero.
            - 'default':                    Replace masked features with default values.
                                            Default values are provided in the x_default tensor (n_feat_default,).
            - 'default_w_perturbation':     Replace masked features with perturbed default values. 
                                            Default values are provided in the x_default tensor (n_feat_default,).
                                            Perturbation is done by adding Gaussian noise with variance equal to the standard deviation of the input features.
            - 'batch_avg':                  Replace features with batch average values.
            - 'batch_avg_w_perturbation':   Replace features with perturbed batch average values. 
                                            Perturbation is done by adding Gaussian noise with variance equal to the standard deviation of the input features.
            - 'permutation_joint':          Replace features with a random permutation of the replacement features, performed jointly for all features dimensions.
                                            Replacement feature vectors are provided in the x_default tensor (n_options, n_feat_default).
            - 'permutation_independent':    Replace features with a random permutation of the provided features, independently for each feature dimension.
                                            Replacement feature vectors are provided in the x_default tensor (n_options, n_feat_default).
            - 'batch_permutation_joint':     Replace features with a random permutation of the batch, performed jointly for all features dimensions.
            - 'batch_permutation_independent': Replace features with a random permutation of the batch, independently for each feature dimension.
        x_default (Tensor): Default value tensor of shape (n_feat_default,) or (n_options, n_feat_default).
    """
    if strategy == 'zero':
        x_output = x_input * mask
    elif strategy == 'default':
        n_rows, n_feat_input = x_input.shape
        n_feat_default = x_default.shape[0]
        assert n_feat_input == n_feat_default, f"Input and default feature dimensions must match. Got input {n_feat_input} and default {n_feat_default}."
        assert x_default is not None, "Default value (, n_feat) must be provided for the 'default' elimination strategy."
        x_output = x_input * mask + x_default * (1 - mask)
    elif strategy == 'default_w_perturbation':
        n_rows, n_feat_input = x_input.shape
        n_feat_default = x_default.shape[0]
        assert n_feat_input == n_feat_default, f"Input and default feature dimensions must match. Got input {n_feat_input} and default {n_feat_default}."
        assert x_default is not None, "Default value (, n_feat) must be provided for the 'default_w_sampling' elimination strategy."
        x_std_row = x_input.std(dim=0).expand(n_rows, n_feat_input)
        x_output = x_input * mask + torch.normal(mean=x_default, std=x_std_row) * (1 - mask)
    elif strategy == 'batch_avg':
        n_rows, n_feat_input = x_input.shape
        x_avg_row = x_input.mean(dim=0).expand(n_rows, n_feat_input)
        x_avg = x_avg_row.expand(n_rows, n_feat_input)
        x_output = x_input * mask + x_avg * (1 - mask)
    elif strategy == 'batch_avg_w_perturbation':
        n_rows, n_feat_input = x_input.shape
        x_avg_row = x_input.mean(dim=0).expand(n_rows, n_feat_input)
        x_std_row = x_input.std(dim=0).expand(n_rows, n_feat_input)
        x_avg = torch.normal(mean=x_avg_row, std=x_std_row)
        x_output = x_input * mask + x_avg * (1 - mask)
    elif strategy == 'batch_permutation_joint':
        n_rows, n_feat_input = x_input.shape
        x_input_permuted = x_input.clone()
        x_input_permuted = x_input_permuted[torch.randperm(n_rows), :]
        x_output = x_input * mask + x_input_permuted * (1 - mask)
    elif strategy == 'batch_permutation_independent':
        n_rows, n_feat_input = x_input.shape
        x_input_permuted = x_input.clone()
        for i in range(n_feat_input):
            x_input_permuted[:, i] = x_input_permuted[torch.randperm(n_rows), i]  
        x_output = x_input * mask + x_input_permuted * (1 - mask)
    elif strategy == 'permutation_joint':
        n_rows, n_feat_input = x_input.shape
        n_options, n_feat_input = x_default.shape
        assert n_feat_input == n_feat_input, f"Input and default feature dimensions must match. Got input {n_feat_input} and default {n_feat_input}."
        assert x_default is not None, "Default value (n_options, n_feat) must be provided for the 'permutation_joint' elimination strategy."
        x_default_permuted = x_default[torch.randperm(n_options), :]
        # when more replacement vectors than original vectors, limit to n_rows
        if x_default.shape[0] > n_rows:
            x_default = x_default_permuted[:n_rows, :] 
        # when less replacement vectors than original vectors, expand the replacement vectors by repetition until n_rows is reached
        elif x_default.shape[0] < n_rows: 
            x_default = x_default_permuted.repeat((n_rows // x_default_permuted.shape[0] + 1, 1))[:n_rows, :]
        else:
            x_default = x_default_permuted
        x_output = x_input * mask + x_default * (1 - mask)
    elif strategy == 'permutation_independent':
        n_rows, n_feat_input = x_input.shape
        n_options, n_feat_input = x_default.shape
        assert n_feat_input == n_feat_input, f"Input and default feature dimensions must match. Got input {n_feat_input} and default {n_feat_input}."
        assert x_default is not None, "Default value (n_options, n_feat) must be provided for the 'permutation_joint' elimination strategy."
        x_default_permuted = x_default.clone()
        # permute each dimension of the replacement vectors independently
        for i in range(n_feat_input): 
            x_default_permuted[:, i] = x_default[torch.randperm(n_options), i]
        # when more replacement vectors than original vectors, limit to n_rows
        if x_default_permuted.shape[0] > n_rows: 
            x_default = x_default_permuted[:n_rows, :]
        # when less replacement vectors than original vectors, expand the replacement vectors by repetition until n_rows is reached
        elif x_default_permuted.shape[0] < n_rows: 
            x_default = x_default_permuted.repeat((n_rows // x_default_permuted.shape[0] + 1, 1))[:n_rows, :]
        else:
            x_default = x_default_permuted
        x_output = x_input * mask + x_default * (1 - mask)
    else:
        raise ValueError(f"Invalid elimination strategy: {strategy}")
    return x_output

def make_schema_graph(
    database: Database,
    directed: bool = True,
    self_loop: bool = False,
) -> Dict[str, List]:
    schema_graph = defaultdict(list)
    for table_name, table in database.table_dict.items():
        # Add self-loop if required
        if self_loop:
            schema_graph[table_name].append({'dst': table_name, 'edge_name': 'self', 'edge_type': '1:1'})
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            schema_graph[table_name].append({'dst': pkey_table, 'edge_name': 'f2p_'+fkey_col, 'edge_type': 'N:1'})
            if not directed:
                schema_graph[pkey_table].append({'dst': table_name, 'edge_name': 'rev_f2p_'+fkey_col, 'edge_type': '1:N'})
    return schema_graph

def make_schema_dag(
    schema_graph: Dict[str, List],
    depth: int,
    source_entity: str,
    layer_specific_node_type: bool = True,
    avoid_backtracking: bool = True,
) -> Dict[str, List]:
    
    schema_DAG = {}
    queue = [(source_entity, 0)]
    visited = set()

    if not layer_specific_node_type:
        while queue:
            # Remove the first table from the queue
            table, table_depth = queue.pop(0)
            # If table is not visited, visit the table and expand to its neighbor entities
            if table not in visited:
                schema_DAG[table] = []
                visited.add(table)
                # If maximum DAG depth is reached, do not expand to neighbors
                if table_depth >= depth:
                    continue
                else:
                    for neighbor_dict in schema_graph[table]:
                        neighbor = neighbor_dict['dst']
                        # Add neighbor entity to queue
                        queue.append((neighbor, table_depth + 1))
                        # Add edge to schema DAG
                        schema_DAG[table].append(neighbor)
    else:
        while queue:
            # Remove the first table from the queue
            table, table_depth = queue.pop(0)
            # If table is not visited in this layer, visit the table and expand to its neighbor entities
            if (table, table_depth) not in visited:
                schema_DAG[(table, table_depth)] = []
                visited.add((table, table_depth))
                # If maximum DAG depth is reached, do not expand to neighbors
                if table_depth >= depth:
                    continue
                else:
                    for neighbor_dict in schema_graph[table]:
                        neighbor = neighbor_dict['dst']
                        # Avoid backtracking to the same neighbor entity in the previous layer
                        if avoid_backtracking and (neighbor, table_depth - 1) in schema_DAG:
                            # If the edge type to be added is many-to-one or one-to-one, avoid backtracking to the same entity instance
                            if neighbor_dict['edge_type'] == 'N:1' or neighbor_dict['edge_type'] == '1:1':
                                continue
                        # Add neighbor entity to queue
                        queue.append((neighbor, table_depth + 1))
                        # Add edge to schema DAG
                        schema_DAG[(table, table_depth)].append({'dst': (neighbor, table_depth + 1), 'edge_name': neighbor_dict['edge_name'], 'edge_type': neighbor_dict['edge_type']})

        # If layer-specific node type is enabled, convert table_depth from 0 to depth-1 to table_depth from depth-1 to 0
        if layer_specific_node_type:
            schema_DAG_reverse_count = {}
            for (table, table_depth), values in schema_DAG.items():
                schema_DAG_reverse_count[(table, depth - table_depth)] = []
                for value in values:
                    schema_DAG_reverse_count[(table, depth - table_depth)].append({'dst': (value['dst'][0], depth - value['dst'][1]), 'edge_name': value['edge_name'], 'edge_type': value['edge_type']})
        
        schema_DAG = schema_DAG_reverse_count

    return schema_DAG

def draw_schema_dag(
    schema_DAG: Dict[str, List], 
    save_path: Optional[str] = None,
) -> None:
    
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in schema_DAG.keys():
        e, l = node
        G.add_node(node, layer=l, name=e)

    # Add edges to the graph
    for src, edges in schema_DAG.items():
        for edge in edges:
            dst = edge['dst']
            G.add_edge(dst, src, edge_name=edge['edge_name'], edge_type=edge['edge_type'])

    # Draw the DAG (requires matplotlib)
    plt.figure(figsize=(10, 6))
    # pos: order according to the DAG level
    pos = nx.multipartite_layout(G, subset_key="layer", align='vertical')
    # nx.draw(G, pos, with_labels=True, node_size=5000, node_color='lightblue', edge_color='gray', font_size=8)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray', font_size=8, font_weight='bold', style='solid', arrowstyle='->', arrowsize=3)
    edge_labels = {(u, v): d['edge_name'] for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches='tight')
    else:
        plt.show()

def explanation_element_wording(k: Any) -> str:
    """
    Convert the mask key to a more readable format.
    """
    if isinstance(k, tuple):
        if len(k) == 2:
            if isinstance(k[0], str):
                # (node_type, col_name) or (node_type, layer)
                if isinstance(k[1], str):
                    return f"{k[0]}-{k[1]}"
                elif isinstance(k[1], int):
                    return f"{k[0]}(layer={k[1]})"
            elif isinstance(k[0], tuple):
                # (edge_type, layer)
                edge_type_label = '-'.join(k[0])
                return f"{edge_type_label}(layer={k[1]})"
            else:
                return '-'.join(map(str, k))
        else:
            # edge_type / filter
            return '-'.join(map(str, k))
    elif isinstance(k, str):
        # node_type
        return k
    else:
        return str(k)


def _subsample_balanced(df, target_col, n, rng):
    """Pick at most n rows from df, balanced 50/50 by binary target_col when
    possible. If either class has fewer than n//2 rows we take all of them
    and top-up from the other class. Returns a row-shuffled copy."""
    n_per_class = n // 2
    pos = df[df[target_col] > 0]
    neg = df[df[target_col] <= 0]
    pos_take = min(len(pos), n_per_class)
    neg_take = min(len(neg), n_per_class)
    # if one side is short, try to make up the difference from the other
    shortfall = (n_per_class - pos_take) + (n_per_class - neg_take)
    if shortfall > 0:
        if len(pos) > pos_take:
            extra = min(len(pos) - pos_take, shortfall)
            pos_take += extra; shortfall -= extra
        if shortfall > 0 and len(neg) > neg_take:
            extra = min(len(neg) - neg_take, shortfall)
            neg_take += extra
    pos_sample = pos.sample(n=pos_take, random_state=rng) if pos_take else pos.iloc[:0]
    neg_sample = neg.sample(n=neg_take, random_state=rng) if neg_take else neg.iloc[:0]
    sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def make_explanation_task(prediction_task, data, inference_dir=None,
                          explanation_target_type='soft',
                          subsample_per_split=None,
                          subsample_balanced=True,
                          subsample_seed=42,
                          predictions=None):
    """Build the *explanation task*: a clone of the model's prediction task with
    the MODEL'S OWN predictions as the target. Mask learning and devΔ evaluation
    therefore measure agreement with the *model*, not the labels — they are
    label-free (original labels are kept in a ``'targets'`` column).

    Splits. Keeps the prediction task's ``train`` / ``val`` / ``test`` (the same
    entity instances), each now carrying the model's predictions. By default the
    pipeline runs entirely on the **train** split: mask learning and devΔ
    evaluation use the *same* (optionally subsampled) train instances. Evaluation
    can be redirected to a different split only by passing it explicitly.

    ``subsample_per_split`` keeps a random (optionally class-balanced) subset of N
    instances per split — and that one subset is what both learning and
    evaluation use. (Restricting to a specific *cohort* is a separate, opt-in
    ``node_id_filter`` on ``learn_masks`` / the evaluator, off by default.)

    Predictions come from one source (exactly one required): ``inference_dir``
    (reads ``predictions_{split}.parquet`` / ``predictions_train.parquet``) or an
    in-memory ``predictions`` dict
    ``{split: {'output', 'processed_output', 'predictions'}}`` of arrays aligned
    to ``prediction_task.get_table(split).df`` rows.

    Args:
        prediction_task:        relbench EntityTask (defines the splits).
        data:                   HeteroData graph (only for the entity-table
                                capitalisation fix).
        inference_dir:          dir with prediction parquets (None if ``predictions``).
        explanation_target_type: ``'soft'`` (``processed_output``), ``'hard'``
                                (``predictions``), or ``'ground_truth'`` (labels).
        subsample_per_split:    random subset size per split (None = keep all).
        subsample_balanced:     class-balance the subsample (default True).
        subsample_seed:         RNG seed for the subsample.
        predictions:            in-memory predictions dict (alternative to
                                ``inference_dir``).

    Returns:
        A deep copy of *prediction_task* with the three split tables, target
        column, and ``entity_table`` capitalisation set for explanation use.
    """
    if (inference_dir is None) == (predictions is None):
        raise ValueError("Provide exactly one of `inference_dir` or `predictions`.")

    explanation_task = copy.deepcopy(prediction_task)
    target_col_original = prediction_task.target_col   # used for balance

    rng = np.random.default_rng(subsample_seed) if subsample_per_split else None

    def _load_one(split):
        """Build the per-split prediction frame from in-memory `predictions` or
        from a parquet (v2 per-split layout, else v1 predictions_train.parquet)."""
        if predictions is not None:
            # In-memory: the task table rows + the prediction arrays.
            df = prediction_task.get_table(split).df.copy()
            p = predictions[split]
            df['output']           = p['output']
            df['processed_output'] = p['processed_output']
            df['predictions']      = p['predictions']
            src_label = f'{split} (in-memory)'
        else:
            path = os.path.join(inference_dir, f'predictions_{split}.parquet')
            if not os.path.exists(path):
                path = os.path.join(inference_dir, 'predictions_train.parquet')
            df = pd.read_parquet(path)
            # v2 layout writes a single 'prediction' column with the sigmoid
            # output; derive the two column names the rest of this expects.
            if 'prediction' in df.columns and 'predictions' not in df.columns:
                df['processed_output'] = df['prediction'].astype(float)
                df['predictions']      = (df['prediction'] >= 0.5).astype(int)
            src_label = os.path.basename(path)

        # pyarrow may serialize timestamp columns at [ms] resolution by default
        # when the source pd.Series was a single Timestamp broadcast across rows.
        # relbench's to_unix_time asserts [s] or [ns]; coerce any datetime column
        # to [ns] so the assertion passes.
        for col in list(df.columns):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype('datetime64[ns]')

        # Optional subsample. Balance signal preference:
        #   1. ground-truth target_col (best — what the user asked for)
        #   2. model's hard 'predictions' (used when target is hidden, e.g.
        #      relbench test splits)
        #   3. random sample (no balance possible)
        if subsample_per_split is not None:
            balance_col = None
            if subsample_balanced:
                if target_col_original in df.columns:
                    balance_col = target_col_original
                elif 'predictions' in df.columns:
                    balance_col = 'predictions'
                    print(f'    note: {src_label} has no '
                          f'{target_col_original!r} column (labels hidden); '
                          f'balancing subsample by model "predictions" instead.')
            if balance_col is not None:
                df = _subsample_balanced(df, balance_col,
                                         subsample_per_split, rng)
            else:
                n_take = min(len(df), subsample_per_split)
                df = df.sample(n=n_take, random_state=rng).reset_index(drop=True)
        return df

    for split in ['train', 'val', 'test']:
        preds_split = _load_one(split)
        # Copy the table before overwriting its frame. `get_table` returns the
        # task's own (cached) Table object, so mutating it in place would also
        # rewrite the CALLER's `prediction_task` -- leaving it holding the
        # explanation frame instead of its labels, and making a second
        # `make_explanation_task` call on the same task read already-mutated
        # input. Copy from `explanation_task` (the deepcopy) for the same reason.
        table    = copy.copy(explanation_task.get_table(split))
        table.df = preds_split
        setattr(explanation_task, f'{split}_table', table)

    # `preds` retained for the assert + downstream targets-column setup, using
    # the train-split frame (matches existing column-shape expectations).
    preds = getattr(explanation_task, 'train_table').df

    assert all(
        col in preds.columns
        for col in ['predictions', 'processed_output']
    ), f"predictions_train.parquet must contain 'predictions' and 'processed_output' columns; got {list(preds.columns)}"

    target_col = prediction_task.target_col
    for split in ['train', 'val', 'test']:
        df_split = getattr(explanation_task, f'{split}_table').df
        if target_col in df_split.columns:
            df_split['targets'] = df_split[target_col]
        else:
            # relbench hides labels on test splits — keep 'targets' as NaN so
            # downstream code that reads it sees that ground truth isn't
            # available. 'soft' / 'hard' explanation modes don't touch this.
            df_split['targets'] = float('nan')

    explanation_task.explanation_target_column      = 'predictions'
    explanation_task.explanation_soft_target_column = 'processed_output'
    explanation_task.target_col = {
        'soft':         'processed_output',  # soft model predictions
        'hard':         'predictions',        # hard/thresholded model predictions
        'ground_truth': 'targets',            # original ground-truth labels
    }[explanation_target_type]
    explanation_task.explanation_target_type = explanation_target_type

    # Fix capitalisation mismatch between relbench task and data.pt node types
    entity_lower = explanation_task.entity_table.lower()
    for key in data.node_types:
        if key.lower() == entity_lower:
            explanation_task.entity_table = key
            break

    return explanation_task