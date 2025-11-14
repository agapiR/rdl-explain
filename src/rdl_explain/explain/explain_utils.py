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
    mask_type: Union[str,List[str]],
    perturbation_type: Union[str, List[str]],
    ignore_blocked_edges: bool = True,
) -> HeteroData:
    """
    Perturb the instance based on the specified mask and perturbation type.
    Args:
        instance (HeteroData): The input instance to perturb.
        mask (dict): A dictionary containing the mask for each masked element.
        mask_type (str): The type of masking to apply ('column', 'row', 'fkpk').
        perturbation_type (str): The type of perturbation to apply. 

    Column mask:
        (node_type, col_name) -> 1-dim boolean tensor (torch.bool) with False for masked columns, True for unmasked
    Row mask:
        (node_type) -> n_rows-dim boolean tensor (torch.bool) with False for masked rows, True for unmasked)
    FKPK mask:
        (edge_type) -> 1-dim boolean tensor (torch.bool) with False for masked edges, True for unmasked edges.
    """
    # Apply multiple perturbation types in the order of 'fkpk', 'column', 'row' sequentially.
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
    
    # The correct mask order is: fkpk, column, row . Rearrange the mask_type_list to ensure this order. 
    # Then sort perturbation_type_list using the same permutation applied to mask_type_list.
    mask_type_order = ['fkpk', 'column', 'row']
    mask_type_order_dict = {mask_type: i for i, mask_type in enumerate(mask_type_order)}
    permutation_indices = sorted(range(len(mask_type_list)), key=lambda i: mask_type_order_dict.get(mask_type_list[i], float('inf')))
    mask_type_list = [mask_type_list[i] for i in permutation_indices]
    perturbation_type_list = [perturbation_type_list[i] for i in permutation_indices]

    # Create a copy of the instance to perturb
    perturbed_instance = instance           # shallow copy of the instance, to be modified in place
    # perturbed_instance = instance.clone() # this does not copy the underlying tensors
    # perturbed_instance = copy.deepcopy(instance) # deep copy of the instance, to be modified in place

    # Start time measurement for the perturbation process
    start_time = time.time()

    for mask_type, perturbation_type in zip(mask_type_list, perturbation_type_list):
        # Check if mask type is supported
        if mask_type not in ('column', 'row', 'fkpk'):
            raise ValueError(f"Invalid mask type ({mask_type}). Supported types are 'column', 'row', 'fkpk'.")

        # Retrieve the mask for the current mask type
        if mask_type not in mask_dict:
            raise ValueError(f"Mask type '{mask_type}' not found in the mask dictionary. Available mask types: {list(mask_dict.keys())}.")
        mask = mask_dict[mask_type]
        
        # Instance permutation is supported for columns of the following stypes: 
        supported_stypes = [torch_frame.numerical, torch_frame.categorical, torch_frame.embedding, torch_frame.timestamp]
        node_type_to_col_names_by_stype_dict = node_type_to_col_names_by_stype(instance)
        for node_type, col_names_by_stype in node_type_to_col_names_by_stype_dict.items():
            for stype, col_names in col_names_by_stype.items():
                if stype not in supported_stypes:
                    raise ValueError(f"Unsupported stype {stype} for perturbation. Supported stypes are: {supported_stypes}")

        # Check if perturbation type is supported
        if mask_type == 'fkpk' and perturbation_type not in ('foreign_key_permutation', 'foreign_key_exchange', 'foreign_key_uniform_random', 'foreign_key_hist_random'):
            raise ValueError(f"Invalid perturbation type ({perturbation_type}) for mask type 'fkpk'. Supported types are 'foreign_key_permutation', 'foreign_key_exchange', 'foreign_key_uniform_random', 'foreign_key_hist_random'.")
        if (mask_type == 'row' or mask_type == 'column') and perturbation_type not in ('permutation_joint', 'permutation_independent'):
            raise ValueError(f"Invalid perturbation type ({perturbation_type}) for mask type '{mask_type}'. Supported types are 'permutation_joint', 'permutation_independent'.")

        if mask_type == 'column':
            if perturbation_type == 'permutation_joint': # permute each masked columns jointly for each table
                for tab in instance.node_types:
                    global_permutated_indices = torch.randperm(instance[tab].tf.num_rows)
                    for stype, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if stype==torch_frame.numerical or stype==torch_frame.categorical: # instance[tab].tf.feat_dict[stype] shape is (n_rows, n_cols)
                            t1 = time.time()
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][global_permutated_indices]
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
                        elif stype==torch_frame.timestamp: # instance[tab].tf.feat_dict[stype] shape is (n_rows, n_cols, n_dim_per_col)
                            t1 = time.time()
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][global_permutated_indices]
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
                        elif stype==torch_frame.embedding: # instance[tab].tf.feat_dict[stype][:, col_idx].values shape is (n_rows, n_dim)
                            t1 = time.time()
                            emb_tensor_list = []
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    emb_tensor_list.append(instance[tab].tf.feat_dict[stype][:, col_idx].values[global_permutated_indices])
                                else:
                                    emb_tensor_list.append(perturbed_instance[tab].tf.feat_dict[stype][:, col_idx].values)
                            perturbed_instance[tab].tf.feat_dict[stype] = MultiEmbeddingTensor.from_tensor_list(emb_tensor_list)
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
            elif perturbation_type == 'permutation_independent': # permute each masked column independently for each table
                for tab in instance.node_types:
                    n_rows = instance[tab].tf.num_rows
                    for stype, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if stype==torch_frame.numerical or stype==torch_frame.categorical:
                            t1 = time.time()
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    local_permutated_indices = torch.randperm(n_rows)
                                    perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][local_permutated_indices]
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
                        elif stype==torch_frame.timestamp:
                            t1 = time.time()
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    local_permutated_indices = torch.randperm(n_rows)
                                    perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][local_permutated_indices]
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
                        elif stype==torch_frame.embedding:
                            t1 = time.time()
                            emb_tensor_list = []
                            for col_idx, col_name in enumerate(col_names):
                                if ~mask[(tab, col_name)]:
                                    local_permutated_indices = torch.randperm(n_rows)
                                    emb_tensor_list.append(instance[tab].tf.feat_dict[stype][:, col_idx].values[local_permutated_indices])
                                else:   
                                    emb_tensor_list.append(perturbed_instance[tab].tf.feat_dict[stype][:, col_idx].values)
                            perturbed_instance[tab].tf.feat_dict[stype] = MultiEmbeddingTensor.from_tensor_list(emb_tensor_list)
                            t2 = time.time()
                            # print(f"Permuted {len(col_names)} columns of type {stype} for table {tab} in {t2 - t1:.2f} seconds.")
            else:
                raise ValueError(f"Invalid perturbation type ({perturbation_type}) for mask type 'column'.")
        elif mask_type == 'row':

            def get_row_permutation_for_masked_rows(n_rows, mask):
                """
                Generate a random permutation of row indices, keeping unmasked rows (mask = False) in their original positions.
                """
                rows_to_permute = torch.where(~mask)[0]
                permutation = rows_to_permute[torch.randperm(len(rows_to_permute))]
                global_indices = torch.arange(n_rows)
                global_permutated_indices = global_indices.clone()
                global_permutated_indices[rows_to_permute] = global_indices[permutation]
                return global_permutated_indices

            if perturbation_type == 'permutation_joint':
                for tab in instance.node_types:
                    # make a global permutation of indices for each table, only permuting the masked rows among themselves
                    global_permutated_indices = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                    for stype, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if stype==torch_frame.numerical or stype==torch_frame.categorical:
                            perturbed_instance[tab].tf.feat_dict[stype] = instance[tab].tf.feat_dict[stype][global_permutated_indices]
                        elif stype==torch_frame.timestamp:
                            perturbed_instance[tab].tf.feat_dict[stype] = instance[tab].tf.feat_dict[stype][global_permutated_indices]
                        elif stype==torch_frame.embedding:
                            emb_tensor_list = []
                            for col_idx, col_name in enumerate(col_names):
                                emb_tensor_list.append(instance[tab].tf.feat_dict[stype][:, col_idx].values[global_permutated_indices])
                            perturbed_instance[tab].tf.feat_dict[stype] = MultiEmbeddingTensor.from_tensor_list(emb_tensor_list)
            elif perturbation_type == 'permutation_independent':
                for tab in instance.node_types:
                    for stype, col_names in node_type_to_col_names_by_stype_dict[tab].items():
                        if stype==torch_frame.numerical or stype==torch_frame.categorical:
                            for col_idx, col_name in enumerate(col_names):
                                local_permutated_indices = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                                perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][local_permutated_indices]
                        elif stype==torch_frame.timestamp:
                            for col_idx, col_name in enumerate(col_names):
                                local_permutated_indices = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                                perturbed_instance[tab].tf.feat_dict[stype][:, col_idx] = instance[tab].tf.feat_dict[stype][:, col_idx][local_permutated_indices]
                        elif stype==torch_frame.embedding:
                            emb_tensor_list = []
                            for col_idx, col_name in enumerate(col_names):
                                local_permutated_indices = get_row_permutation_for_masked_rows(instance[tab].tf.num_rows, mask[tab])
                                emb_tensor_list.append(instance[tab].tf.feat_dict[stype][:, col_idx].values[local_permutated_indices])
                            perturbed_instance[tab].tf.feat_dict[stype] = MultiEmbeddingTensor.from_tensor_list(emb_tensor_list)
            else: 
                raise ValueError(f"Invalid perturbation type ({perturbation_type}) for mask type 'row'.") 
        elif mask_type == 'fkpk':
            # collect node types that appear in the induced subgraph of unmasked edges 
            # when no end of an edge is unmasked the edge is blocked and we can skip perturbation of this edge type
            unmasked_nodes = set()
            for edge_type in instance.edge_types:
                if mask[edge_type]:
                    src, edge_name, dst = edge_type
                    unmasked_nodes.add(src)
                    unmasked_nodes.add(dst)
            if not unmasked_nodes:
                # if no unmasked nodes are found (in the case of complete masking), disable the ignore_blocked_edges option
                ignore_blocked_edges = False
            else:
                print(f"Unmasked nodes (i.e., nodes adjacent to unmasked edges): {unmasked_nodes}.")
            # always symmetrically perturb the edge type and its reverse. NOTE: Convention: (src, 'name', dst) -> (dst, 'rev_name', src)
            perturbed_fk_pairs = set() # to keep track of perturbed foreign key pairs
            for edge_type in instance.edge_types:
                src, edge_name, dst = edge_type
                # if both src and dst are not in the unmasked nodes, skip this edge type
                # no perturbation is needed since signal passed through this edge type, i.e., the edge is blocked
                if ignore_blocked_edges and (src not in unmasked_nodes) and (dst not in unmasked_nodes):
                    print(f"Skipping edge type {edge_type} since both source ({src}) and destination ({dst}) nodes are not in the unmasked nodes.")
                    continue
                if 'rev_' in edge_name:
                    rev_edge_type = (dst, edge_name.replace('rev_', ''), src)
                else:
                    rev_edge_type = (dst, 'rev_' + edge_name, src)
                assert rev_edge_type in instance.edge_types, f"Reverse edge type {rev_edge_type} not found in the instance edge types. Available edge types: {instance.edge_types}."
                # if edge type already perturbed, skip it
                if edge_type in perturbed_fk_pairs or rev_edge_type in perturbed_fk_pairs:
                    continue
                # for edges that are masked on both directions, we need to perturb the corresponding foreign keys
                if ~mask[edge_type] and ~mask[rev_edge_type]:
                    # print(f"Perturbing foreign keys for edge type {edge_type} and its reverse {rev_edge_type}.")
                    num_edges = instance.edge_index_dict[edge_type].shape[1]
                    num_unique_src = instance.edge_index_dict[edge_type][0].unique().shape[0]
                    num_unique_dst = instance.edge_index_dict[edge_type][1].unique().shape[0]
                    fk = 0 if num_unique_dst < num_edges else 1
                    rev_fk = 1 - fk # reverse foreign key index
                    perturbed_instance[edge_type].edge_index = instance.edge_index_dict[edge_type]
                    perturbed_instance[rev_edge_type].edge_index = instance.edge_index_dict[edge_type][[1, 0], :] # reverse edge index
                    assert torch.all(perturbed_instance.edge_index_dict[edge_type][fk]==perturbed_instance.edge_index_dict[rev_edge_type][rev_fk]), \
                        f"Edge type {edge_type} and its reverse {rev_edge_type} are not symmetrically stored in the edge_index_dict."
                    if perturbation_type == 'foreign_key_permutation':
                        fk_permutation = torch.randperm(num_edges)
                        perturbed_instance[edge_type].edge_index[fk] = perturbed_instance.edge_index_dict[edge_type][fk][fk_permutation]
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = perturbed_instance.edge_index_dict[rev_edge_type][rev_fk][fk_permutation]
                    elif perturbation_type == 'foreign_key_exchange':
                        unique_fks = perturbed_instance.edge_index_dict[edge_type][fk].unique().numpy()
                        unique_fks_indices = np.arange(len(unique_fks))
                        unique_fks_permuted = unique_fks[np.random.permutation(len(unique_fks))]
                        for i, unique_fk in enumerate(unique_fks):
                            # Find the indices of the unique foreign key in the edge index
                            fk_indices = (perturbed_instance.edge_index_dict[edge_type][fk] == unique_fk).nonzero(as_tuple=True)[0]
                            if len(fk_indices) > 0:
                                # Replace the unique foreign key with the permuted one
                                perturbed_instance[edge_type].edge_index[fk][fk_indices] = unique_fks_permuted[unique_fks_indices[i]]
                                perturbed_instance[rev_edge_type].edge_index[rev_fk][fk_indices] = unique_fks_permuted[unique_fks_indices[i]]
                    elif perturbation_type == 'foreign_key_uniform_random': 
                        # For each edge and its reverse, randomly select a foreign key from the unique foreign keys
                        unique_fks = perturbed_instance.edge_index_dict[edge_type][fk].unique().numpy()
                        # Sample random foreign keys uniformly from the unique foreign keys
                        random_fks = np.random.choice(unique_fks, size=num_edges, replace=True)
                        # Perturb the edge index with the sampled foreign keys
                        perturbed_instance[edge_type].edge_index[fk] = torch.tensor(random_fks, dtype=torch.long)
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = torch.tensor(random_fks, dtype=torch.long)
                    elif perturbation_type == 'foreign_key_hist_random':
                        # For each edge and its reverse, randomly select a foreign key from the unique foreign keys
                        # Preserve the frequency distribution of the foreign keys, randomly shuffle the foreign key frequencies
                        unique_fks, counts = perturbed_instance.edge_index_dict[edge_type][fk].unique(return_counts=True)
                        # Shuffle the counts to create a new foreign key distribution
                        shuffled_counts = np.random.permutation(counts.numpy())
                        # Sample the foreign keys according to the shuffled counts
                        random_fks = np.random.choice(unique_fks, size=num_edges, replace=True, p=shuffled_counts / shuffled_counts.sum())
                        # Perturb the edge index with the sampled foreign keys
                        perturbed_instance[edge_type].edge_index[fk] = torch.tensor(random_fks, dtype=torch.long)
                        perturbed_instance[rev_edge_type].edge_index[rev_fk] = torch.tensor(random_fks, dtype=torch.long)
                    else:
                        raise ValueError(f"Invalid perturbation type ({perturbation_type}) for mask type 'fkpk'.")
                    # Add the edge type and its reverse to the set of perturbed foreign key pairs to avoid double perturbation
                    perturbed_fk_pairs.add(edge_type)
                    perturbed_fk_pairs.add(rev_edge_type)
        else:
            raise ValueError(f"Invalid mask type ({mask_type}). Supported types are 'column', 'row', 'fkpk'.")

    end_time = time.time()
    print(f"Successfully perturbed instance with mask type '{mask_type}' and perturbation strategy '{perturbation_type}'. Elapsed time: {end_time - start_time:.2f} seconds.")

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