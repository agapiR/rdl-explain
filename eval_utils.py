import math
import os
import sys
import time
import copy
import json
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_frame import stype
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Explain module imports
from src.explain.explain_utils import make_schema_graph, make_schema_dag, draw_schema_dag, explanation_element_wording, prepare_node_explanation_task
from src.explain.explainer import RDLExplainer

def calculate_explanation_size(
    boolean_mask: Dict[str, torch.Tensor],
    n_selection_predicates: int,
    n_tuples_per_relation: Dict[str, int],
    n_data_columns_per_relation: Dict[str, int],
    n_FK_all: int,
) -> Tuple[int, int]:
    """
    Calculate the size of the explanation based on the boolean mask.
    Returns the number of True values in the mask and the total number of elements in the mask.
    """
    # detect relations with projections
    relations_with_projections = set()
    if 'column' in boolean_mask:
        relations_with_projections = set([relation for (relation, col), mask in boolean_mask['column'].items() if torch.sum(mask).item() > 0])        
    # detect relations with joins
    relations_with_joins = set()
    if 'fkpk' in boolean_mask:
        for (rel1, _, rel2), mask in boolean_mask['fkpk'].items():
            if torch.sum(mask).item() > 0:
                relations_with_joins.add(rel1)
                relations_with_joins.add(rel2)

    ## Data size
    # if a relation has no projections or joins, it is dropped
    all_relations = set(n_tuples_per_relation.keys())
    relations_to_drop = all_relations - relations_with_projections - relations_with_joins
    # count the tuples per relation that can be removed
    # 1. remove all tuples of removed relations
    n_tuples_per_relation_removed = {relation: n_tuples_per_relation[relation] for relation in relations_to_drop} 
    # 2. if a relation has selections, remove all tuples that are not selected, i.e. where the boolean mask is False
    if 'row' in boolean_mask:
        for relation, mask in boolean_mask['row'].items():
            if relation in all_relations:
                n_tuples_per_relation_removed[relation] = torch.sum(~mask).item()
    # count the columns per relation that can be removed
    if 'column' in boolean_mask:
        n_data_columns_per_relation_removed = {relation: len([col for col in boolean_mask['column'] if col[0] == relation and not boolean_mask['column'][col].any()]) for relation in n_data_columns_per_relation.keys()}
    else:
        n_data_columns_per_relation_removed = {relation: 0 for relation in n_data_columns_per_relation.keys()}
    ## SQL size
    # count the number of projections and joins
    if 'column' in boolean_mask:
        n_projections = sum([torch.sum(mask).item() for mask in boolean_mask['column'].values()])
    else:
        n_projections = 0
    if 'fkpk' in boolean_mask:
        n_joins = sum([torch.sum(mask).item() for mask in boolean_mask['fkpk'].values()]) / 2
    else:
        n_joins = 0
        # if n_projections > 0: # add all the FKs in the projections, TODO: we can remove the ones that are in dropped relations
        #     n_projections += n_FK_all if n_FK_all is not None else 0

    size_stats = {
        'n_projections': n_projections,
        'n_joins': n_joins,
        'n_selection_predicates': n_selection_predicates,
        'n_fk_total': n_FK_all,
        'n_tuples_per_relation': n_tuples_per_relation,
        'n_tuples_per_relation_removed': n_tuples_per_relation_removed,
        'n_data_columns_per_relation': n_data_columns_per_relation,
        'n_data_columns_per_relation_removed': n_data_columns_per_relation_removed,
        'relations_removed': list(relations_to_drop),
    }
    return size_stats

def estimate_fidelity_given_boolean_mask(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: Union[str, List[str]],
    boolean_mask: Union[Dict[Any, torch.Tensor], Dict[Any, Dict[Any, torch.Tensor]]],
    perturbation_strategy: Union[str, List[str]] = 'permutation_independent',
    n_samples: int = 10,
    splits: List[str] = ['test'],
) -> Dict[str, float]:
    """Estimate fidelity of the explanation."""

    # initialize the fidelity results
    fidelity = {split: 0.0 for split in splits}
    fidelity_var = {split: 0.0 for split in splits}

    # initialize perturbed prediction results
    predictions = {split: {} for split in splits}
            
    # estimate the fidelity for the mask
    for split in splits:
        fid, fid_std, pred, target = explainer.estimate_fidelity(split, boolean_mask, explanation_type=explanation_type, perturbation_strategy=perturbation_strategy, num_samples=n_samples)
        fidelity[split] = fid
        fidelity_var[split] = fid_std
        predictions[split]['predictions_per_sample'] = pred
        predictions[split]['targets'] = target  
        print(f"Estimated fidelity: {fid:.4f} +/- {fid_std:.4f} {split}")         

    # Fidelity results
    fidelity_results = {
        'fidelity': fidelity,
        'fidelity_var': fidelity_var,
    }
    
    return fidelity_results, predictions


def estimate_fidelity_given_ranking(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements_ranking: List[Any],
    klist: list = None,
    kmax: int = 10,
    kstep: int = 2,
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    splits: List[str] = ['test'],
) -> Dict[str, float]:
    """Estimate fidelity of the explanation."""

    # check if klist is provided, otherwise use kmax and kstep
    if klist is None:
        klist = [kmax] # ensure kmax is included in the klist
        if kmax <= 0:
            raise ValueError("kmax must be greater than 0.")
        if kstep <= 0:
            raise ValueError("kstep must be greater than 0.")
        klist.extend(list(range(0, kmax + 1, kstep)))  # add kstep increments up to kmax
        klist = sorted(list(set(klist)))  # remove duplicates

    # initialize the fidelity results
    fidelity_top_k = {split: [] for split in splits}
    fidelity_var_top_k = {split: [] for split in splits}
    # initialize perturbed prediction results for kmax elements
    predictions = {split: {} for split in splits}
    
    for k in klist:
        # get the top-k elements
        top_k_elements = explanation_elements_ranking[:k] if k > 0 else []
        print(f"Estimating fidelity for top-{k} elements : {top_k_elements}")

        # define the hard mask
        hard_mask_top_k = {}
        for e in explanation_elements_ranking:
            if e in top_k_elements:
                hard_mask_top_k[e] = torch.tensor([True])
            else:
                hard_mask_top_k[e] = torch.tensor([False])
        
        # estimate the fidelity for the top-k elements
        for split in splits:
            fid, fid_std, pred, target = explainer.estimate_fidelity(split, hard_mask_top_k, explanation_type=explanation_type, perturbation_strategy=perturbation_strategy, num_samples=n_samples)
            fidelity_top_k[split].append(fid)
            fidelity_var_top_k[split].append(fid_std)
            print(f"Top-{k} elements: {fid:.4f} +/- {fid_std:.4f} {split} fidelity")     
            # save the predictions for the last top-k 
            predictions[split]['predictions_per_sample'] = pred
            predictions[split]['targets'] = target           

    # Fidelity results
    fidelity_results = {
        'fidelity_top_k': fidelity_top_k,
        'fidelity_var_top_k': fidelity_var_top_k,
        'k_list': klist,
        'explanation_elements_ranking': explanation_elements_ranking,
    }
    
    return fidelity_results, predictions

def initialize_explainer(
    data_config_path: str,
    model_config_path: str,
    model_params_path: str,
    task_dir: str,
    explanation_target_type: str = 'soft',
) -> Tuple[RDLExplainer, str, str, Dict[str, pd.DataFrame]]:
    # Load configuration
    config = load_config(model_config_path)

    # Load dataset and task
    dataset, task, task_parser = load_dataset_and_task(data_config_path) 

    # Construct graph data
    data, col_stats_dict = construct_graph(config, dataset)
    del dataset # Delete dataset to free memory
    
    # Load model
    model_to_explain = load_model(config, model_params_path, construct=True, data=data, col_stats_dict=col_stats_dict, task=task)

    # Make explanation task
    predictions= {split: pd.read_parquet(os.path.join(task_dir, f'predictions_{split}.parquet')) for split in ['train', 'val', 'test']}
    explanation_task = prepare_node_explanation_task(task, predictions, explanation_target_type=explanation_target_type)
    del task # Delete original task to free memory

    # Initialize the explainer
    explainer = RDLExplainer(config, model_to_explain, data, explanation_task)

    return explainer, explanation_task

def combine_stats(
    means: np.ndarray, 
    stds: np.ndarray, 
    counts: Optional[np.ndarray] = None
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Combine means and stds across the first dimension using weighted average and total variance.

    Parameters:
        means (np.ndarray): Shape (n, ...) – sample means.
        stds (np.ndarray): Shape (n, ...) – sample standard deviations.
        counts (Optional[np.ndarray]): Shape (n,) – sample sizes per run. 
                                       If None, assumes equal weights.

    Returns:
        Tuple[mean, std]: Either scalars or arrays depending on input dimensionality.
    """
    if counts is None:
        counts = np.ones(means.shape[0], dtype=means.dtype)

    # Reshape counts for broadcasting
    counts = counts.reshape((-1,) + (1,) * (means.ndim - 1))  # shape (n, 1, ..., 1)
    total_count = np.sum(counts, axis=0)

    weighted_mean = np.sum(counts * means, axis=0) / total_count
    variance = np.sum(counts * (stds**2 + (means - weighted_mean)**2), axis=0) / total_count
    combined_std = np.sqrt(variance)

    # Return scalars if 0-dimensional
    if weighted_mean.ndim == 0:
        return weighted_mean.item(), combined_std.item()
    else:
        return weighted_mean, combined_std

def visualize_predictions(
    predictions: pd.DataFrame,
    predictions_col: str,
    dataset_name: str,
    task_name: str,
    file_path: str,
    group_col: str = None,
) -> None:
    """Visualize prediction histogram."""
    plt.figure(figsize=(10, 6))
    if group_col and group_col in predictions.columns:
        predictions.groupby(group_col)[predictions_col].hist(bins=50, alpha=0.5, legend=True)
        plt.title(f'Predictions Distribution by {group_col} - {dataset_name} - {task_name}')
    else:
        predictions[predictions_col].hist(bins=50)
        plt.title(f'Predictions Distribution - {dataset_name} - {task_name}')
    plt.xlabel('Prediction Value')
    plt.ylabel(f'Frequency (n_samples = {len(predictions)})')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def visualize_subgraph(
    database: Any,
    entity_table: str,
    num_layers: int,
    dataset_name: str,
    task_name: str,
    file_path: str,
) -> None:
    """Visualize the computational subgraph."""
    schema_graph = make_schema_graph(database, directed=False, self_loop=True)
    schema_DAG = make_schema_dag(schema_graph, depth=num_layers, source_entity=entity_table, avoid_backtracking=False)
    draw_schema_dag(schema_DAG, save_path=file_path)
    
def visualize_schema_graph_with_important_joins(
    schema_graph: Dict[str, List],
    important_joins: List[Tuple[str, str, str]],
    dataset_name: str,
    task_name: str,
    file_path: str,
    important_tables: Optional[List[str]] = None,
) -> None:
    """Visualize the schema graph with important joins highlighted."""
    # Create graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for table_name in schema_graph.keys():
        G.add_node(table_name)
    
    # Add edges to the graph
    for table_name, edges in schema_graph.items():
        for edge in edges:
            G.add_edge(table_name, edge['dst'], label=edge['edge_name'], edge_type=edge['edge_type'])    

    # Draw the graph (requires matplotlib)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    # If important tables are provided, highlight the corresponding nodes with red boundaries
    highlight_nodes = []
    if important_tables is not None:
        for table in important_tables:
            if table in G.nodes:
                highlight_nodes.append(table)
            else:
                print(f"Important table {table} not found in the schema graph.")
    # 1. Draw nodes
    # first draw only the selected nodes with custom edge color
    if highlight_nodes:
        nx.draw(G, pos, nodelist=highlight_nodes, node_color='lightblue', edgecolors='red', linewidths=5, node_size=2000)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, linewidths=1, node_size=1800)
    # 2. Draw edges
    edge_labels = nx.get_edge_attributes(G, 'label')
    for (u, v), label in edge_labels.items():
        if 'rev_' in label:  # do not add a label for reverse edges
            continue
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        plt.text(x, y + 0.05, label, fontsize=8, ha='center', va='bottom', color='gray')
    # Highlight important joins
    for join in important_joins:
        src_table, edge_name, dst_table = join
        if G.has_edge(src_table, dst_table) and G[src_table][dst_table]['label'].split('_')[-1] == edge_name.split('_')[-1]:
            print(f"Highlighting edge type {join} in the schema graph with color red.")
            G[src_table][dst_table]['color'] = 'red'
        else:
            print(f"Join {join} not found in the schema graph.")
    edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges()]
    print(f"Drawing schema graph with {len(G.edges())} edges with {edge_colors.count('red')} important edge types highlighted in red., {edge_colors.count('black')} normal edges.")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    plt.title(f'Schema Graph with Important Joins - {dataset_name} - {task_name}')
    plt.savefig(file_path, bbox_inches='tight')

def visualize_masks(
    mask: Dict[Any, Union[float, list]],
    dataset_name: str,
    task_name: str,
    file_path: str,
) -> None:
    """Visualize the mask values."""
    plt.figure(figsize=(10, 1 + int(0.3 * len(mask))))
    if all(isinstance(v, list) for v in mask.values()):
        mask_vals_avg = {explanation_element_wording(k): np.mean(v) for k, v in mask.items()}
        mask_vals_std = {explanation_element_wording(k): np.std(v) for k, v in mask.items()}
        plt.barh(list(mask_vals_avg.keys()), list(mask_vals_avg.values()), xerr=list(mask_vals_std.values()))
    else:
        mask_vals_avg = {explanation_element_wording(k): v for k, v in mask.items()}
        mask_vals_std = {explanation_element_wording(k): 0.0 for k in mask.keys()} # No standard deviation for single values
        plt.barh(list(mask_vals_avg.keys()), list(mask_vals_avg.values()))
    plt.xlim(0, 1.0)
    plt.xlabel('Mask value')
    plt.ylabel('Explanation element')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return mask_vals_avg, mask_vals_std

def visualize_fidelity_curves(
    fidelity_results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    task_name: str,
    file_path: str,
    split: str = 'test',
    kstar: Optional[int] = None,
) -> None:
    plt.figure(figsize=(5, 3))
    k_tot = []
    for method, fidelity_results_for_method in fidelity_results.items():
        k_list = fidelity_results_for_method['k_list']
        fidelity_top_k = fidelity_results_for_method['fidelity_top_k']
        fidelity_var_top_k = fidelity_results_for_method['fidelity_var_top_k']
        plt.plot(k_list, fidelity_top_k[split], label=method, marker='o')
        plt.fill_between(k_list,
                        np.array(fidelity_top_k[split]) - np.array(fidelity_var_top_k[split]),
                        np.array(fidelity_top_k[split]) + np.array(fidelity_var_top_k[split]),
                        alpha=0.2)
        if 'explanation_elements_ranking' in fidelity_results_for_method:
            ranking = fidelity_results_for_method['explanation_elements_ranking']
            if ranking:
                k_tot.append(len(ranking))
        else:
            print(f"No ranking found for method {method}. Using kmax as total k.")
    if kstar is not None: # add a vertical dashed line for kstar if provided
        plt.axvline(x=kstar, color='black', linestyle='--', label=f'k* = {kstar}')
    plt.title(f'Fidelity Curves - {dataset_name} - {task_name} - {split} split')
    plt.xlabel(f'Top-k Explanation Elements (Total: {max(k_tot)})')
    plt.ylabel('Fidelity')
    plt.legend()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def visualize_boolean_mask(
    index_mask: Dict[str, torch.Tensor], 
    dataset_name: str, 
    task_name: str, 
    file_path: str,
) -> None:
    """
    Visualize the boolean mask as a bar chart.
    X-axis: Node types
    Y-axis: Count of True and False values (side by side for each node type)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    node_types = list(index_mask.keys())
    n_node_types = len(node_types)

    # rank node types by the total size of the mask
    node_types.sort(key=lambda nt: index_mask[nt].numel(), reverse=True)
    
    # collect mask values counts
    mask_values_counts = {'True': [], 'False': []}
    for i, node_type in enumerate(node_types):
        mask = index_mask[node_type]
        n_false = torch.sum(~mask).item()
        n_true = torch.sum(mask).item()
        mask_values_counts['True'].append(n_true)
        mask_values_counts['False'].append(n_false)

    x = np.arange(n_node_types)  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width/2, mask_values_counts['True'], width, label='True')
    rects2 = ax.bar(x + width/2, mask_values_counts['False'], width, label='False')
    # also show the exact count values above each bar
    ax.bar_label(rects1, mask_values_counts['True'], padding=3)
    ax.bar_label(rects2, mask_values_counts['False'], padding=3)
    ax.set_ylabel('Count')
    ax.set_title(f'Boolean Mask Counts for {dataset_name} - {task_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(node_types, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(file_path)

def get_important_projections(
    dataset_name: str,
    task_name: str,
    mask_dir: str,
    reduce: float = 1.0,  # Reduce the number of important columns by this factor (1.0 means no reduction, 0.5 means half)
    suffix: str = 'best',
):
    # Set data and mask directories
    column_mask_file = os.path.join(mask_dir, 'column', f'{dataset_name}/{task_name}/{dataset_name}-{task_name}-{suffix}-mask_vals.json')

    # Load column masks
    if not os.path.exists(column_mask_file):
        print(f"Mask file not found: {column_mask_file}.")
        important_columns = []
        column_ranking = []
        column_kstar = 0
    else:
        with open(column_mask_file, 'r') as f:
            mask_vals = json.load(f)
            mask_vals = {tuple(k.split('-')): v[-1] for k, v in mask_vals.items()}
            n_mask_vals_above_threshold = sum(1 for v in mask_vals.values() if v > 0.1)
            column_ranking = [k for k, v in sorted(mask_vals.items(), key=lambda item: item[1], reverse=True)]
            column_kstar = n_mask_vals_above_threshold
            column_kstar = math.ceil(column_kstar * reduce) # Reduce the number of important columns
            important_columns = column_ranking[:column_kstar]
            print(f"Number of important projections: {len(column_ranking[:column_kstar])}. Important columns: {important_columns}")

    return important_columns, column_ranking, column_kstar

def get_important_joins(
    dataset_name: str,
    task_name: str,
    mask_dir: str,
    reduce: float = 1.0, # Reduce the number of important joins by this factor (1.0 means no reduction, 0.5 means half)
    suffix: str = 'best',
):
    # Set data and mask directories
    fkpk_mask_file = os.path.join(mask_dir, 'fkpk', f'{dataset_name}/{task_name}/{dataset_name}-{task_name}-{suffix}-mask_vals.json')

    # Load join masks
    if not os.path.exists(fkpk_mask_file):
        print(f"Mask file not found: {fkpk_mask_file}.")
        important_joins = []
        join_ranking = []
        join_kstar = 0
    else:
        with open(fkpk_mask_file, 'r') as f:
            mask_vals = json.load(f)
            mask_vals = {tuple(k.split('-')): v[-1] for k, v in mask_vals.items()}
            n_mask_vals_above_threshold = sum(1 for v in mask_vals.values() if v > 0.1)
            join_ranking = [k for k, v in sorted(mask_vals.items(), key=lambda item: item[1], reverse=True)]
            join_kstar = n_mask_vals_above_threshold
            join_kstar = math.ceil(join_kstar * reduce) # Reduce the number of important joins
            important_joins = join_ranking[:join_kstar]
            print(f"Number of important joins: {len(join_ranking[:join_kstar])/2}. Important joins: {[j for j in important_joins if 'rev_' not in j[1]]}")

    return important_joins, join_ranking, join_kstar

def get_time_to_learn_masks(
    dataset_name: str,
    task_name: str,
    mask_dir: str,
    suffix: str = 'best',
):
    # Set mask directory
    mask_learning_metrics_file = os.path.join(mask_dir, f'{dataset_name}/{task_name}/{dataset_name}-{task_name}-{suffix}-metrics.json')
    # Load metrics
    if not os.path.exists(mask_learning_metrics_file):
        raise FileNotFoundError(f"Mask learning metrics file not found: {mask_learning_metrics_file}")
    with open(mask_learning_metrics_file, 'r') as f:
        metrics = json.load(f)
    time_to_learn_masks = metrics['time']
    return time_to_learn_masks

def construct_boolean_mask_from_explanation_elements_ranking(
    explanation_elements_ranking: List[Any],
    k: int = 10,
):
    """Construct a boolean mask from the explanation elements ranking."""
    print(f"Constructing boolean mask from explanation elements ranking with kmax={k}...")
    top_k_elements = explanation_elements_ranking[:k] if k > 0 else []
    boolean_mask_top_k = {}
    for e in explanation_elements_ranking:
        if e in top_k_elements:
            boolean_mask_top_k[e] = torch.tensor([True])
        else:
            boolean_mask_top_k[e] = torch.tensor([False])
    return boolean_mask_top_k

def construct_row_boolean_mask(
    explainer: RDLExplainer,
    explanation_task: Any,
    hard_mask_vals: Dict[Tuple[str, str, stype, str, List], torch.Tensor],
    result_dir: str = './results',
    suffix: str = '',
    visualize: bool = True,
):
    """Evaluate explanation fidelity for provided masks."""
    print("Constructing boolean mask for rows from hard mask values...")
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Construct the boolean mask 
    instance = explainer.data
    filter_node_types = [filter_pred[0] for filter_pred in hard_mask_vals['filter_predicate']]
    print(f"Filter node types: {filter_node_types}")
    boolean_mask = {node_type: torch.zeros(instance[node_type].tf.num_rows, dtype=torch.bool) for node_type in instance.node_types if node_type in filter_node_types}
    # for all unfiltered node types, set the boolean mask to True
    for node_type in instance.node_types:
        if node_type not in boolean_mask:
            boolean_mask[node_type] = torch.ones(instance[node_type].tf.num_rows, dtype=torch.bool)
    filter_predicates = hard_mask_vals['filter_predicate']
    for fi, filter_predicate in enumerate(filter_predicates):
        # Get index-level masks from explainer
        index_level_mask_fi = explainer.initialize_masks('filter', filter_predicate=filter_predicate)
        # Get hard mask values 
        hard_mask_vals_fi = hard_mask_vals['hard_mask'][fi]
        # Store the boolean mask
        mask_indices = index_level_mask_fi['indices'].detach().cpu()
        filter_node_type = filter_predicate[0]
        boolean_mask[filter_node_type] = torch.logical_or(boolean_mask[filter_node_type], hard_mask_vals_fi[mask_indices].clone().detach())

    # Count indices that are False in the boolean mask
    n_false_indices = sum([torch.sum(~mask).item() for mask in boolean_mask.values()])
    print(f"Number of indices that are False in the boolean mask: {n_false_indices} / {sum([mask.numel() for mask in boolean_mask.values()])}")

    # Construct all-True and all-False masks
    all_true_mask = {node_type: torch.ones(mask.shape, dtype=torch.bool) for node_type, mask in boolean_mask.items()}
    all_false_mask = {node_type: torch.zeros(mask.shape, dtype=torch.bool) for node_type, mask in boolean_mask.items()}

    if visualize:
        # Visualize the boolean mask
        boolean_mask_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-boolean_mask.png')
        visualize_boolean_mask(boolean_mask, dataset_name, task_name, boolean_mask_file)

        # Visualize all True mask (for reference)
        all_true_mask_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-all_true_mask.png')
        visualize_boolean_mask(all_true_mask, dataset_name, task_name, all_true_mask_file)

    return boolean_mask, all_true_mask, all_false_mask
