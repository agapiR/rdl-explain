import math
import os
import sys
import time
import copy
import json
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_frame import stype
from torch_frame.data.stats import StatType
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Explain module imports
from src.explain.explainer import RDLExplainer
from src.explain.explain_utils import explanation_element_wording, prepare_node_explanation_task, node_type_to_col_names_by_stype

# Eval imports
from eval_utils import visualize_masks

SUPPORTED_STYPES_FOR_FILTERS = [stype.numerical, stype.categorical, stype.timestamp, stype.embedding]

def collect_candidate_filters_given_important_columns(
    data: HeteroData,
    col_stats_dict: Dict[str, Any],
    important_col_ranking_top_k: list,
) -> list:
    """
    Collect candidate explanation filters based on important columns.
    
    Args:
        data (HeteroData): The graph data.
        col_stats_dict (Dict[str, Any]): Column statistics dictionary.
        important_col_ranking_top_k (list): List of important columns ranked by importance.
    
    Returns:
        list: A list of candidate filters.
        filter value mapping: A mapping of processed filter values to their original values.
    """
    node_type_to_col_names_by_stype_dict = node_type_to_col_names_by_stype(data)

    # Collect candidate filters based on important columns
    candidate_filters = []
    value_mapping = []
    for col in important_col_ranking_top_k:
        col_node_type, col_name = col

        # Find the stype for the column from node_type_to_col_names_by_stype_dict
        col_stype = None
        for stype, col_names in node_type_to_col_names_by_stype_dict[col_node_type].items():
            if col_name in col_names:
                col_stype = stype
                break
        
        # If the column's stype is not supported, skip it
        if col_stype not in SUPPORTED_STYPES_FOR_FILTERS:
            print(f"Column {col_name} with stype {col_stype} is not supported for filters. Skipping.")
            continue

        # Get the column statistics
        if col_stype == stype.categorical:
            categories, counts = col_stats_dict[col_node_type][col_name][StatType.COUNT]
        elif col_stype == stype.numerical:
            mean = col_stats_dict[col_node_type][col_name][StatType.MEAN]
            std = col_stats_dict[col_node_type][col_name][StatType.STD]
            quantiles = col_stats_dict[col_node_type][col_name][StatType.QUANTILES]

        # Collect values for the filter 
        # 1. For categorical columns, use the values. Update the value --> category mapping.
        # 2. For numerical columns, use the mean std and quantiles to create |{ri}| range filters. values = [(r1v1, r1v2), (r2v1, r2v2), ...]. Update the value --> original value mapping.
        if col_stype == stype.categorical:
            value_mapping_dict = {v: cat for v, cat in enumerate(categories)}
            values = list(range(len(categories)))
            # Sort candidate filter values in ascending order
            values.sort()   # Sort by the value itself
            op = 'equality' # Equality operation for categorical columns
        elif col_stype == stype.numerical:
            values = [
                (quantiles[i], quantiles[i + 1]) for i in range(len(quantiles) - 1)
            ]
            values = list(set(values))  # Ensure unique ranges
            # Sort candidate filter values in ascending order.
            values.sort(key=lambda x: (x[0], x[1])) # Sort by the first element of the range, break ties by the second element
            # Shift right bound to avoid range overlap, except for the last range
            eps = 1e-8
            values = [
                (values[i][0], values[i][1]+eps) if i<len(values)-1 else (values[i][0], values[i][1]) for i in range(len(values))
            ]
            # Shift left bound to avoid range overlap, except for the first range or single value ranges
            values = [
                (values[i][0]+eps, values[i][1]) if (i>0 and values[i][0]+eps < values[i][1]) else (values[i][0], values[i][1]) for i in range(len(values))
            ]
            # print(f"Numerical column {col_name} with mean {mean} and std {std}. Quantiles: {quantiles}")
            col_index = node_type_to_col_names_by_stype_dict[col_node_type][col_stype].index(col_name)
            unique_values_in_tf = data[col_node_type].tf.feat_dict[col_stype][:, col_index].unique().numpy()
            if len(unique_values_in_tf) > 100:
                # sparsify so that I get max 100 unique values
                keep_every = math.ceil(len(unique_values_in_tf) / 100)
                unique_values_in_tf_sparse = unique_values_in_tf[::keep_every]
            else:
                unique_values_in_tf_sparse = unique_values_in_tf
            # print(f"Values for numerical columns in tf: {data[col_node_type].tf.feat_dict[col_stype][:, col_index].unique()}")
            value_mapping_dict = {f'{r[0]}-{r[1]}': [
                v for v in unique_values_in_tf_sparse if r[0] <= v <= r[1]
            ] for r in values}
            value_mapping_dict['total-unique-values'] = len(unique_values_in_tf)
            value_mapping_dict['sparse-unique-values'] = len(unique_values_in_tf_sparse)
            op = 'range'  # Range operation for numerical columns
        else:
            # just store dummy values and mapping for the rest of the stypes
            values = []
            value_mapping_dict = {}
            op = 'dummy'  # Dummy operation for unsupported stypes
            print(f"Column {col_name} with stype {col_stype} is not supported for filters. Skipping.")

        # Collect the filter
        candidate_filters.append((col_node_type, col_name, col_stype, op, values))
        
        # Update the value mapping
        value_mapping.append((col_node_type, col_name, value_mapping_dict))

    return candidate_filters, value_mapping

def main(
    model_config_path: str,
    model_params_path: str,
    data_config_path: str,
    task_dir: str,
    column_mask_dir: str,
    result_dir: str,
    explanation_target_type: str,
    elimination_strategy: str,
    reg_eps: float = 0.005,
    epochs: int = 200,
    learning_rate: float = 0.1,
    suffix: str = '',
    column_mask_suffix: str = '',
    joint_learning: bool = True,
) -> bool:
    # Load configuration
    config = load_config(model_config_path)

    # Load dataset and task
    dataset, task, task_parser = load_dataset_and_task(data_config_path) 
    dataset_name = task.dataset.dataset_name
    task_name = task.task_name

    # Construct graph data
    data, col_stats_dict = construct_graph(config, dataset)
    del dataset # Delete dataset to free memory

    # print(col_stats_dict)

    # Load model
    model_to_explain = load_model(config, model_params_path, construct=True, data=data, col_stats_dict=col_stats_dict, task=task)

    # Load model predictions
    predictions = {}
    for split in ['train', 'val', 'test']:
        predictions_path = os.path.join(task_dir, f'predictions_{split}.parquet')
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        predictions[split] = pd.read_parquet(predictions_path)

    # Make explanation task
    explanation_task = prepare_node_explanation_task(task, predictions, explanation_target_type=explanation_target_type)
    del task # Delete original task to free memory

    # Initialize the explainer
    explainer = RDLExplainer(config, model_to_explain, data, explanation_task)

    # Load important columns (projection exp results)
    column_mask_file = os.path.join(column_mask_dir, f'{dataset_name}-{task_name}-{column_mask_suffix}mask_vals.json')
    if not os.path.exists(column_mask_file):
        raise FileNotFoundError(f"Column mask file not found: {column_mask_file}. Please run the projection explanation first.")
    with open(column_mask_file, 'r') as f:
        mask_vals = json.load(f)
    mask_vals = {tuple(k.split('-')): v[-1] for k, v in mask_vals.items()}
    n_mask_vals_above_threshold = sum(1 for v in mask_vals.values() if v > 0.1)
    important_col_ranking = [k for k, v in sorted(mask_vals.items(), key=lambda item: item[1], reverse=True)]
    print(f"Number of column mask values above threshold: {n_mask_vals_above_threshold}")
    important_col_ranking_top_k = important_col_ranking[:n_mask_vals_above_threshold]

    # Collect candidate filters
    candidate_filters, value_mapping = collect_candidate_filters_given_important_columns(data, col_stats_dict, important_col_ranking_top_k)

    # Store candidate filters and value mapping
    candidate_filters_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-important-column-{column_mask_suffix}candidate_filters.txt')
    with open(candidate_filters_output_path, 'w') as f:
        for fltr in candidate_filters:
            f.write(f"{fltr[0]}-{fltr[1]}-{fltr[2]}-{fltr[3]}: {','.join([str(v) for v in fltr[4]])}\n")
    value_mapping_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-important-column-{column_mask_suffix}value_mapping.txt')
    with open(value_mapping_output_path, 'w') as f:
        for col_node_type, col_name, value_mapping_dict in value_mapping:
            f.write(f"{col_node_type}-{col_name}: {','.join([f'{k}:{v}' for k, v in value_mapping_dict.items()])}\n")

    # assert False, "stop here to inspect the candidate filters and value mapping. If they look good, remove this line and continue."

    # Remove dummy filters from candidate filters
    candidate_filters = [f for f in candidate_filters if f[3] != 'dummy']
    if not candidate_filters:
        print(f"No candidate filters found for dataset {dataset_name} and task {task_name}. Exiting.")
        return

    if joint_learning:
        print(f"Learning filter masks jointly for dataset {dataset_name} and task {task_name}...")
        # Learn explanation masks
        mask, mask_vals, metrics = explainer.learn_masks(
                                                    eps = reg_eps,
                                                    explanation_type = 'filter', 
                                                    elimination_strategy = elimination_strategy,
                                                    n_epochs = epochs,
                                                    lr = learning_rate,
                                                    filter_predicates = candidate_filters,
                                                )
        # Store the mask values jointly for all filters
        mask_vals_json = {mask_element_wording(f): m for f, m in mask_vals.items()}
        mask_vals_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-filter-joint-{suffix}mask_vals.json')   
        with open(mask_vals_output_path, 'w') as f:
            json.dump(mask_vals_json, f)

        # Store the metrics
        metrics_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-filter-joint-{suffix}metrics.json')
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f)
    else:
        print(f"Learning filter masks sequentially for dataset {dataset_name} and task {task_name}...")
        for fid, f in enumerate(candidate_filters):
            print(f"Learning filter mask {fid+1}/{len(candidate_filters)}: {f}")
            # Learn explanation masks 
            mask, mask_vals, metrics = explainer.learn_masks(
                                                    eps = reg_eps,
                                                    explanation_type = 'filter', 
                                                    elimination_strategy = elimination_strategy,
                                                    n_epochs = epochs,
                                                    lr = learning_rate,
                                                    filter_predicates = [f],
                                                )

            # Store the mask values
            mask_vals_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-filter-{fid+1}-{suffix}mask_vals.json')
            if f[3] == 'range':
                mask_vals_json = {
                    f'{f[0]}--{f[1]}--in--range--{f[4][0][0]}--{f[4][0][1]}': m for (_, i), m in mask_vals.items()
                }
            elif f[3] == 'equality':
                mask_vals_json = {
                    f'{f[0]}--{f[1]}--equal--to--{f[4][0]}': m for (_, i), m in mask_vals.items()
                }
            else:
                raise ValueError(f"Unsupported filter operation: {f[3]}. Supported operations are 'range' and 'equality'.")
            with open(mask_vals_output_path, 'w') as f:
                json.dump(mask_vals_json, f)

            # Visualize the mask for each filter
            last_mask_vals_json = {k: v[-1] for k, v in mask_vals_json.items()}
            visualize_masks(last_mask_vals_json, dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}mask_vals-filter-{fid+1}.png'))

            # Store the metrics
            metrics_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-filter-{fid+1}-{suffix}metrics.json')
            with open(metrics_output_path, 'w') as f:
                json.dump(metrics, f)
    

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Run explanation mask learning for a GNN model.")
    parser.add_argument("--data_config", type=str, required=True, help="Dataset config yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Model config yaml")
    parser.add_argument("--model_params", type=str, required=True, help="Model parameters pth file")
    parser.add_argument("--task_dir", type=str, required=True, help="Input directory to load predictions from")
    parser.add_argument("--column_mask_dir", type=str, required=True, help="Input directory to load column mask values from")
    parser.add_argument("--result_dir", type=str, required=True, help="Output directory to store masks")
    parser.add_argument("--target_type", type=str, default='soft', choices=['hard', 'soft'], help="Target type for explanation")
    parser.add_argument("--elimination_strategy", type=str, default='zero', choices=['zero', 'avg', 'avg_with_noise', 'permutation_joint', 'permutation_independent', 'batch_permutation_joint', 'batch_permutation_independent'], help="Strategy for feature elimination")
    parser.add_argument("--joint_learning", action='store_true', help="Whether to learn all filters jointly or sequentially")
    parser.add_argument("--reg_eps", type=float, default=0.005, help="Regularization epsilon for perturbation")
    parser.add_argument("--epochs", type=int, default=250, help="Max number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--suffix", type=str, default='', help="Suffix for output files")
    args = parser.parse_args()
    suffix = str(args.suffix)+"-" if args.suffix else ""

    # Create output directory if it does not exist
    os.makedirs(args.result_dir, exist_ok=True)

    # Fix all seeds for reproducibility
    fix_all_seeds(seed=int(args.seed))

    # Run the main function
    main(
        model_config_path=args.model_config,
        model_params_path=args.model_params,
        data_config_path=args.data_config,
        task_dir=args.task_dir,
        column_mask_dir=args.column_mask_dir,
        result_dir=args.result_dir,
        explanation_target_type=args.target_type,
        elimination_strategy=args.elimination_strategy,
        reg_eps=args.reg_eps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        suffix=suffix,
        column_mask_suffix='best-',
        joint_learning=args.joint_learning,
    )