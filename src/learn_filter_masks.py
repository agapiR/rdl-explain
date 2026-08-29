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
from torch_geometric.seed import seed_everything

# Explain module imports
from rdl_explain.explain.explainer import RDLExplainer
from rdl_explain.loaders import load_config, load_dataset_and_task, construct_graph, load_model
from rdl_explain.explain.explain_utils import explanation_element_wording, make_explanation_task, node_type_to_col_names_by_stype
# Moved into the package so notebooks can use it without importing this script.
from rdl_explain.explain.filters import (SUPPORTED_STYPES_FOR_FILTERS,
                                         collect_candidate_filters_given_important_columns)

# Eval imports
from evaluation.eval_utils import visualize_masks

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
    # Run with PYTHONPATH=src so `rdl_explain` and `evaluation` resolve.
    # Load configuration
    config = load_config(model_config_path)

    # Load dataset and task
    dataset, task = load_dataset_and_task(data_config_path) 
    dataset_name = task.dataset_name
    task_name = task.task_name

    # Construct graph data
    data, col_stats_dict = construct_graph(config, dataset)
    del dataset # Delete dataset to free memory

    # print(col_stats_dict)

    # Load model
    model_to_explain = load_model(config, model_params_path, construct=True, data=data, col_stats_dict=col_stats_dict, task=task)

    # Make explanation task (make_explanation_task reads predictions_{split}.parquet from task_dir itself)
    explanation_task = make_explanation_task(task, data, task_dir, explanation_target_type=explanation_target_type)
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
        mask_vals_json = {explanation_element_wording(f): m for f, m in mask_vals.items()}
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
    seed_everything(int(args.seed))

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