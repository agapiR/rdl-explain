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
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch_geometric.seed import seed_everything

# Explain module imports
from rdl_explain.explain.explainer import RDLExplainer
from rdl_explain.loaders import load_config, load_dataset_and_task, construct_graph, load_model
from rdl_explain.explain.explain_utils import explanation_element_wording, make_explanation_task

def main(
    model_config_path: str,
    model_params_path: str,
    data_config_path: str,
    task_dir: str,
    result_dir: str,
    explanation_target_type: str,
    explanation_type: str,
    elimination_strategy: str,
    reg_eps: float = 0.005,
    epochs: int = 200,
    learning_rate: float = 0.1,
    suffix: str = '',
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

    # Load model
    model_to_explain = load_model(config, model_params_path, construct=True, data=data, col_stats_dict=col_stats_dict, task=task)

    # Make explanation task (make_explanation_task reads predictions_{split}.parquet from task_dir itself)
    explanation_task = make_explanation_task(task, data, task_dir, explanation_target_type=explanation_target_type)
    del task # Delete original task to free memory

    # Initialize the explainer
    explainer = RDLExplainer(config, model_to_explain, data, explanation_task)

    # Learn explanation masks
    mask, mask_vals, metrics = explainer.learn_masks(
                                            eps = reg_eps,
                                            explanation_type = explanation_type, 
                                            elimination_strategy = elimination_strategy,
                                            n_epochs = epochs,
                                            lr = learning_rate,
                                        )

    # Save the mask values
    mask_vals_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}mask_vals.json')
    mask_vals_json = {explanation_element_wording(k): v for k, v in mask_vals.items()}
    with open(mask_vals_output_path, 'w') as f:
        json.dump(mask_vals_json, f)

    # Save the metrics
    metrics_output_path = os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}metrics.json')
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Run explanation mask learning for a GNN model.")
    parser.add_argument("--data_config", type=str, required=True, help="Dataset config yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Model config yaml")
    parser.add_argument("--model_params", type=str, required=True, help="Model parameters pth file")
    parser.add_argument("--task_dir", type=str, required=True, help="Input directory to load predictions from")
    parser.add_argument("--result_dir", type=str, required=True, help="Output directory to store masks")
    parser.add_argument("--mask_type", type=str, default='column', choices=['column', 'fkpk'], help="Type of mask to learn")
    parser.add_argument("--target_type", type=str, default='soft', choices=['hard', 'soft'], help="Target type for explanation")
    parser.add_argument("--elimination_strategy", type=str, default='permutation_independent', choices=['zero', 'avg', 'avg_with_noise', 'permutation_joint', 'permutation_independent', 'batch_permutation_joint', 'batch_permutation_independent'], help="Strategy for feature elimination")
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
        result_dir=args.result_dir,
        explanation_target_type=args.target_type,
        explanation_type=args.mask_type,
        elimination_strategy=args.elimination_strategy,
        reg_eps=args.reg_eps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        suffix=suffix,
    )