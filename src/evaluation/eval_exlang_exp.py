import math
import os
import sys
import time
import copy
import json
from typing import Any, Dict, Optional, Tuple, Union, List

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_frame import stype
from collections import deque
import matplotlib.pyplot as plt

# Explain module imports
from src.explain.explainer import RDLExplainer
from src.explain.explain_utils import make_schema_graph

# Eval imports
from eval_utils import (initialize_explainer, 
                        estimate_fidelity_given_boolean_mask, 
                        get_time_to_learn_masks,
                        calculate_explanation_size,
                        construct_boolean_mask_from_explanation_elements_ranking,
                        construct_row_boolean_mask,
                        get_important_projections,
                        get_important_joins)

def evaluate(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_language: str,
    n_samples: int = 10,
    result_dir: str = './results',
    mask_dir: str = './masks',
    explanation_task_name: str = 'top_1_model_small_explanation_task_2',
    selection_mask_hard_mask_vals: Optional[Dict[str, torch.Tensor]] = None,
    time_to_learn_masks: Optional[Dict[str, Any]] = None,
    filter_suffix: str = '',
    reduce_cols: float = 1.0,
    reduce_joins: float = 1.0,
    suffix: str = '',
):
    """Evaluate explanation fidelity for provided explanation languages."""
    print(f"Evaluating explanation fidelity for provided explanation language {explanation_language}...")
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # For each explanation language, produce the explanation_type, perturbation_strategy (lists)
    if explanation_language == 'Proj':
        explanation_type = ['column']
        perturbation_strategy = ['permutation_independent']
    elif explanation_language == 'Join':
        explanation_type = ['fkpk']
        perturbation_strategy = ['foreign_key_uniform_random']
    elif explanation_language == 'Select':
        explanation_type = ['row']
        perturbation_strategy = ['permutation_independent', 'permutation_independent']
    elif explanation_language == 'ProjSelect':
        explanation_type = ['column', 'row']
        perturbation_strategy = ['permutation_independent', 'permutation_independent']
    elif explanation_language == 'JoinProj':
        explanation_type = ['fkpk', 'column']
        perturbation_strategy = ['foreign_key_uniform_random', 'permutation_independent']
    elif explanation_language == 'JoinProjSelect':
        explanation_type = ['fkpk', 'column', 'row']
        perturbation_strategy = ['foreign_key_uniform_random', 'permutation_independent', 'permutation_independent']
    elif explanation_language == 'Null': # estimate fidelity with no explanations, perturb all columns and FKPK pairs
        explanation_type = ['fkpk', 'column']
        perturbation_strategy = ['foreign_key_uniform_random', 'permutation_independent']
    else:
        raise ValueError(f"Unknown explanation language: {explanation_language}")

    # Get important projections
    important_columns, column_ranking, column_kstar = get_important_projections(dataset_name, task_name, mask_dir, reduce=reduce_cols)
    print(f"Important columns: {important_columns}, k*={column_kstar}")
    if not column_ranking: # if no important columns are found, use the default column ranking
        column_ranking = list(explainer.initialize_masks('column').keys())
    # Get important joins
    important_joins, join_ranking, join_kstar = get_important_joins(dataset_name, task_name, mask_dir, reduce=reduce_joins)
    print(f"Important joins: {important_joins}, k*={join_kstar}")
    if not join_ranking: # if no important joins are found, use the default join ranking
        join_ranking = list(explainer.initialize_masks('fkpk').keys())
    # Get the boolean masks for row-wise explanations
    row_boolean_mask, all_true_mask, all_false_mask = construct_row_boolean_mask(
        explainer=explainer,
        explanation_task=explanation_task,
        hard_mask_vals=selection_mask_hard_mask_vals if selection_mask_hard_mask_vals is not None else {'filter_predicate': [], 'hard_mask': []},
        result_dir=result_dir,
        suffix=filter_suffix,
        visualize=False,
    )
    total_rows = sum(mask.numel() for mask in row_boolean_mask.values())
    true_rows = sum(torch.sum(mask).item() for mask in row_boolean_mask.values())
    filter_predicates = selection_mask_hard_mask_vals['filter_predicate'] if selection_mask_hard_mask_vals is not None else []
    print(f"Important rows: {true_rows} / {total_rows}.")
    for filter_predicate, mask in zip(filter_predicates, selection_mask_hard_mask_vals['hard_mask'] if selection_mask_hard_mask_vals is not None else []):
        print(f"Filter predicate applied: {filter_predicate}, selected rows: {torch.sum(mask).item()} / {mask.numel()}. Hard mask values: {mask.tolist()}")
    
    # For each explanation language, collect the boolean mask
    boolean_mask = {}
    for exp in explanation_type:
        if explanation_language == 'Null':
            # For 'Null', we produce the dummy boolean mask with all False values for both column and fkpk
            if exp == 'column':
                boolean_mask[exp] = construct_boolean_mask_from_explanation_elements_ranking(column_ranking, k=0)
            elif exp == 'fkpk':
                boolean_mask[exp] = construct_boolean_mask_from_explanation_elements_ranking(join_ranking, k=0)
            else:
                raise ValueError(f"Unknown explanation type for Null language: {exp}")
        else:
            if exp == 'column':
                boolean_mask[exp] = construct_boolean_mask_from_explanation_elements_ranking(column_ranking, k=column_kstar)
                assert all(boolean_mask[exp][col] for col in important_columns), \
                    f"Boolean mask for {exp} does not have True values for all important columns: {important_columns}"
            elif exp == 'fkpk':
                boolean_mask[exp] = construct_boolean_mask_from_explanation_elements_ranking(join_ranking, k=join_kstar)
                assert all(boolean_mask[exp][join] for join in important_joins), \
                    f"Boolean mask for {exp} does not have True values for all important joins: {important_joins}"
            elif exp == 'row':
                assert selection_mask_hard_mask_vals is not None, "selection_mask_hard_mask_vals must be provided for 'row' explanation type"
                boolean_mask[exp] = row_boolean_mask
                n_positive_predicates = 0
                for filter_mask in selection_mask_hard_mask_vals['hard_mask']:
                    n_positive_predicates += torch.sum(filter_mask).item()

    # Validate the boolean masks
    assert len(boolean_mask) > 0, "No boolean masks were constructed."
    print(f"Boolean masks constructed for: {list(boolean_mask.keys())}, following the explanation language selection: {explanation_language}")
    for exp, mask in boolean_mask.items():
        print(f"Boolean mask for {exp} has {sum(torch.sum(m).item() for m in mask.values())} / {sum(m.numel() for m in mask.values())} True values.")

    # Estimate fidelity using boolean mask
    fidelity_results, masked_predictions = estimate_fidelity_given_boolean_mask(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        boolean_mask=boolean_mask,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )

    # Get database stats
    all_relations = list(set([r for r,c in column_ranking]))
    n_tuples_per_relation={relation: all_true_mask[relation].numel() for relation in all_relations}
    n_data_columns_per_relation = {relation: len([c for r,c in column_ranking if r == relation]) for relation in all_relations}
    n_FKs=len(join_ranking) / 2

    # Store fidelity results
    method = f'mask'
    if suffix:
        fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{explanation_language}-{method}-{suffix}-fidelity_results.json')
    else:
        fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{explanation_language}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['explanation_type'] = explanation_type
        fidelity_results['perturbation_strategy'] = perturbation_strategy
        fidelity_results['explanation_language'] = explanation_language
        size_stats = calculate_explanation_size(
            boolean_mask=boolean_mask,
            n_selection_predicates=n_positive_predicates if 'row' in boolean_mask else 0,
            n_tuples_per_relation=n_tuples_per_relation,
            n_data_columns_per_relation=n_data_columns_per_relation,
            n_FK_all=n_FKs
        )
        fidelity_results.update(size_stats)
        fidelity_results.update(time_to_learn_masks)
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def get_time_to_learn_masks_for_all_mask_types(
    dataset_name: str,
    task_name: str,
    base_mask_dir: str,
    filter_mask_dir: str = None,
    filter_suffix: str = '',
):
    try:
        # TODO: properly handle the case when filter_mask_dir is None
        time_to_learn_filter_masks = get_time_to_learn_masks(dataset_name, task_name, filter_mask_dir, suffix=filter_suffix)
        if isinstance(time_to_learn_filter_masks, list):
            time_to_learn_filter_masks_total = sum(time_to_learn_filter_masks)
            time_to_learn_filter_masks_average = sum(time_to_learn_filter_masks) / len(time_to_learn_filter_masks)
        else:
            time_to_learn_filter_masks_total = time_to_learn_filter_masks
            time_to_learn_filter_masks_average = time_to_learn_filter_masks
    except FileNotFoundError:
        time_to_learn_filter_masks, time_to_learn_filter_masks_total, time_to_learn_filter_masks_average = None, None, None
    try:
        time_to_learn_column_masks = get_time_to_learn_masks(dataset_name, task_name, os.path.join(base_mask_dir, 'column'))
    except FileNotFoundError:
        print(f"Column masks not found for {dataset_name} - {task_name}. Returning None for time to learn column masks.")
        time_to_learn_column_masks = None
    try:
        time_to_learn_fkpk_masks = get_time_to_learn_masks(dataset_name, task_name, os.path.join(base_mask_dir, 'fkpk'))
    except FileNotFoundError:
        print(f"FKPK masks not found for {dataset_name} - {task_name}. Returning None for time to learn FKPK masks.")
        time_to_learn_fkpk_masks = None
    time_stats = {
        'time_to_learn_filter_masks': time_to_learn_filter_masks,
        'time_to_learn_filter_masks_total': time_to_learn_filter_masks_total,
        'time_to_learn_filter_masks_average': time_to_learn_filter_masks_average,
        'time_to_learn_column_masks': time_to_learn_column_masks,
        'time_to_learn_fkpk_masks': time_to_learn_fkpk_masks,
    }
    return time_stats

def parse_filter_masks(
    dataset_name: str,
    task_name: str,
    mask_dir: str,
    suffix: str = '',
):
    # Load filter masks
    mask_file = os.path.join(mask_dir, f'{dataset_name}/{task_name}/{dataset_name}-{task_name}-{suffix}-mask_vals.json')
    if not os.path.exists(mask_file):
        print(f"Mask file {mask_file} does not exist. Returning empty hard mask values.")
        return None
    with open(mask_file, 'r') as f:
        mask_vals = json.load(f)

    # Process the mask values to create soft mask values per filter predicate
    soft_mask_vals = {}
    for k, v in mask_vals.items():
        k = tuple(k.split('--'))
        if f'{k[2]}-{k[3]}' == 'in-range':
            if (k[0], k[1], stype.numerical, 'range') not in soft_mask_vals:
                soft_mask_vals[(k[0], k[1], stype.numerical, 'range')] = {}
                soft_mask_vals[(k[0], k[1], stype.numerical, 'range')]['selection values'] = [(float(k[4]), float(k[5]))]
                soft_mask_vals[(k[0], k[1], stype.numerical, 'range')]['mask values'] = [v[-1]]
            else:
                soft_mask_vals[(k[0], k[1], stype.numerical, 'range')]['selection values'].append((float(k[4]), float(k[5])))
                soft_mask_vals[(k[0], k[1], stype.numerical, 'range')]['mask values'].append(v[-1])
        elif f'{k[2]}-{k[3]}' == 'equal-to':
            if (k[0], k[1], stype.categorical, 'equality') not in soft_mask_vals:
                soft_mask_vals[(k[0], k[1], stype.categorical, 'equality')] = {}
                soft_mask_vals[(k[0], k[1], stype.categorical, 'equality')]['selection values'] = [float(k[4])]
                soft_mask_vals[(k[0], k[1], stype.categorical, 'equality')]['mask values'] = [v[-1]]
            else:
                soft_mask_vals[(k[0], k[1], stype.categorical, 'equality')]['selection values'].append(float(k[4]))
                soft_mask_vals[(k[0], k[1], stype.categorical, 'equality')]['mask values'].append(v[-1])
        else:
            raise ValueError(f"Unknown filter predicate type: {k[2]}-{k[3]}")

    # For each each filter predicate, 
    # sort the selection values according to the left bound or equality value
    # then sort the mask values according to the selection value sorting
    for k, v in soft_mask_vals.items():
        if k[3] == 'range':
            sorted_indices = np.argsort([x[0] for x in v['selection values']])
            v['selection values'] = [v['selection values'][i] for i in sorted_indices]
            v['mask values'] = [v['mask values'][i] for i in sorted_indices]
        elif k[3] == 'equality':
            sorted_indices = np.argsort(v['selection values'])
            v['selection values'] = [v['selection values'][i] for i in sorted_indices]
            v['mask values'] = [v['mask values'][i] for i in sorted_indices]
    
    # Make the hard mask
    threshold = 0.1
    hard_mask_vals = {'filter_predicate': [], 'hard_mask': []}
    for k, v in soft_mask_vals.items():
        unique_selection_values = list(set(v['selection values']))
        filter_predicate = k + (unique_selection_values,)
        print(f"Creating hard mask for {filter_predicate} with threshold {threshold}")
        hard_mask_vals['filter_predicate'].append(filter_predicate)
        hard_mask_vals['hard_mask'].append(torch.tensor(v['mask values']) > threshold)

    return hard_mask_vals

def main(
    data_config_path: str,
    model_config_path: str,
    model_params_path: str,
    task_dir: str,
    base_mask_dir: str,
    filter_mask_dir: str,
    filter_suffix: str,
    result_dir: str,
    explanation_language: str,
    explanation_target_type: str = 'soft',
    n_fidelity_estimation_samples: int = 10,
    projection_percentage: float = 1.0,
    join_percentage: float = 1.0,
    suffix: str = '',
) -> bool:

    # Initialize the explainer and explanation task
    explainer, explanation_task = initialize_explainer(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        model_params_path=model_params_path,
        task_dir=task_dir,
        explanation_target_type=explanation_target_type
    )
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Parse filter masks to get hard mask values for selection explanations
    selection_mask_hard_mask_vals = parse_filter_masks(dataset_name, task_name, filter_mask_dir, suffix=filter_suffix)

    # Get time to learn masks, for all available mask types
    time_to_learn_masks = get_time_to_learn_masks_for_all_mask_types(
        dataset_name=dataset_name,
        task_name=task_name,
        base_mask_dir=base_mask_dir,
        filter_mask_dir=filter_mask_dir,
        filter_suffix=filter_suffix
    )

    # Compute fidelity for all explanation languages
    print(f"Evaluating explanation language: {explanation_language}")
    assert explanation_language in ['Proj', 'Join', 'Select', 'ProjSelect', 'JoinProj', 'JoinProjSelect', 'Null'], "Unknown explanation language. Supported languages are: Proj, Join, Select, ProjSelect, JoinProj, JoinProjSelect, Null."
    fidelity_results = evaluate(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_language=explanation_language,
        n_samples=n_fidelity_estimation_samples,
        result_dir=result_dir,
        mask_dir=base_mask_dir,
        explanation_task_name=task_name,
        selection_mask_hard_mask_vals=selection_mask_hard_mask_vals,
        time_to_learn_masks=time_to_learn_masks,
        filter_suffix=filter_suffix,
        reduce_cols=projection_percentage,
        reduce_joins=join_percentage,
        suffix=suffix,
    )


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Run GNN training and inference")
    parser.add_argument("--data_config", type=str, required=True, help="Dataset config yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Model config yaml")
    parser.add_argument("--model_params", type=str, required=True, help="Model parameters pth file")
    parser.add_argument("--task_dir", type=str, required=True, help="Input directory to load predictions from")
    parser.add_argument("--filter_mask_dir", type=str, required=True, help="Input directory to load filter masks from")
    parser.add_argument("--filter_suffix", type=str, required=True, help="Suffix for filter masks")
    parser.add_argument("--base_mask_dir", type=str, required=True, help="Input directory for all masks")
    parser.add_argument("--result_dir", type=str, required=True, help="Output directory to store results")
    parser.add_argument("--exlang", type=str, required=True, choices=['Proj', 'Join', 'Select', 'ProjSelect', 'JoinProj', 'JoinProjSelect', 'Null'], help="Explanation language to evaluate")
    parser.add_argument("--target_type", type=str, default='soft', choices=['hard', 'soft'], help="Target type for explanation")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for reproducibility")
    parser.add_argument("--n_fidelity_estimation_samples", type=int, default=5, help="Number of samples for fidelity estimation")
    parser.add_argument("--projection_percentage", type=float, default=1.0, help="Percentage of projections to consider for evaluation (1.0 means all important projections are maintained)")
    parser.add_argument("--join_percentage", type=float, default=1.0, help="Percentage of joins to consider for evaluation (1.0 means all important joins are maintained)")
    parser.add_argument("--suffix", type=str, default='', help="Suffix for the output files (optional)")
    args = parser.parse_args()

    print(f"Evaluating explanations for {args.task_dir} with explanation language {args.exlang}...")
    if args.filter_mask_dir is None:
        print("No filter mask directory provided. Using None for filter masks.")
    else:
        print(f"Using filter mask directory: {args.filter_mask_dir} with filter suffix {args.filter_suffix}.")

    # Create output directory if it does not exist
    os.makedirs(args.result_dir, exist_ok=True)

    # Fix all seeds for reproducibility
    fix_all_seeds(seed=int(args.seed))

    # Run the main function
    main(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        model_params_path=args.model_params,
        task_dir=args.task_dir,
        base_mask_dir=args.base_mask_dir,
        result_dir=args.result_dir,
        explanation_language=args.exlang,
        explanation_target_type=args.target_type,
        n_fidelity_estimation_samples=args.n_fidelity_estimation_samples,
        projection_percentage=args.projection_percentage,
        join_percentage=args.join_percentage,
        filter_mask_dir=args.filter_mask_dir,
        filter_suffix=args.filter_suffix,
        suffix=args.suffix,
    )