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
from collections import deque

# Explain module imports
from src.explain.explainer import RDLExplainer
from src.explain.explain_utils import make_schema_graph

# Eval imports
from eval_utils import (initialize_explainer, 
                        estimate_fidelity_given_ranking, 
                        visualize_masks, 
                        visualize_predictions, 
                        visualize_fidelity_curves, 
                        combine_stats)


def evaluate_masks(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements_ranking: List[Any],
    kmax: int = 10,
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
):
    """Evaluate explanation fidelity for provided masks."""
    print("Evaluating explanation fidelity for provided masks...")

    # TODO: set k-list externally
    necessary_ks = [0, 1, kstar] # necessarily include 0, 1, and kstar 
    klist = necessary_ks + list(range(2, kstar+3, 2)) # add all even numbers up to kstar+2 (inclusive)
    klist = sorted(list(set(klist)))  # remove duplicates and sort

    # Estimate fidelity using the explanation elements ranking
    fidelity_results, masked_predictions = estimate_fidelity_given_ranking(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        explanation_elements_ranking=explanation_elements_ranking,
        klist=klist,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Visualize prediction histogram for masked predictions
    for split, preds in masked_predictions.items():
        preds_df = pd.DataFrame(preds['predictions_per_sample'].transpose(), columns=[f'pred_{i}' for i in range(preds['predictions_per_sample'].shape[0])])
        preds_df['avg_pred'] = preds_df.mean(axis=1)
        visualize_predictions(preds_df, 'avg_pred', dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-masked_predictions-{split}.png'), group_col='targets' if explanation_task.task_type == TaskType.BINARY_CLASSIFICATION else None)

    # Store fidelity results
    method = f'{explanation_type}_mask'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['explanation_elements_ranking'] = explanation_elements_ranking[:max(klist)]  # Store only the top k elements
        fidelity_results['kstar'] = kstar
        fidelity_results['ktotal'] = len(explanation_elements_ranking)
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_random_subset_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements: List[Any],
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    reps: int = 5,
):
    """Evaluate fidelity for random subset baseline."""
    print("Evaluating random subset baseline...")

    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    fidelity_results = {
        'fidelity_top_k': {},
        'fidelity_var_top_k': {},
    }
    for k in klist:
        fidelity_k_for_reps = {}
        fidelity_var_k_for_reps = {}
        for r in range(reps):
            print(f"Evaluating random subset baseline for k={k}, rep={r+1}/{reps}...")

            # Generate random ranking of explanation elements
            explanation_elements_random_k_subset_indices = np.random.choice(np.arange(len(explanation_elements)), size=k, replace=False)
            explanation_elements_random_k_subset = [e for i, e in enumerate(explanation_elements) if i in explanation_elements_random_k_subset_indices]
            explanation_elements_ranking = explanation_elements_random_k_subset + list(set(explanation_elements) - set(explanation_elements_random_k_subset))
            
            # Estimate fidelity for the top-k elements of the random ranking
            fidelity_results_k, _ = estimate_fidelity_given_ranking(
                explainer=explainer,
                explanation_task=explanation_task,
                explanation_type=explanation_type,
                explanation_elements_ranking=explanation_elements_ranking,
                klist=[k],
                n_samples=n_samples,
                perturbation_strategy=perturbation_strategy,
            )

            # Store fidelity results for this k and rep
            for split in fidelity_results_k['fidelity_top_k'].keys():
                if split not in fidelity_k_for_reps:
                    fidelity_k_for_reps[split] = []
                    fidelity_var_k_for_reps[split] = []
                fidelity_k_for_reps[split].append(fidelity_results_k['fidelity_top_k'][split][0])
                fidelity_var_k_for_reps[split].append(fidelity_results_k['fidelity_var_top_k'][split][0])

        # Combine fidelity results across reps
        for split in fidelity_k_for_reps.keys():
            fdl_k, fdl_var_k = combine_stats(
                means=np.array(fidelity_k_for_reps[split]),
                stds=np.array(fidelity_var_k_for_reps[split])
            )
            if split not in fidelity_results['fidelity_top_k']:
                fidelity_results['fidelity_top_k'][split] = []
                fidelity_results['fidelity_var_top_k'][split] = []
            fidelity_results['fidelity_top_k'][split].append(fdl_k)
            fidelity_results['fidelity_var_top_k'][split].append(fdl_var_k)

    # Add klist to fidelity results
    fidelity_results['k_list'] = klist
    fidelity_results['explanation_elements_ranking'] = None

    # Store fidelity results
    method = f'random_subset'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['reps'] = reps
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_random_ranking_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements_ranking: List[Any],
    klist: List[int],
    kmax: int = 10,
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    reps: int = 5,
):
    """Evaluate fidelity for random ranking baseline."""
    print("Evaluating random ranking baseline...")

    fidelity_for_rep = {}
    fidelity_var_for_rep = {}
    ranking_for_rep = []
    for r in range(reps):
        print(f"Evaluating random ranking baseline, rep {r+1}/{reps}...")

        # Generate random ranking of top-k explanation elements
        print(f"Original explanation elements: {explanation_elements_ranking[:kmax]} (total {len(explanation_elements_ranking)})")
        explanation_elements_ranking_shuffle_top_k_indices = np.random.permutation(np.arange(kmax))
        explanation_elements_ranking_shuffle_top_k = [explanation_elements_ranking[i] for i in explanation_elements_ranking_shuffle_top_k_indices if i < len(explanation_elements_ranking)]
        explanation_elements_ranking = explanation_elements_ranking_shuffle_top_k + list(set(explanation_elements_ranking) - set(explanation_elements_ranking_shuffle_top_k))
        print(f"Shuffled explanation elements: {explanation_elements_ranking[:kmax]} (total {len(explanation_elements_ranking)})")
        
        # Estimate fidelity for the random ranking
        fidelity_results_, _ = estimate_fidelity_given_ranking(
            explainer=explainer,
            explanation_task=explanation_task,
            explanation_type=explanation_type,
            explanation_elements_ranking=explanation_elements_ranking,
            klist=klist,
            n_samples=n_samples,
            perturbation_strategy=perturbation_strategy,
        )

        # Store fidelity results for this rep
        for split in fidelity_results_['fidelity_top_k'].keys():
            if split not in fidelity_for_rep:
                fidelity_for_rep[split] = []
                fidelity_var_for_rep[split] = []
            fidelity_for_rep[split].append(fidelity_results_['fidelity_top_k'][split])
            fidelity_var_for_rep[split].append(fidelity_results_['fidelity_var_top_k'][split])
        ranking_for_rep.append(explanation_elements_ranking[:max(klist)])  # Store only the top k elements of the ranking

    # Combine fidelity results across reps
    fidelity_results = {
        'fidelity_top_k': {},
        'fidelity_var_top_k': {},
        'k_list': klist,
        'explanation_elements_ranking': {f'rep_{r+1}': ranking_for_rep[r] for r in range(reps)}
    }
    for split in fidelity_for_rep.keys():
        fdl, fdl_var= combine_stats(
            means=np.array(fidelity_for_rep[split]),
            stds=np.array(fidelity_var_for_rep[split])
        )
        if split not in fidelity_results['fidelity_top_k']:
            fidelity_results['fidelity_top_k'][split] = []
            fidelity_results['fidelity_var_top_k'][split] = []
        fidelity_results['fidelity_top_k'][split] = fdl.tolist()
        fidelity_results['fidelity_var_top_k'][split] = fdl_var.tolist()

    # Store fidelity results
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name
    method = f'random_top_k_ranking'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['reps'] = reps
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_greedy_subset_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements: List[Any],
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    greedy_subset_source: Optional[str] = None,
    timeout: Optional[int] = 18000, # 5 hours timeout for greedy subset evaluation
):
    """Evaluate fidelity for greedy subset baseline."""
    print("Evaluating greedy subset baseline...")

    start_time = time.time()
    if greedy_subset_source is not None:
        print(f"Using greedy subset source: {greedy_subset_source}")
        # Load the explanation elements from the greedy subset source
        try:
            with open(greedy_subset_source, 'r') as f:
                greedy_subset_res = json.load(f)
                greedy_ranking_up_to_kmax = [tuple(e) for e in greedy_subset_res['explanation_elements_ranking']]
                greedy_ranking = greedy_ranking_up_to_kmax + list(set(explanation_elements) - set(greedy_ranking_up_to_kmax))
                number_of_fidelity_estimates = greedy_subset_res['number_of_fidelity_estimates']
        except FileNotFoundError:
            raise FileNotFoundError(f"Greedy subset source file not found: {greedy_subset_source}")
    else:
        # For each of the explanation elements, compute the fidelity of the explanation using just that element
        fidelity_results_using_single_element = []
        # shuffle the explanation elements list, in case of timeout, we will not be able to evaluate all elements
        np.random.shuffle(explanation_elements)
        for element in explanation_elements:
            print(f"Evaluating greedy subset baseline for element: {element}")
            fidelity_results, _ = estimate_fidelity_given_ranking(
                explainer=explainer,
                explanation_task=explanation_task,
                explanation_type=explanation_type,
                explanation_elements_ranking=[element] + list(set(explanation_elements) - {element}),
                klist=[1],
                n_samples=n_samples,
                perturbation_strategy=perturbation_strategy,
                splits=['train']
            )
            fidelity_results_using_single_element.append(fidelity_results['fidelity_top_k']['train'][0])
            print(f"Fidelity score for element {element}: {fidelity_results_using_single_element[-1]:.4f}")

            # Check if time limit is reached
            if time.time() - start_time > timeout:
                print(f"Timeout reached after {timeout} seconds. Stopping greedy subset evaluation early.")
                exited_early = True
                break
        
        # Sort the elements by their fidelity
        greedy_ranking = [x for _, x in sorted(zip(fidelity_results_using_single_element, explanation_elements), reverse=True)]
        number_of_fidelity_estimates = len(greedy_ranking)

        # If we reached the timeout, we will not be able to compute the full greedy ranking.
        # Add the elements not evaluated yet at the end of the ranking.
        if 'exited_early' in locals():
            remaining_elements = list(set(explanation_elements) - set(greedy_ranking))
            greedy_ranking.extend(remaining_elements)
    
    end_time = time.time()

    print(f"Greedy ranking of explanation elements based on single-element fidelity: {greedy_ranking}. Time taken: {end_time - start_time:.2f} seconds")

    # Now, use this ranking to compute the fidelity for the top-k elements
    fidelity_results, _ = estimate_fidelity_given_ranking(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        explanation_elements_ranking=greedy_ranking,
        klist=klist,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Store fidelity results
    method = f'greedy_subset'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['time'] = end_time - start_time
        fidelity_results['number_of_fidelity_estimates'] = number_of_fidelity_estimates
        fidelity_results['explanation_elements_ranking'] = greedy_ranking # store the entire ranking, since it is computed already. for later use
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_greedy_subset_iterative_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements: List[Any],
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    greedy_subset_iterative_source: Optional[str] = None,
    timeout: Optional[int] = 3600,  # 1 hour timeout for each iteration of greedy subset iterative evaluation
    timeout_total: Optional[int] = 18000, # 5 hours timeout for greedy subset iterative evaluation in total
):
    """
    Evaluate fidelity for greedy subset iterative baseline.
    For each iteration k, 
    - compute the fidelity of the explanation using the current top-(k-1) elements and adding the next most important element.
    - add the most important element to the explanation until we reach the desired kmax
    """
    print("Evaluating greedy subset iterative baseline...")

    start_time = time.time()
    if greedy_subset_iterative_source is not None:
        print(f"Using greedy subset iterative source: {greedy_subset_iterative_source}")
        # Load the explanation elements from the greedy subset iterative source
        try:
            with open(greedy_subset_iterative_source, 'r') as f:
                greedy_iterative_res = json.load(f)
                greedy_iterative_ranking_up_to_kmax = [tuple(e) for e in greedy_iterative_res['explanation_elements_ranking']]
                iterative_greedy_ranking = greedy_iterative_ranking_up_to_kmax + list(set(explanation_elements) - set(greedy_iterative_ranking_up_to_kmax))
                number_of_fidelity_estimates = greedy_iterative_res['number_of_fidelity_estimates']
        except FileNotFoundError:
            raise FileNotFoundError(f"Greedy subset iterative source file not found: {greedy_subset_iterative_source}")
    else:
        top_k_elements = []
        cnt_fidelity_estimates = 0
        kmax = max(klist)
        while len(top_k_elements) < kmax:
            # Check if total timeout is reached
            if time.time() - start_time > timeout_total:
                print(f"Total timeout reached after {timeout_total} seconds. Stopping greedy subset iterative evaluation early.")
                break
            print(f"Current top-k elements: {top_k_elements}. Total elements: {len(top_k_elements)}")
            # For each of the remaining explanation elements, compute the fidelity of the explanation using the current top-k elements and adding that element
            fidelity_results_top_k_plus_one = []
            # shuffle the explanation elements list, in case of timeout, we will not be able to evaluate all elements
            np.random.shuffle(explanation_elements)
            start_time_for_iteration = time.time()
            for element in explanation_elements:
                if element in top_k_elements:
                    continue
                print(f"Evaluating greedy subset iterative baseline for element: {element}")
                cnt_fidelity_estimates += 1
                fidelity_results, _ = estimate_fidelity_given_ranking(
                    explainer=explainer,
                    explanation_task=explanation_task,
                    explanation_type=explanation_type,
                    explanation_elements_ranking=top_k_elements + [element] + list(set(explanation_elements) - set(top_k_elements) - {element}),
                    klist=[len(top_k_elements) + 1],  # fidelity when top-k elements + this element is used
                    n_samples=n_samples,
                    perturbation_strategy=perturbation_strategy,
                    splits=['train']
                )
                fidelity_results_top_k_plus_one.append((fidelity_results['fidelity_top_k']['train'][0], element))
                # Check if time limit is reached
                if time.time() - start_time_for_iteration > timeout:
                    print(f"Timeout reached after {timeout} seconds for this iteration. Stopping the iteration early.")
                    break
            # Sort the elements by their fidelity, choose the element with the highest fidelity
            fidelity_results_top_k_plus_one.sort(key=lambda x: x[0], reverse=True)
            if fidelity_results_top_k_plus_one:
                best_element = fidelity_results_top_k_plus_one[0][1]
                top_k_elements.append(best_element)
                print(f"Best element to add: {best_element} with fidelity {fidelity_results_top_k_plus_one[0][0]:.4f}")
            else:
                print("No elements left to evaluate. Stopping the greedy subset iterative evaluation.")
                break

        iterative_greedy_ranking = top_k_elements + list(set(explanation_elements) - set(top_k_elements))
        number_of_fidelity_estimates = cnt_fidelity_estimates
        
    end_time = time.time()
    
    print(f"Greedy iterative ranking of explanation elements based on iterative greedy subset fidelity: {iterative_greedy_ranking}. Time taken: {end_time - start_time:.2f} seconds")

    # Now, use this ranking to compute the fidelity for the top-k elements
    fidelity_results, _ = estimate_fidelity_given_ranking(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        explanation_elements_ranking=iterative_greedy_ranking,
        klist=klist,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Store fidelity results
    method = f'greedy_subset_iterative'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['time'] = end_time - start_time
        fidelity_results['number_of_fidelity_estimates'] = number_of_fidelity_estimates
        fidelity_results['explanation_elements_ranking'] = iterative_greedy_ranking
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results


def evaluate_PFI_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements: List[Any],
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    PFI_source: Optional[str] = None,
    timeout: Optional[int] = 18000, # 5 hours timeout for PFI evaluation
):
    """Evaluate fidelity for PFI baseline."""
    print("Evaluating PFI baseline...")

    start_time = time.time()
    if PFI_source is not None:
        print(f"Using PFI source: {PFI_source}")
        # Load the explanation elements from the PFI source
        try:
            with open(PFI_source, 'r') as f:
                PFI_res = json.load(f)
                PFI_ranking_up_to_kmax = [tuple(e) for e in PFI_res['explanation_elements_ranking']]
                PFI_ranking = PFI_ranking_up_to_kmax + list(set(explanation_elements) - set(PFI_ranking_up_to_kmax))
                number_of_fidelity_estimates = PFI_res['number_of_fidelity_estimates']
        except FileNotFoundError:
            raise FileNotFoundError(f"PFI source file not found: {PFI_source}")
    else:
        # For each of the explanation element, compute the fidelity of the explanation using all but that element
        fidelity_results_using_all_but_single_element_only_one_permutation_per_element = []
        # shuffle the explanation elements list, in case of timeout, we will not be able to evaluate all elements
        np.random.shuffle(explanation_elements)
        for element in explanation_elements:
            print(f"Evaluating PFI baseline for element: {element}")
            fidelity_results, _ = estimate_fidelity_given_ranking(
                explainer=explainer,
                explanation_task=explanation_task,
                explanation_type=explanation_type,
                explanation_elements_ranking=list(set(explanation_elements) - {element}) + [element],
                klist=[len(explanation_elements)-1], # fidelity when all but this element is used
                n_samples=1,
                perturbation_strategy=perturbation_strategy,
                splits=['train']
            )
            fidelity_results_using_all_but_single_element_only_one_permutation_per_element.append(fidelity_results['fidelity_top_k']['train'][0])
            print(f"Fidelity score for all-but element {element}: {fidelity_results_using_all_but_single_element_only_one_permutation_per_element[-1]:.4f}")

            # Check if time limit is reached
            if time.time() - start_time > timeout:
                print(f"Timeout reached after {timeout} seconds. Stopping PFI evaluation early.")
                exited_early = True
                break
        
        # Sort the elements by their fidelity, lower fidelity means higher importance
        PFI_ranking = [x for _, x in sorted(zip(fidelity_results_using_all_but_single_element_only_one_permutation_per_element, explanation_elements), reverse=False)]
        number_of_fidelity_estimates = len(PFI_ranking)

        # If we reached the timeout, we will not be able to compute the full PFI ranking.
        # Add the elements not evaluated yet at the end of the ranking.
        if 'exited_early' in locals():
            remaining_elements = list(set(explanation_elements) - set(PFI_ranking))
            PFI_ranking.extend(remaining_elements)
    
    end_time = time.time()

    print(f"PFI ranking of explanation elements based on all-but-single-element fidelity: {PFI_ranking}. Time taken: {end_time - start_time:.2f} seconds")

    # Now, use this ranking to compute the fidelity for the top-k elements
    fidelity_results, _ = estimate_fidelity_given_ranking(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        explanation_elements_ranking=PFI_ranking,
        klist=klist,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    # Store fidelity results
    method = f'PFI'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['time'] = end_time - start_time
        fidelity_results['number_of_fidelity_estimates'] = number_of_fidelity_estimates
        fidelity_results['explanation_elements_ranking'] = PFI_ranking # store the entire ranking, since it is computed already. for later use
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results


def evaluate_schema_proximity_ranking_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements: List[Any],
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    result_dir: str = './results',
    reps: int = 5,
):
    """Evaluate fidelity for schema proximity ranking baseline."""

    # Rank the explanation elements (table name, column name) based on their proximity to the prediction table in the schema graph.
    def rank_explanation_elements_by_proximity(
        schema_graph: Any,
        prediction_entity: str,
        elements: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Rank elements based on their proximity to the prediction entity in the schema graph."""
        # Traverse the schema graph in breadth-first manner to get a ranking of entity types. Break ties randomly.
        if prediction_entity not in schema_graph:
            raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
        queue = deque([prediction_entity])
        visited = set()
        entity_proximity_ranking = []
        while queue:
            current_entity = queue.popleft()
            if current_entity in visited:
                continue
            visited.add(current_entity)
            neighbors = list(set([edge['dst'] for edge in schema_graph[current_entity]]))
            entity_proximity_ranking.append(current_entity)
            # Randomly shuffle neighbors to break ties
            np.random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        # Now, rank the elements (entity type, attr) based on their proximity to the prediction entity. Break ties among attributes of the same entity type randomly.
        element_proximity_ranking = []
        for entity in entity_proximity_ranking:
            # Get all attributes of the entity in the elements
            entity_attrs = [e for e in elements if e[0] == entity]
            if entity_attrs:
                # Randomly shuffle the attributes to break ties
                np.random.shuffle(entity_attrs)
                element_proximity_ranking.extend(entity_attrs)

        return element_proximity_ranking
            
    database = explanation_task.dataset.db
    schema_graph = make_schema_graph(database, directed=False)
    prediction_entity = explanation_task.entity_table
    print(f"Ranking explanation elements by proximity to prediction entity {prediction_entity} in schema graph...")

    fidelity_for_rep = {}
    fidelity_var_for_rep = {}
    ranking_for_rep = []
    time_for_rep = []
    for r in range(reps):
        print(f"Ranking explanation elements by proximity, rep {r+1}/{reps}...")

        start_time = time.time()

        # Rank the explanation elements by proximity to the prediction entity
        explanation_elements_proximity_ranking = rank_explanation_elements_by_proximity(
            schema_graph=schema_graph,
            prediction_entity=prediction_entity,
            elements=explanation_elements
        )

        end_time = time.time()

        print(f"Proximity ranking of explanation elements: {explanation_elements_proximity_ranking[:klist[-1]]}... (total {len(explanation_elements_proximity_ranking)})")
        
        # Estimate fidelity for the proximity ranking
        fidelity_results, _ = estimate_fidelity_given_ranking(
            explainer=explainer,
            explanation_task=explanation_task,
            explanation_type=explanation_type,
            explanation_elements_ranking=explanation_elements_proximity_ranking,
            klist=klist,
            n_samples=n_samples,
            perturbation_strategy=perturbation_strategy,
        )

        # Store fidelity results for this rep
        for split in fidelity_results['fidelity_top_k'].keys():
            if split not in fidelity_for_rep:
                fidelity_for_rep[split] = []
                fidelity_var_for_rep[split] = []
            fidelity_for_rep[split].append(fidelity_results['fidelity_top_k'][split])
            fidelity_var_for_rep[split].append(fidelity_results['fidelity_var_top_k'][split])
        ranking_for_rep.append(explanation_elements_proximity_ranking[:max(klist)])  # Store only the top k elements of the ranking
        time_for_rep.append(end_time - start_time)

    # Combine fidelity results across reps
    fidelity_results = {
        'fidelity_top_k': {},
        'fidelity_var_top_k': {},
        'k_list': klist,
        'explanation_elements_ranking': {f'rep_{r+1}': ranking_for_rep[r] for r in range(reps)}
    }
    for split in fidelity_for_rep.keys():
        fdl, fdl_var = combine_stats(
            means=np.array(fidelity_for_rep[split]),
            stds=np.array(fidelity_var_for_rep[split])
        )
        fidelity_results['fidelity_top_k'][split] = fdl.tolist()
        fidelity_results['fidelity_var_top_k'][split] = fdl_var.tolist()

    # Store fidelity results
    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name
    method = f'schema_proximity_ranking'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['time'] = np.mean(time_for_rep)
        fidelity_results['reps'] = reps
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def main(
    data_config_path: str,
    model_config_path: str,
    model_params_path: str,
    task_dir: str,
    mask_dir: str,
    result_dir: str,
    explanation_type: str = 'column',
    explanation_target_type: str = 'soft',
    n_fidelity_estimation_samples: int = 10,
    perturbation_strategy: str = 'permutation_independent',
    suffix: str = '',
    reps: int = 5,
    greedy_base_result: Optional[str] = None,
    greedy_iterative_base_result: Optional[str] = None,
    pfi_base_result: Optional[str] = None,
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

    # Load explanation masks learning metrics
    metrics_file = os.path.join(mask_dir, f'{dataset_name}-{task_name}-{suffix}-metrics.json')
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    time_to_learn_masks = metrics['time']
    print(f"Time to learn masks: {time_to_learn_masks:.2f} seconds")

    # Load explanation masks
    mask_file = os.path.join(mask_dir, f'{dataset_name}-{task_name}-{suffix}-mask_vals.json')
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    with open(mask_file, 'r') as f:
        mask_vals = json.load(f)
    mask_vals = {tuple(k.split('-')): v[-1] for k, v in mask_vals.items()}
    n_mask_vals_above_threshold = sum(1 for v in mask_vals.values() if v > 0.1)
    mask_key_ranking = [k for k, v in sorted(mask_vals.items(), key=lambda item: item[1], reverse=True)]
    mask_keys = list(mask_vals.keys())
    kstar = n_mask_vals_above_threshold
    print(f"Number of mask values above threshold kstar = {kstar}. Number of important columns = {kstar}")

    # Visualize mask values
    visualize_masks(mask_vals, dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-mask_vals.png'))

    run_evaluation = True

    # Compute fidelity for explanation masks
    fidelity_results_from_masks_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{explanation_type}_mask-fidelity_results.json')    
    if run_evaluation and not os.path.exists(fidelity_results_from_masks_file):
        evaluate_masks(explainer, explanation_task, explanation_type, mask_key_ranking, kmax=kstar, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, perturbation_strategy=perturbation_strategy)
    if os.path.exists(fidelity_results_from_masks_file):
        with open(fidelity_results_from_masks_file, 'r') as f:
            fidelity_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for masks not found: {fidelity_results_from_masks_file}")
    klist = fidelity_results['k_list'][1:]  # Exclude k=0

    # Compute fidelity for PFI baseline
    fidelity_PFI_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-PFI-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_PFI_file):
        evaluate_PFI_baseline(explainer, explanation_task, explanation_type, mask_keys, klist=klist, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, perturbation_strategy=perturbation_strategy, PFI_source=pfi_base_result)
    if os.path.exists(fidelity_PFI_file):
        with open(fidelity_PFI_file, 'r') as f:
            fidelity_PFI_results = json.load(f)
            time_to_learn_PFI_ranking = fidelity_PFI_results['time']
            print(f"Time to learn PFI ranking: {time_to_learn_PFI_ranking:.2f} seconds")
    else:
        raise FileNotFoundError(f"Fidelity results file for PFI not found: {fidelity_PFI_file}")

    # Compute fidelity for greedy baseline
    fidelity_greedy_subset_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-greedy_subset-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_greedy_subset_file):
        evaluate_greedy_subset_baseline(explainer, explanation_task, explanation_type, mask_keys, klist=klist, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, perturbation_strategy=perturbation_strategy, greedy_subset_source=greedy_base_result)
    if os.path.exists(fidelity_greedy_subset_file):
        with open(fidelity_greedy_subset_file, 'r') as f:
            fidelity_greedy_subset_results = json.load(f)
            time_to_learn_greedy_ranking = fidelity_greedy_subset_results['time']
            print(f"Time to learn greedy ranking: {time_to_learn_greedy_ranking:.2f} seconds")
    else:
        raise FileNotFoundError(f"Fidelity results file for greedy subset not found: {fidelity_greedy_subset_file}")

    # Compute fidelity for greedy iterative baseline
    fidelity_greedy_subset_iterative_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-greedy_subset_iterative-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_greedy_subset_iterative_file):
        evaluate_greedy_subset_iterative_baseline(explainer, explanation_task, explanation_type, mask_keys, klist=klist, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, perturbation_strategy=perturbation_strategy, greedy_subset_iterative_source=greedy_iterative_base_result)
    if os.path.exists(fidelity_greedy_subset_iterative_file):
        with open(fidelity_greedy_subset_iterative_file, 'r') as f:
            fidelity_greedy_subset_iterative_results = json.load(f)
            time_to_learn_greedy_iterative_ranking = fidelity_greedy_subset_iterative_results['time']
            print(f"Time to learn greedy iterative ranking: {time_to_learn_greedy_iterative_ranking:.2f} seconds")
    else:
        raise FileNotFoundError(f"Fidelity results file for greedy subset iterative not found: {fidelity_greedy_subset_iterative_file}")

    # Compute fidelity for random baseline
    fidelity_random_subset_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-random_subset-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_random_subset_file):
        evaluate_random_subset_baseline(explainer, explanation_task, explanation_type, mask_keys, klist=klist, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, reps=reps, perturbation_strategy=perturbation_strategy)
    if os.path.exists(fidelity_random_subset_file):
        with open(fidelity_random_subset_file, 'r') as f:
            fidelity_random_subset_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for random subset not found: {fidelity_random_subset_file}")

    # # Compute fidelity for random ranking baseline
    # fidelity_random_ranking_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-random_top_k_ranking-fidelity_results.json')

    # Compute fidelity for proximity ranking baseline
    fidelity_schema_proximity_ranking_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-schema_proximity_ranking-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_schema_proximity_ranking_file):
        evaluate_schema_proximity_ranking_baseline(explainer, explanation_task, explanation_type, mask_keys, klist=klist, n_samples=n_fidelity_estimation_samples, result_dir=result_dir, reps=reps, perturbation_strategy=perturbation_strategy)
    if os.path.exists(fidelity_schema_proximity_ranking_file):
        with open(fidelity_schema_proximity_ranking_file, 'r') as f:
            fidelity_schema_proximity_ranking_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for schema proximity ranking not found: {fidelity_schema_proximity_ranking_file}")

    # Visualize fidelity curves
    for split in ['test']:
        visualize_fidelity_curves(
            fidelity_results = {'mask': fidelity_results, 
                                'random_subset': fidelity_random_subset_results, 
                                'greedy_subset': fidelity_greedy_subset_results,
                                'greedy_subset_iterative': fidelity_greedy_subset_iterative_results, 
                                'PFI': fidelity_PFI_results,
                                'schema_proximity_ranking': fidelity_schema_proximity_ranking_results},
            dataset_name=dataset_name,
            task_name=task_name,
            file_path=os.path.join(result_dir, f'{dataset_name}-{task_name}-fidelity_curves-{split}.png'),
            split=split,
            kstar=kstar,
        )

    # Time comparison
    time_results = {'mask': time_to_learn_masks,
                    'greedy_subset': fidelity_greedy_subset_results['time'],
                    'greedy_subset_iterative': fidelity_greedy_subset_iterative_results['time'],
                    'PFI': fidelity_PFI_results['time'],
                    'schema_proximity_ranking': fidelity_schema_proximity_ranking_results['time'],}
    time_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-time_comparison_results.json')
    with open(time_results_file, 'w') as f:
        json.dump(time_results, f, indent=4)

    print(f"Evaluation completed for dataset {dataset_name} and task {task_name}. Results stored in {result_dir}.")


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Run GNN training and inference")
    parser.add_argument("--data_config", type=str, required=True, help="Dataset config yaml")
    parser.add_argument("--model_config", type=str, required=True, help="Model config yaml")
    parser.add_argument("--model_params", type=str, required=True, help="Model parameters pth file")
    parser.add_argument("--task_dir", type=str, required=True, help="Input directory to load predictions from")
    parser.add_argument("--mask_dir", type=str, required=True, help="Input directory to load masks from")
    parser.add_argument("--result_dir", type=str, required=True, help="Output directory to store results")
    parser.add_argument("--exp_type", type=str, default='column', choices=['column', 'column,fkpk'], help="Type of explanations to evaluate")
    parser.add_argument("--target_type", type=str, default='soft', choices=['hard', 'soft'], help="Target type for explanation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_fidelity_estimation_samples", type=int, default=5, help="Number of samples for fidelity estimation")
    parser.add_argument("--suffix", type=str, default='best', help="Suffix for result files")
    parser.add_argument("--perturbation_strategy", type=str, default='permutation_independent', choices=['permutation_independent', 'permutation_joint'], help="Perturbation strategy for explanation evaluation")
    parser.add_argument("--greedy_base_result", type=str, default=None, help="Path to the greedy subset source file (if available)")
    parser.add_argument("--greedy_iterative_base_result", type=str, default=None, help="Path to the greedy subset iterative source file (if available)")
    parser.add_argument("--pfi_base_result", type=str, default=None, help="Path to the PFI source file (if available)")
    args = parser.parse_args()

    print(f"Evaluating projection explanations from {args.mask_dir} for task {args.task_dir}.")

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
        mask_dir=args.mask_dir,
        result_dir=args.result_dir,
        explanation_type=args.exp_type,
        explanation_target_type=args.target_type,
        n_fidelity_estimation_samples=args.n_fidelity_estimation_samples,
        perturbation_strategy=args.perturbation_strategy,
        suffix=args.suffix,
        greedy_base_result=args.greedy_base_result,
        greedy_iterative_base_result=args.greedy_iterative_base_result,
        pfi_base_result=args.pfi_base_result,
    )