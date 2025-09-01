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
                        visualize_schema_graph_with_important_joins, 
                        combine_stats)

def edges_in_schema_graph(schema_graph: Dict[str, List]) -> List[Tuple[str, str, str]]:
    """Collect all edge types that appear in the schema graph"""
    all_edges = []
    for src, candidate_edges in schema_graph.items():
        for neighbor in candidate_edges:
            all_edges.append((src, neighbor['edge_name'], neighbor['dst']))
    all_edges = list(set(all_edges))  # Remove duplicates (if any)
    return all_edges

def is_schema_subgraph_connected(
    schema_graph: Dict[str, List], 
    edges_to_keep: List[Tuple[str, str, str]], 
    prediction_entity: str
) -> Tuple[bool, List[Tuple[str, str, str]]]:
    """
    Check if the subgraph formed by edges_to_keep is connected and contains the prediction entity.
    If not, return only the edges in edges_to_keep that form a connected subgraph starting from the prediction entity. 
    If yes, return the edges_to_keep.
    In both cases, reorder the edges_to_keep to respect the distance from the prediction entity. 
    Order by distance and then break ties by ranking.
    """
    # Create a subgraph from the schema graph with the edges to keep
    subgraph = {}
    for src, edge_name, dst in edges_to_keep:
        if src not in subgraph:
            subgraph[src] = []
        subgraph[src].append((src, edge_name, dst))

    # Check if the prediction entity is in the subgraph
    if prediction_entity not in subgraph:
        raise ValueError(f"Prediction entity {prediction_entity} not found in subgraph.")

    # Perform traversal of the subgraph starting from the prediction entity
    # For each traversal step, choose the edge higher in the ranking that maintains subgraph connectivity
    candidate_edges = subgraph[prediction_entity]
    connected_edges_to_keep = []
    while candidate_edges:
        ranked_candidate_edges = sorted(candidate_edges, key=lambda x: (x[1], edges_to_keep.index((x[0], x[1], x[2]))))
        # Take the first neighbor (highest ranked edge)
        current_edge = ranked_candidate_edges[0]
        connected_edges_to_keep.append(current_edge)
        candidate_edges.remove(current_edge)
        # Update the candidate edges with the neighbors of the current edge dst
        new_node = current_edge[2]
        if new_node in subgraph:
            for new_candidate_edge in subgraph[new_node]:
                if new_candidate_edge not in connected_edges_to_keep and new_candidate_edge not in candidate_edges:
                    candidate_edges.append(new_candidate_edge)

    if len(connected_edges_to_keep) == len(edges_to_keep):
        is_connected = True
    else:
        is_connected = False

    return is_connected, connected_edges_to_keep    


def evaluate_masks(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    explanation_elements_ranking: List[Any],
    schema_graph: Dict[str, List],
    kmax: int,
    n_samples: int = 10,
    perturbation_strategy: str = 'foreign_key_exchange',
    result_dir: str = './results',
):
    """Evaluate explanation fidelity for provided masks."""
    print("Evaluating explanation fidelity for provided masks...")

    mask_keys_only_f2p_ranking = [edge_type for edge_type in explanation_elements_ranking if 'rev_' not in edge_type[1]]
    mask_keys_only_rev_f2p_ranking = [edge_type for edge_type in explanation_elements_ranking if 'rev_' in edge_type[1]]

    # Verify that the top-k elements in the ranking form a connected component in the schema graph, containing the prediction entity node
    prediction_entity = explanation_task.entity_table
    if prediction_entity not in schema_graph:
        raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
    is_connected, mask_top_kmax_edges = is_schema_subgraph_connected(schema_graph, edges_to_keep=explanation_elements_ranking[:(kmax*2)], prediction_entity=prediction_entity)
    if not is_connected:
        print(f"The top-{kmax} explanation elements do not form a connected subgraph in the schema graph starting from the prediction entity {prediction_entity}. Reducing the top-k elements to only those that form a connected subgraph.")
    else:
        print(f"The top-{kmax} explanation elements form a connected subgraph in the schema graph starting from the prediction entity {prediction_entity}. Traversal order: {mask_top_kmax_edges}")

    mask_top_kmax_edges_only_f2p = [e for e in mask_top_kmax_edges if e in mask_keys_only_f2p_ranking]
    remaining_edges_only_f2p_sorted_by_explanation_elements_ranking = [e for e in mask_keys_only_f2p_ranking if e not in mask_top_kmax_edges_only_f2p]
    mask_keys_only_f2p_new_ranking = mask_top_kmax_edges_only_f2p + remaining_edges_only_f2p_sorted_by_explanation_elements_ranking
    explanation_elements_ranking = mask_keys_only_f2p_new_ranking + mask_keys_only_rev_f2p_ranking

    print(f"Using the following explanation elements ranking for fidelity estimation: {explanation_elements_ranking[:kmax]}...")

    # TODO: set k-list externally
    klist = [k for k in range(0, kmax+2) if k <= len(mask_keys_only_f2p_new_ranking)]

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
        fidelity_results['explanation_elements_ranking'] = mask_keys_only_f2p_new_ranking[:max(klist)]  # Store only the top k elements
        fidelity_results['is_connected'] = is_connected
        fidelity_results['kstar'] = kstar
        fidelity_results['ktotal'] = len(mask_keys_only_f2p_new_ranking)
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_random_expansion_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    schema_graph: Dict[str, List],
    kmax: int,
    klist: List[int],
    n_samples: int = 10,
    perturbation_strategy: str = 'foreign_key_exchange',
    result_dir: str = './results',
    suffix: str = '',
    reps: int = 5,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """Evaluate fidelity for random expansion baseline."""
    print("Evaluating random expansion baseline...")

    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name
    all_edges = edges_in_schema_graph(schema_graph)

    # Generate a random expansion of the schema graph
    # schema_graph : { src: [{'dst': dst1, 'edge_name': label}, {'dst': dst2, 'edge_name': label}, ...], ...}
    random_expansion_top_kmax_edges = []
    visited = set()
    def random_walk(start_node):
        queue = [start_node]
        while queue:
            if len(random_expansion_top_kmax_edges) >= kmax:
                return
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            candidate_edges = schema_graph.get(node, [])
            if not candidate_edges:
                continue
            # Randomly shuffle candidate_edges to create a random order
            np.random.shuffle(candidate_edges)
            for neighbor in candidate_edges:
                dst = neighbor['dst']
                edge_name = neighbor['edge_name']
                edge = (node, edge_name, dst)
                rev_edge = (dst, f'rev_{edge_name}', node) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), node)
                # if the reverse edge is already in the random expansion, skip the edge, else add it to the random expansion
                if rev_edge in random_expansion_top_kmax_edges:
                    continue
                random_expansion_top_kmax_edges.append(edge)
                # Enqueue the neighbor node for further exploration
                # If the random expansion has reached kmax edges, stop the exploration
                if len(random_expansion_top_kmax_edges) < kmax:
                    queue.append(dst)
                else:
                    return

    # Compute fidelity for random expansion baseline for multiple repetitions
    fidelity_for_rep = {}
    fidelity_var_for_rep = {}
    ranking_for_rep = []
    for r in range(reps):
        print(f"Evaluating random expansion baseline, rep={r+1}/{reps}...")

        # Start random walk from the entity table of the explanation task
        prediction_entity = explanation_task.entity_table
        if prediction_entity not in schema_graph:
            raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
        print(f"Starting random walk from prediction entity: {prediction_entity}, max hops: {kmax}")
        # Re-initialize ranking and visited set
        random_expansion_top_kmax_edges = []
        visited = set()
        random_walk(prediction_entity)

        # Use the random mask to create a ranking of explanation elements, 
        # have the ranking start with the top-k elements of the random mask and then the rest
        explanation_elements_ranking = random_expansion_top_kmax_edges + list(set(all_edges) - set(random_expansion_top_kmax_edges))

        # Collect the top-k edges from the random expansion and their reverses
        assert len(random_expansion_top_kmax_edges)==kmax, f"Random expansion produced unexpected number of edges: {len(random_expansion_top_kmax_edges)}. Expected: {kmax}."
        random_expansion_top_kmax_edges += [(dst, f'rev_{edge_name}', src) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), src) for src, edge_name, dst in random_expansion_top_kmax_edges]
        
        print(f"Random expansion top-{kmax} edges: {random_expansion_top_kmax_edges[:kmax]}")
        visualize_schema_graph_with_important_joins(schema_graph, random_expansion_top_kmax_edges, dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-schema_graph_with_random_walk_edges_{r}.png'))

        # Estimate fidelity for the random mask
        fidelity_results, _ = estimate_fidelity_given_ranking(
            explainer=explainer,
            explanation_task=explanation_task,
            explanation_type=explanation_type,
            explanation_elements_ranking=explanation_elements_ranking,
            klist=klist,
            n_samples=n_samples,
            perturbation_strategy=perturbation_strategy,
        )

        # store the random expansion mask for later use
        ranking_for_rep.append(random_expansion_top_kmax_edges[:kmax])

        # store the fidelity results for the repetition
        for split in fidelity_results['fidelity_top_k'].keys():
            if split not in fidelity_for_rep:
                fidelity_for_rep[split] = []
                fidelity_var_for_rep[split] = []
            fidelity_for_rep[split].append(fidelity_results['fidelity_top_k'][split])
            fidelity_var_for_rep[split].append(fidelity_results['fidelity_var_top_k'][split])

    # Combine fidelity results across reps
    fidelity_results = {
        'fidelity_top_k': {},
        'fidelity_var_top_k': {},
        'k_list': klist,
        'explanation_elements_ranking': ranking_for_rep
    }

    for split in fidelity_for_rep.keys():
        fdl, fdl_var = combine_stats(
            means=np.array(fidelity_for_rep[split]),
            stds=np.array(fidelity_var_for_rep[split])
        )
        if split not in fidelity_results['fidelity_top_k']:
            fidelity_results['fidelity_top_k'][split] = []
            fidelity_results['fidelity_var_top_k'][split] = []
        fidelity_results['fidelity_top_k'][split] = fdl.tolist()
        fidelity_results['fidelity_var_top_k'][split] = fdl_var.tolist()
    
    # Store fidelity results
    method = f'random_expansion'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

# def evaluate_random_bfs_baseline(
#     explainer: RDLExplainer,
#     explanation_task: Any,
#     explanation_type: str,
#     schema_graph: Dict[str, List],
#     kmax: int,
#     klist: List[int],
#     n_samples: int = 10,
#     perturbation_strategy: str = 'foreign_key_exchange',
#     result_dir: str = './results',
#     suffix: str = '',
#     reps: int = 5,
# ) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
#     """Evaluate fidelity for random BFS expansion baseline."""
#     print("Evaluating random BFS expansion baseline...")

#     dataset_name = explanation_task.dataset.dataset_name
#     task_name = explanation_task.task_name
#     all_edges = edges_in_schema_graph(schema_graph)

#     # Generate a random BFS expansion of the schema graph
#     # schema_graph : { src: [{'dst': dst1, 'edge_name': label}, {'dst': dst2, 'edge_name': label}, ...], ...}
#     random_bfs_expansion_top_kmax_edges = []
#     visited = set()
#     def bfs(start_node):
#         queue = [start_node]
#         while queue:
#             node = queue.pop(0)
#             if node in visited:
#                 continue
#             if len(random_bfs_expansion_top_kmax_edges) >= kmax:
#                 return
#             visited.add(node)
#             candidate_edges = schema_graph.get(node, [])
#             if not candidate_edges:
#                 continue
#             # Remove candidate_edges that have already been visited to avoid cycles
#             candidate_edges = [n for n in candidate_edges if n['dst'] not in visited]
#             # Randomly shuffle candidate_edges to create a random BFS order
#             np.random.shuffle(candidate_edges)
#             for neighbor in candidate_edges:
#                 dst = neighbor['dst']
#                 edge_name = neighbor['edge_name']
#                 # Add the edge to the random BFS mask
#                 random_bfs_expansion_top_kmax_edges.append((node, edge_name, dst))
#                 # Enqueue the neighbor node for further exploration
#                 if len(random_bfs_expansion_top_kmax_edges) < kmax:
#                     queue.append(dst)
#                 else:
#                     return

#     # Compute fidelity for random BFS expansion baseline for multiple repetitions
#     fidelity_for_rep = {'test': [], 'train': []}
#     fidelity_var_for_rep = {'test': [], 'train': []}
#     ranking_for_rep = []
#     for r in range(reps):
#         print(f"Evaluating random BFS expansion baseline, rep={r+1}/{reps}...")

#         # Start BFS from the entity table of the explanation task
#         prediction_entity = explanation_task.entity_table
#         if prediction_entity not in schema_graph:
#             raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
#         print(f"Starting BFS from prediction entity: {prediction_entity}, max hops: {kmax}")
#         # Re-initialize ranking and visited set
#         random_bfs_expansion_top_kmax_edges = []
#         visited = set()
#         bfs(prediction_entity)

#         # Use the random BFS mask to create a ranking of explanation elements, 
#         # have the ranking start with the top-k elements of the random BFS mask and then the rest
#         explanation_elements_ranking = random_bfs_expansion_top_kmax_edges + list(set(all_edges) - set(random_bfs_expansion_top_kmax_edges))

#         # Collect the top-k edges from the random BFS expansion and their reverses
#         random_bfs_expansion_top_kmax_edges += [(dst, f'rev_{edge_name}', src) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), src) for src, edge_name, dst in random_bfs_expansion_top_kmax_edges]
#         print(f"Random BFS top-{kmax} edges: {random_bfs_expansion_top_kmax_edges[:kmax]}")

#         # Estimate fidelity for the random BFS mask
#         fidelity_results, _ = estimate_fidelity_given_ranking(
#             explainer=explainer,
#             explanation_task=explanation_task,
#             explanation_type=explanation_type,
#             explanation_elements_ranking=explanation_elements_ranking,
#             klist=klist,
#             n_samples=n_samples,
#             perturbation_strategy=perturbation_strategy,
#         )

#         # store the random DFS mask for later use
#         ranking_for_rep.append(random_bfs_expansion_top_kmax_edges[:kmax])

#         # store the fidelity results for the repetition
#         for split in ['test', 'train']:
#             fidelity_for_rep[split].append(fidelity_results['fidelity_top_k'][split])
#             fidelity_var_for_rep[split].append(fidelity_results['fidelity_var_top_k'][split])

#     # Combine fidelity results across reps
#     fidelity_results = {
#         'fidelity_top_k': {},
#         'fidelity_var_top_k': {},
#         'k_list': klist,
#         'explanation_elements_ranking': ranking_for_rep
#     }
#     for split in ['test', 'train']:
#         fdl, fdl_var= combine_stats(
#             means=np.array(fidelity_for_rep[split]),
#             stds=np.array(fidelity_var_for_rep[split])
#         )
#         fidelity_results['fidelity_top_k'][split] = fdl.tolist()
#         fidelity_results['fidelity_var_top_k'][split] = fdl_var.tolist()

#     # Store fidelity results
#     method = f'random_bfs'
#     fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
#     with open(fidelity_results_file, 'w') as f:
#         fidelity_results['method'] = method
#         json.dump(fidelity_results, f, indent=4)

#     return fidelity_results

# def evaluate_random_dfs_baseline(
#     explainer: RDLExplainer,
#     explanation_task: Any,
#     explanation_type: str,
#     schema_graph: Dict[str, List],
#     kmax: int,
#     klist: List[int],
#     n_samples: int = 10,
#     perturbation_strategy: str = 'foreign_key_exchange',
#     result_dir: str = './results',
#     suffix: str = '',
#     reps: int = 5,
# ) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """Evaluate fidelity for random DFS expansion baseline."""
    print("Evaluating random DFS expansion baseline...")

    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name
    all_edges = edges_in_schema_graph(schema_graph)

    # Generate a random DFS expansion of the schema graph
    # schema_graph : { src: [{'dst': dst1, 'edge_name': label}, {'dst': dst2, 'edge_name': label}, ...], ...}
    random_dfs_expansion_top_kmax_edges = []
    visited = set()
    def dfs(node):
        if node in visited or len(random_dfs_expansion_top_kmax_edges) >= kmax:
            return
        visited.add(node)
        candidate_edges = schema_graph.get(node, [])
        if not candidate_edges:
            return
        # Remove candidate_edges that have already been visited to avoid cycles
        candidate_edges = [n for n in candidate_edges if n['dst'] not in visited]
        # Randomly shuffle candidate_edges to create a random DFS order
        np.random.shuffle(candidate_edges)
        for neighbor in candidate_edges:
            dst = neighbor['dst']
            edge_name = neighbor['edge_name']
            # Add the edge to the random DFS mask
            random_dfs_expansion_top_kmax_edges.append((node, edge_name, dst))
            # Recur for the neighbor node
            if len(random_dfs_expansion_top_kmax_edges) < kmax:
                dfs(dst)

    # Compute fidelity for random DFS expansion baseline for multiple repetitions
    fidelity_for_rep = {'test': [], 'train': []}
    fidelity_var_for_rep = {'test': [], 'train': []}
    ranking_for_rep = []
    for r in range(reps):
        print(f"Evaluating random DFS expansion baseline, rep={r+1}/{reps}...")

        # Start DFS from the entity table of the explanation task
        prediction_entity = explanation_task.entity_table
        if prediction_entity not in schema_graph:
            raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
        print(f"Starting DFS from prediction entity: {prediction_entity}, max hops: {kmax}")
        # Re-initialize ranking and visited set
        random_dfs_expansion_top_kmax_edges = []
        visited = set()
        dfs(prediction_entity)

        # Use the random DFS mask to create a ranking of explanation elements, 
        # have the ranking start with the top-k elements of the random DFS mask and then the rest
        explanation_elements_ranking = random_dfs_expansion_top_kmax_edges + list(set(all_edges) - set(random_dfs_expansion_top_kmax_edges))

        # Collect the top-k edges from the random DFS expansion and their reverses
        random_dfs_expansion_top_kmax_edges += [(dst, f'rev_{edge_name}', src) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), src) for src, edge_name, dst in random_dfs_expansion_top_kmax_edges]
        print(f"Random DFS top-{kmax} edges: {random_dfs_expansion_top_kmax_edges[:kmax]}")

        # Estimate fidelity for the random DFS mask
        fidelity_results, _ = estimate_fidelity_given_ranking(
            explainer=explainer,
            explanation_task=explanation_task,
            explanation_type=explanation_type,
            explanation_elements_ranking=explanation_elements_ranking,
            klist=klist,
            n_samples=n_samples,
            perturbation_strategy=perturbation_strategy,
        )

        # store the random DFS mask for later use
        ranking_for_rep.append(random_dfs_expansion_top_kmax_edges[:kmax])

        # store the fidelity results for the repetition
        for split in ['test', 'train']:
            fidelity_for_rep[split].append(fidelity_results['fidelity_top_k'][split])
            fidelity_var_for_rep[split].append(fidelity_results['fidelity_var_top_k'][split])

    # Combine fidelity results across reps
    fidelity_results = {
        'fidelity_top_k': {},
        'fidelity_var_top_k': {},
        'k_list': klist,
        'explanation_elements_ranking': ranking_for_rep
    }
    for split in ['test', 'train']:
        fdl, fdl_var= combine_stats(
            means=np.array(fidelity_for_rep[split]),
            stds=np.array(fidelity_var_for_rep[split])
        )
        fidelity_results['fidelity_top_k'][split] = fdl.tolist()
        fidelity_results['fidelity_var_top_k'][split] = fdl_var.tolist()

    # Store fidelity results
    method = f'random_dfs'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results

def evaluate_greedy_expansion_baseline(
    explainer: RDLExplainer,
    explanation_task: Any,
    explanation_type: str,
    schema_graph: Dict[str, List],
    kmax: int,
    klist: List[int], 
    n_samples: int = 10,
    perturbation_strategy: str = 'foreign_key_exchange',
    result_dir: str = './results',
    greedy_expansion_source: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """
    Expand a connected subgraph greedily starting from the prediction entity. 
    For every k (1, ... , kmax), choose the edge that maximizes the fidelity gain computed on the training split.
    Then estimate the fidelity for the greedy top-kmax edge mask.
    """
    print("Evaluating greedy expansion baseline...")

    dataset_name = explanation_task.dataset.dataset_name
    task_name = explanation_task.task_name

    start_time = time.time()
    cnt = 0  # Counter for fidelity estimates

    # Start with the prediction entity
    prediction_entity = explanation_task.entity_table
    if prediction_entity not in schema_graph:
        raise ValueError(f"Prediction entity {prediction_entity} not found in schema graph.")
    
    greedy_top_kmax_edges = []
    current_edges = [(prediction_entity, edge['edge_name'], edge['dst']) for edge in schema_graph[prediction_entity]]
    all_edges = edges_in_schema_graph(schema_graph)
    
    if greedy_expansion_source is not None:
        print(f"Using greedy expansion source: {greedy_expansion_source}")
        # Load the explanation elements from the greedy expansion source
        try:
            with open(greedy_expansion_source, 'r') as f:
                greedy_expansion_res = json.load(f)
                greedy_top_kmax_edges = [tuple(e) for e in greedy_expansion_res['explanation_elements_ranking']]
        except FileNotFoundError:
            raise FileNotFoundError(f"Greedy subset source file not found: {greedy_subset_source}")
    else:
        while len(greedy_top_kmax_edges) < kmax:
            if not current_edges:
                break

            # Evaluate the fidelity gain for each edge in the current edges
            fidelity_gains = []
            print(f"Evaluating fidelity gains for {len(current_edges)} candidate edges...")
            for edge in current_edges:
                print(f"Evaluating fidelity for edge: {edge}, current greedy top-kmax edges: {greedy_top_kmax_edges}")
                
                # Temporarily add the edge to the mask
                greedy_top_kmax_edges.append(edge)
                
                # Estimate fidelity gain
                fidelity_results, _ = estimate_fidelity_given_ranking(
                    explainer=explainer,
                    explanation_task=explanation_task,
                    explanation_type=explanation_type,
                    explanation_elements_ranking=greedy_top_kmax_edges + list(set(all_edges) - set(greedy_top_kmax_edges)),
                    klist=[len(greedy_top_kmax_edges)],
                    n_samples=n_samples,
                    perturbation_strategy=perturbation_strategy,
                    splits=['train']  # Only compute fidelity gain on training split
                )
                
                # Store the fidelity gain
                fidelity_gain = fidelity_results['fidelity_top_k']['train'][0]  # Assuming we want to maximize training fidelity
                fidelity_gains.append((fidelity_gain, edge))
                cnt += 1  # Increment the counter for fidelity estimates
                
                # Remove the edge from the mask for the next iteration
                greedy_top_kmax_edges.pop()
            
            # Choose the edge with the maximum fidelity gain
            best_gain, best_edge = max(fidelity_gains, key=lambda x: x[0])
            greedy_top_kmax_edges.append(best_edge)
            greedy_top_kmax_edges_reverse = [(dst, f'rev_{edge_name}', src) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), src) for src, edge_name, dst in greedy_top_kmax_edges]
            
            # Update current edges to include candidate_edges of the newly added node
            new_node = best_edge[2]
            if new_node in schema_graph:
                current_edges.extend([(new_node, edge['edge_name'], edge['dst']) for edge in schema_graph[new_node]])
            # Remove the edges already in the greedy ranking along with their reverses, to avoid re-evaluation
            current_edges = [edge for edge in current_edges if ( edge not in greedy_top_kmax_edges ) and ( edge not in greedy_top_kmax_edges_reverse )]
            current_edges = list(set(current_edges))  # Remove duplicates

    end_time = time.time()

    print(f"Greedy expansion subgraph edges: {greedy_top_kmax_edges}. Time taken: {end_time - start_time:.2f} seconds")

    # Collect the top-k edges from the greedy expansion and their reverses
    greedy_top_kmax_edges = greedy_top_kmax_edges[:kmax]
    greedy_top_kmax_edges += [(dst, f'rev_{edge_name}', src) if 'rev_' not in edge_name else (dst, edge_name.replace('rev_', ''), src) for src, edge_name, dst in greedy_top_kmax_edges]
    # Append the remaining edges to the greedy top-kmax edge ranking
    greedy_top_kmax_ranking = greedy_top_kmax_edges + list(set(all_edges) - set(greedy_top_kmax_edges))

    # Visualize the greedy expansion subgraph
    visualize_schema_graph_with_important_joins(schema_graph, greedy_top_kmax_edges, dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-greedy_expansion_schema_graph.png'))

    # Estimate fidelity
    fidelity_results, _ = estimate_fidelity_given_ranking(
        explainer=explainer,
        explanation_task=explanation_task,
        explanation_type=explanation_type,
        explanation_elements_ranking=greedy_top_kmax_ranking,
        klist=klist,
        n_samples=n_samples,
        perturbation_strategy=perturbation_strategy,
    )

    # Store fidelity results
    method = f'greedy_expansion'
    fidelity_results_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{method}-fidelity_results.json')
    with open(fidelity_results_file, 'w') as f:
        fidelity_results['method'] = method
        fidelity_results['time'] = end_time - start_time
        fidelity_results['number_of_fidelity_estimates'] = cnt
        fidelity_results['explanation_elements_ranking'] = greedy_top_kmax_ranking[:max(klist)]  # Store only the top k elements
        json.dump(fidelity_results, f, indent=4)

    return fidelity_results
    
def main(
    data_config_path: str,
    model_config_path: str,
    model_params_path: str,
    task_dir: str,
    mask_dir: str,
    result_dir: str,
    explanation_type: str = 'fkpk',
    explanation_target_type: str = 'soft',
    n_fidelity_estimation_samples: int = 10,
    perturbation_strategy: str = 'foreign_key_exchange',
    suffix: str = '',
    reps: int = 5,
    greedy_base_result: Optional[str] = None,
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
    kstar = n_mask_vals_above_threshold
    kstar_only_f2p = n_mask_vals_above_threshold // 2  # Use only f2p edges (each f2p edge has a corresponding rev_f2p edge with the same mask value)
    print(f"Number of mask values above threshold kstar = {kstar}. Number of important join key pairs = {kstar_only_f2p}")

    database = explanation_task.dataset.db
    schema_graph = make_schema_graph(database, directed=False)

    # Visualize important joins on schema graph
    visualize_schema_graph_with_important_joins(schema_graph, mask_key_ranking[:kstar], dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-schema_graph_with_important_joins.png'))

    # Visualize mask values
    visualize_masks(mask_vals, dataset_name, task_name, os.path.join(result_dir, f'{dataset_name}-{task_name}-{suffix}-mask_vals.png'))

    run_evaluation = True

    # Compute fidelity for explanation masks
    fidelity_results_from_masks_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-{explanation_type}_mask-fidelity_results.json')    
    if run_evaluation and not os.path.exists(fidelity_results_from_masks_file):
        evaluate_masks(explainer, explanation_task, explanation_type, mask_key_ranking, schema_graph, kmax=kstar_only_f2p, n_samples=n_fidelity_estimation_samples, perturbation_strategy=perturbation_strategy, result_dir=result_dir)
    if os.path.exists(fidelity_results_from_masks_file):
        with open(fidelity_results_from_masks_file, 'r') as f:
            fidelity_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for masks not found: {fidelity_results_from_masks_file}")
    klist = fidelity_results['k_list'][1:]  # Exclude k=0
    # NOTE: For greedy competitor: kstar gives a time advantage (when kstar is small), but max(klist) might give a better fidelity result (when max(klist) is large and mask removes too many joins).
    #                              best would be to implement a stopping criterion based on the fidelity results for greedy expansion..
    #       For random baselines: always use max(klist), since we need to calculate fidelity up to kmax.
    kmax = max(klist)
    # kmax = kstar_only_f2p

    # Compute fidelity for random expansion baseline
    fidelity_results_from_random_expansion_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-random_expansion-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_results_from_random_expansion_file):
        evaluate_random_expansion_baseline(explainer, explanation_task, explanation_type, schema_graph, klist=klist, kmax=kmax, n_samples=n_fidelity_estimation_samples, perturbation_strategy=perturbation_strategy, result_dir=result_dir, suffix=suffix, reps=reps)
    if os.path.exists(fidelity_results_from_random_expansion_file):
        with open(fidelity_results_from_random_expansion_file, 'r') as f:
            fidelity_random_expansion_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for random expansion not found: {fidelity_results_from_random_expansion_file}")

    # Compute fidelity for greedy expansion baseline
    fidelity_results_from_greedy_file = os.path.join(result_dir, f'{dataset_name}-{task_name}-greedy_expansion-fidelity_results.json')
    if run_evaluation and not os.path.exists(fidelity_results_from_greedy_file):
        evaluate_greedy_expansion_baseline(explainer, explanation_task, explanation_type, schema_graph, klist=klist, kmax=kmax, n_samples=n_fidelity_estimation_samples, perturbation_strategy=perturbation_strategy, result_dir=result_dir, greedy_expansion_source=greedy_base_result)
    if os.path.exists(fidelity_results_from_greedy_file):
        with open(fidelity_results_from_greedy_file, 'r') as f:
            fidelity_greedy_results = json.load(f)
    else:
        raise FileNotFoundError(f"Fidelity results file for greedy expansion not found: {fidelity_results_from_greedy_file}")

    # Visualize fidelity curves
    for split in ['test']:
        visualize_fidelity_curves(
            fidelity_results = {'mask': fidelity_results, 
                                'random_walk': fidelity_random_expansion_results,
                                'greedy_expansion': fidelity_greedy_results},
            dataset_name=dataset_name,
            task_name=task_name,
            file_path=os.path.join(result_dir, f'{dataset_name}-{task_name}-fidelity_curves-{split}.png'),
            split=split,
            kstar=kstar_only_f2p,
        )

    # Time comparison
    time_results = {'mask': time_to_learn_masks,
                    'greedy_expansion': fidelity_greedy_results['time']}
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
    parser.add_argument("--exp_type", type=str, default='fkpk', choices=['fkpk'], help="Type of explanations to evaluate")
    parser.add_argument("--target_type", type=str, default='soft', choices=['hard', 'soft'], help="Target type for explanation")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    parser.add_argument("--n_fidelity_estimation_samples", type=int, default=5, help="Number of samples for fidelity estimation")
    parser.add_argument("--suffix", type=str, default='best', help="Suffix for result files")
    parser.add_argument("--reps", type=int, default=5, help="Number of repetitions for random baselines")
    parser.add_argument("--perturbation_strategy", type=str, default='foreign_key_permutation', choices=['foreign_key_permutation', 'foreign_key_exchange', 'foreign_key_uniform_random', 'foreign_key_hist_random'], help="Perturbation strategy for explanation evaluation")
    parser.add_argument("--greedy_base_result", type=str, default=None, help="Path to the greedy expansion source file (if available)")
    args = parser.parse_args()

    print(f"Evaluating join explanations from {args.mask_dir} for task {args.task_dir}.")

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
        reps=args.reps,
        greedy_base_result=args.greedy_base_result,
    )