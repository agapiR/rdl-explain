"""Evaluation utilities for RDL explanations.

Standalone functions for evaluating explanation quality.
These functions are intentionally decoupled from RDLExplainer so they can be
called with any mask without constructing a full explainer instance.
"""

import gc
import os
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from relbench.base import EntityTask, TaskType

from rdl_explain.explain.explain_utils import perturb_instance


def estimate_deviation_from_determinacy(
    model: torch.nn.Module,
    data: HeteroData,
    task: EntityTask,
    mask: Dict,
    loader_factory: Callable[[], NeighborLoader],
    explanation_type: str = 'column',
    perturbation_strategy: str = 'permutation_independent',
    num_samples: int = 100,
    prediction_type: str = 'soft',
    device: str = 'cpu',
    random_seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate deviation from determinacy (devΔ) for an explanation mask.

    devΔ(E, s) = E_{D' ~ Δ} [ dist( M(D, s), M(D', s) ) | ∀ V ∈ E: V(D) = V(D') ]
    devΔ(E)    = (1/N) Σ_s devΔ(E, s)

    Uses **paired inference**: for each perturbation draw s, the same torch seed
    is set before the original forward pass and before the perturbed forward pass,
    so both passes sample identical subgraphs from NeighborLoader. This eliminates
    the upward bias that arises when the two passes use different sampled subgraphs.

    Ordering within each draw:
        1. torch.manual_seed(seed_s)  →  original inference on D
        2. perturb_instance(...)      →  D modified to D' (uses residual RNG state)
        3. torch.manual_seed(seed_s)  →  perturbed inference on D'  (same subgraph as 1)
        4. restore D from backup

    Note: perturb_instance uses torch.randperm internally, so the seed set in step 1
    is consumed by the loader before perturb_instance runs in step 2. The perturbation
    therefore sees a different RNG state per draw, which is the desired behaviour.

    Supported task types: BINARY_CLASSIFICATION, REGRESSION.

    Distance metrics:
        BINARY_CLASSIFICATION: |ŷ_i - ŷ'_{s,i}|           ∈ [0, 1]
        REGRESSION:            SMAPE/2                      ∈ [0, 1]
                               |ŷ - ŷ'| / ((|ŷ| + |ŷ'|)/2 + ε) / 2

    NOTE: data is modified in-place during perturbation and fully restored
    before the function returns. The caller's reference remains valid.

    Args:
        model:                The trained model.
        data:                 HeteroData graph (modified in-place then restored).
        task:                 EntityTask with entity_table and task_type.
        mask:                 Hard (boolean) explanation mask.
        loader_factory:       Zero-argument callable returning a fresh NeighborLoader
                              over the evaluation split. Called twice per draw (once
                              for original, once for perturbed inference).
                              Typical usage:
                                loader_factory = lambda: explainer.create_loader(
                                    data, 'train', node_id_filter=...
                                )
        explanation_type:     Mask type passed to perturb_instance ('column', 'row', 'fkpk').
        perturbation_strategy: Perturbation type passed to perturb_instance.
        num_samples:          Number of perturbation draws S. Total forward passes = 2S.
        prediction_type:      'soft' (probabilities / continuous) or 'hard' (thresholded).
                              'ground_truth' raises ValueError.
                              For REGRESSION, both options return the same output.
        device:               Device string for model inference.

    Returns:
        dev_delta:                  Scalar mean devΔ.
        dev_std:                    Std error of the mean across perturbation draws
                                    = std_s(m_s) / sqrt(S).
        per_instance_dev:           (N,) per-instance mean deviation
                                    dev_i = (1/S) Σ_s d_{s,i}.
        per_instance_dev_std:       (N,) per-instance std across perturbation draws
                                    = std_s(d_{s,i}).
        per_sample_mean_distances:  (S,) per-perturbation mean distance
                                    m_s = (1/N) Σ_i d_{s,i}.
        original_predictions:       (S, N) the model's predictions on the ORIGINAL
                                    (unperturbed) database for each draw s and
                                    instance i, i.e. ŷ_{s,i}. Raw per-draw (draws
                                    differ only by subgraph sampling).
        perturbed_predictions:      (S, N) the model's predictions on the PERTURBED
                                    database for each draw s and instance i, i.e.
                                    ŷ'_{s,i}. Raw per-draw, same shape as
                                    original_predictions.
    """
    if prediction_type == 'ground_truth':
        raise ValueError(
            "prediction_type='ground_truth' is not valid for devΔ. "
            "Use 'soft' or 'hard'."
        )
    if prediction_type not in ('soft', 'hard'):
        raise ValueError(
            f"prediction_type must be 'soft' or 'hard', got '{prediction_type}'."
        )
    if task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        raise NotImplementedError(
            f"Task type {task.task_type} is not supported by "
            "estimate_deviation_from_determinacy."
        )

    entity_table = task.entity_table

    def _process(out: Tensor) -> np.ndarray:
        """Convert raw model output to predictions as numpy array."""
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            probs = torch.sigmoid(out)
            if prediction_type == 'hard':
                return (probs >= 0.5).float().cpu().numpy()
            return probs.cpu().numpy()
        else:  # REGRESSION — soft and hard are identical
            return out.cpu().numpy()

    def _distance(orig: np.ndarray, pert: np.ndarray) -> np.ndarray:
        """Per-instance distance between original and perturbed predictions."""
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            return np.abs(orig - pert)                                   # ∈ [0, 1]
        else:  # REGRESSION — SMAPE/2
            eps = 1e-8
            return np.abs(orig - pert) / ((np.abs(orig) + np.abs(pert)) / 2 + eps) / 2

    @torch.no_grad()
    def _run_inference(loader) -> np.ndarray:
        preds = []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, entity_table)
            out = out.view(-1) if out.size(1) == 1 else out
            preds.append(_process(out))
        return np.concatenate(preds)

    model.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # RANDOMNESS GUARANTEES — see /dev_variance.md
    #
    # WITHIN this call (always):
    #   For each draw s, torch.manual_seed(seeds[s]) is called BEFORE both the
    #   original (paired-inference line ~162) and the perturbed (line ~180)
    #   inferences. NeighborLoader sampling is controlled by torch.manual_seed
    #   (verified empirically — see deviation_from_determinacy.md, section
    #   "Empirical finding"). Therefore both forward passes for sample s use
    #   the SAME subgraph, and the difference between them reflects the
    #   perturbation only — subgraph-sampling noise cancels within-sample.
    #
    # ACROSS calls — depends on `random_seed`:
    #
    #   random_seed = None  (default, backward-compatible):
    #       The numpy generator below draws fresh entropy from the OS each
    #       call, so two calls to this function (even with the same mask and
    #       same loader composition) use DIFFERENT torch seeds → different
    #       subgraphs per instance per call. The returned `dev_std` is the
    #       within-call cross-sample SEM only — it does NOT capture this
    #       cross-call variability.
    #
    #   random_seed = <int>  (recommended for strategy comparisons):
    #       The numpy generator is seeded with `random_seed`, so every call
    #       with the same `random_seed` produces the same sequence of torch
    #       seeds → the SAME subgraph samples per draw → strategy
    #       comparisons within a fixed group become apples-to-apples (the
    #       only thing varying across calls is the mask, since perturb_
    #       instance also reads from the seeded torch RNG state).
    #       Callers that compare multiple masks on the same group (cohort
    #       script, peeling script) pass a per-group integer here.
    # ─────────────────────────────────────────────────────────────────────────

    # Pre-generate one seed per draw using numpy so torch.manual_seed calls
    # below do not affect seed generation itself.
    rng = np.random.default_rng(random_seed)   # OS entropy if random_seed is None
    seeds = rng.integers(0, 2**31, size=num_samples)

    # --- Step 1: save data to disk for restoration ---
    cache_id = np.random.randint(0, 1_000_000)
    backup_path = f'graph_data_backup_{cache_id}.pt'
    torch.save(data, backup_path)

    # --- Step 2: paired perturbation loop ---
    # Discover N from the first original pass; allocate arrays after.
    original_predictions  = None   # (S, N) — original predictions per draw
    perturbed_predictions = None   # (S, N) — perturbed predictions per draw
    distance              = None   # (S, N) — per-draw per-instance distances

    for s in range(num_samples):
        # Original inference — fix seed so NeighborLoader uses a known subgraph.
        torch.manual_seed(int(seeds[s]))
        orig_s = _run_inference(loader_factory())

        if original_predictions is None:
            N = len(orig_s)
            original_predictions  = np.empty((num_samples, N), dtype=np.float32)
            perturbed_predictions = np.empty((num_samples, N), dtype=np.float32)
            distance              = np.empty((num_samples, N), dtype=np.float32)
        original_predictions[s] = orig_s

        # Perturbation — consumes whatever torch RNG state remains after inference.
        perturb_instance(
            data, mask,
            mask_type=explanation_type,
            perturbation_type=perturbation_strategy,
        )

        # Perturbed inference — reset to the same seed so the loader samples the
        # same subgraph it used for the original inference above. These are the
        # model's predictions on the PERTURBED database for draw s.
        torch.manual_seed(int(seeds[s]))
        pert_s = _run_inference(loader_factory())
        perturbed_predictions[s] = pert_s
        distance[s] = _distance(orig_s, pert_s)

        # Restore data from backup.
        clean_data = torch.load(backup_path, weights_only=False)
        for node_type in data.node_types:
            data[node_type].tf = clean_data[node_type].tf
        for edge_type in data.edge_types:
            data[edge_type].edge_index = clean_data[edge_type].edge_index
        del clean_data
        gc.collect()

    # Delete backup file.
    try:
        os.remove(backup_path)
    except Exception as e:
        print(f"Warning: could not delete backup file {backup_path}: {e}")

    # --- Step 3: compute statistics ---
    per_instance_dev           = distance.mean(axis=0)          # (N,)
    per_instance_dev_std       = distance.std(axis=0)           # (N,)
    per_sample_mean_distances  = distance.mean(axis=1)          # (S,)
    dev_delta                  = float(per_sample_mean_distances.mean())
    dev_std                    = float(per_sample_mean_distances.std() / math.sqrt(num_samples))

    return (
        dev_delta,
        dev_std,
        per_instance_dev,
        per_instance_dev_std,
        per_sample_mean_distances,
        original_predictions,       # (S, N) per-draw original predictions
        perturbed_predictions,      # (S, N) per-draw perturbed predictions
    )
