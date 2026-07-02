"""High-level orchestration helpers built on top of the explainer + eval.

These are convenience routines that drive `RDLExplainer` and the standalone
`estimate_deviation_from_determinacy` evaluator: learning a mask, evaluating its
deviation-from-determinacy and cost, and subsampling prediction instances.

Dependency layer: this module sits ON TOP of `explainer.py` and `eval.py` (it
imports both). Leaf utilities (perturbation/masking mechanics, task building)
live in `explain_utils.py`; the devΔ metric lives in `eval.py`.

Ported from the cohort-discovery workflow (the cohort-specific machinery —
artifact loading, predicate evaluation, experiment IO — is intentionally not
included here). GraphSAGE only for now.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from rdl_explain.explain.explainer import RDLExplainer
from rdl_explain.explain.eval import estimate_deviation_from_determinacy

# Soft-mask → hard-mask threshold (sigmoid value at/above which a column is kept).
DELTA = 0.1


# ── Mask-learning eps lookup ─────────────────────────────────────────────────
# Loss = task_loss + EPS * reg_loss. Larger EPS → stronger L1 pull → fewer
# columns survive. Auto-eps (`eps='auto'`) sometimes picks values that either
# kill all columns or fail to prune any; these are known-good values per task.
#
# NOTE: these values are placeholders tuned against a particular set of trained
# models / subsample sizes. Re-tune them when the underlying model changes.
EPS_HYPERPARAMS: Dict[str, float] = {
    "rel-trial/study-outcome":                 0.01,
    "rel-arxiv/paper-citation":                0.001,
    "rel-hm/user-churn":                       0.001,
    "rel-event/user-repeat":                   0.01,
    "synthetic_cohort/cohort-task":            0.01,
    "synthetic_cohort/cohort-task-symmetric":  0.01,
}


def lookup_eps(db_task: str, default="auto"):
    """Return a known-good eps for mask learning, or `default` if untuned.

    Pass `default='auto'` to let the explainer's auto-eps kick in when there is
    no entry for `db_task`.
    """
    return EPS_HYPERPARAMS.get(db_task, default)


# ── Prediction-instance subsampling ──────────────────────────────────────────

def random_subsample_ids(
    ids: Tensor,
    n: Optional[int],
    seed: int = 42,
    return_indices: bool = False,
):
    """Random subsample without replacement.

    Use this to cap a group's training-row count before passing it as
    `node_id_filter` to `learn_mask`. Returns `ids` unchanged if `n is None` or
    `len(ids) <= n`.

    If `return_indices=True`, returns `(ids_sub, picked_positions)` so the
    caller can look up any parallel array (labels, df rows, etc.) at the
    exact positions that were sampled. Use this when validating balance on
    tasks where the same entity_id appears in multiple rows — the dict-based
    `id → label` lookup would lose information across duplicates.
    """
    if n is None or len(ids) <= n:
        if return_indices:
            return ids, np.arange(len(ids))
        return ids
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g)[:n]
    take = perm.cpu().numpy()
    if return_indices:
        return ids[perm], take
    return ids[perm]


def balanced_subsample(
    ids: Tensor,
    labels: np.ndarray,
    n: Optional[int],
    seed: int = 42,
    return_indices: bool = False,
):
    """Stratified subsample by integer label class.

    Splits `n` evenly across the unique values of `labels`. If a class has
    fewer than `n // n_classes` rows, takes all of them (the result may be
    smaller than `n`). Returns `ids` unchanged if `n is None` or
    `len(ids) <= n`.

    Use when the population is class-imbalanced (e.g. driver-dnf with ~30%
    DNF) and you want mask learning to see both classes in equal proportion.

    If `return_indices=True`, returns `(ids_sub, picked_positions)` so the
    caller can look up any parallel array (e.g. the labels themselves) at
    the exact positions that were sampled. Required for correct validation
    on tasks where entity_id repeats across rows; a dict-based `id → label`
    lookup would silently lose the per-row label and report a misleading
    balance.
    """
    if n is None or len(ids) <= n:
        if return_indices:
            return ids, np.arange(len(ids))
        return ids
    labels = np.asarray(labels)
    if len(labels) != len(ids):
        raise ValueError(
            f"labels has len {len(labels)} but ids has len {len(ids)}; "
            f"they must be parallel arrays."
        )
    g = torch.Generator()
    g.manual_seed(seed)
    classes = np.unique(labels)
    n_per   = max(1, n // len(classes))
    picks: list = []
    for c in classes:
        cls_pos = np.where(labels == c)[0]
        if len(cls_pos) <= n_per:
            picks.append(torch.from_numpy(cls_pos))
        else:
            perm = torch.randperm(len(cls_pos), generator=g)[:n_per].numpy()
            picks.append(torch.from_numpy(cls_pos[perm]))
    take = torch.cat(picks).cpu().numpy()
    if return_indices:
        return ids[take], take
    return ids[take]


# ── Mask learning ────────────────────────────────────────────────────────────

def learn_mask(
    explainer: RDLExplainer,
    *,
    node_id_filter: Optional[Tensor] = None,
    n_epochs: int = 200,
    lr: float = 0.01,
    eps: float | str = "auto",
    elimination_strategy: str = "zero",
    mask_init_mu: float = 10,
    pinned_columns: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray, dict]:
    """Run `learn_masks` on (optionally a subset of) the train split, then
    flatten the returned dict into parallel (col_keys, soft, hard) arrays.

    Returns: (col_keys, soft, hard, metrics)
      col_keys : list[(table, col)] in dict-iter order
      soft     : (n_cols,) float32 — sigmoid of the learned logits
      hard     : (n_cols,) bool    — soft >= DELTA
      metrics  : dict with 'loss', 'task_loss', 'reg_loss', 'time' etc.

    `pinned_columns`: list of `(node_type, col_name)` tuples to FREEZE at
    mask≈1 (always-keep) during optimization. The pinned columns still appear
    in the returned `col_keys` with `soft ≈ 1`, so downstream comparisons are
    well-defined.
    """
    mask_logits, _, metrics = explainer.learn_masks(
        eps=eps,
        explanation_type="column",
        elimination_strategy=elimination_strategy,
        n_epochs=n_epochs,
        lr=lr,
        node_id_filter=node_id_filter,
        mask_init_mu=mask_init_mu,
        pinned_columns=pinned_columns,
    )
    col_keys = list(mask_logits.keys())
    soft = np.array(
        [torch.sigmoid(mask_logits[k]).item() for k in col_keys],
        dtype=np.float32,
    )
    hard = (soft >= DELTA)
    return col_keys, soft, hard, metrics


# ── devΔ + cost ──────────────────────────────────────────────────────────────

def _hard_to_mask_dict(col_keys: List[Tuple[str, str]],
                       hard: np.ndarray) -> Dict:
    """Convert a flat boolean array + col_keys into the dict format that
    `perturb_instance` + `estimate_deviation_from_determinacy` expect.
    """
    return {k: torch.tensor([bool(v)], dtype=torch.bool)
            for k, v in zip(col_keys, hard)}


def compute_dev_and_cost(
    explainer: RDLExplainer,
    col_keys: List[Tuple[str, str]],
    hard: np.ndarray,
    *,
    node_id_filter: Tensor,
    num_samples: int = 100,
    perturbation_strategy: str = "permutation_joint",
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute devΔ (deviation from determinacy) for a hard mask, evaluated
    on the prediction-entity subset given by `node_id_filter`, plus the
    mask's *cost* (number of active columns).

    GraphSAGE-only for now.

    Returns:
        {
          'dev':                  scalar mean devΔ
          'dev_sem':              std error of the mean across S draws
          'per_instance_dev':     (|node_id_filter|,) per-prediction-row devΔ
          'cost':                 int — number of active columns (sum of hard)
          'n_instances':          int — len(node_id_filter)
          'num_samples':          int — S
        }
    """
    node_ids_t = node_id_filter if isinstance(node_id_filter, torch.Tensor) \
                                  else torch.as_tensor(node_id_filter)
    loader_factory = lambda: explainer.create_loader(
        explainer.data, "train", shuffle=False, node_id_filter=node_ids_t,
    )
    mask_dict = _hard_to_mask_dict(col_keys, hard)

    dev_delta, dev_sem, per_inst_dev, _, _, _, _ = estimate_deviation_from_determinacy(
        model=explainer.model_to_explain,
        data=explainer.data,
        task=explainer.explanation_task,
        mask=mask_dict,
        loader_factory=loader_factory,
        explanation_type="column",
        perturbation_strategy=perturbation_strategy,
        num_samples=num_samples,
        prediction_type="soft",
        device=str(explainer.device),
        random_seed=random_seed,
    )
    return {
        "dev":              float(dev_delta),
        "dev_sem":          float(dev_sem),
        "per_instance_dev": np.asarray(per_inst_dev),
        "cost":             int(np.asarray(hard).sum()),
        "n_instances":      int(len(node_ids_t)),
        "num_samples":      int(num_samples),
    }
