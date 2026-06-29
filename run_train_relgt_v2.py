"""Train Relational Graph Transformer (RelGT) models in the v2 layout.

Mirrors run_train_models_v2.py's structure (per-task TASKS_RELGT config,
(lr × seed) grid, v2 output layout under relbench_models_v2/) but uses the
in-package RelGT model + the RelGTTokens transformer-style sampler instead
of HeteroGraphSAGE + NeighborLoader.

Adapted from train_relgt.py in the rdl-explain repo. Key differences:
  * Reads the v2 graph_data.pt (data + col_stats_dict bundled) rather than
    the v1 separate data.pt + col_stats_dict.pt files.
  * Runs a (lr × seed) grid with per-task hyperparameters from TASKS_RELGT.
  * Supports MULTICLASS_CLASSIFICATION (e.g. rel-arxiv/author-category) —
    the original train_relgt.py raised on this.
  * Output: relbench_models_v2/{db}/{task}_relgt/{best_model, runs/, ...}.
  * Token cache: relbench_models_v2/{db}/{task}_relgt/relgt_tokens/ — kept
    next to the model artifacts, not in the project root.

Output schema (per (lr, seed) run):
  relbench_models_v2/{db}/{task}_relgt/
    runs/lr_{lr:g}_seed_{seed}/
      model_state_dict.pth
      relgt_params.json
      meta.yaml
      training_curve.{json,png}
      inference_results/
        computed_metrics.json
        predictions_{train,val,test}.parquet
    best_model/   (verbatim copy of best run by val metric)
    tuning_summary.json
    training_summary.png
    relgt_tokens/  (RelGTTokens precomputed cache, shared across runs)

Usage:
  python run_train_relgt_v2.py --task rel-f1/driver-dnf
  python run_train_relgt_v2.py --task rel-trial/study-outcome --lrs 5e-4 --seeds 42
  python run_train_relgt_v2.py                                 # all tasks
"""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.seed import seed_everything
from torch_frame.data.stats import StatType

from relbench.base import EntityTask, TaskType
from relbench.tasks import get_task

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "src")
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdl_explain.model import RelGT, RelGTConfig, RelGTTokens


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT = "/home/rissakiagapi/rdl-explain-data"
OUT_ROOT  = os.path.join(DATA_ROOT, "relbench_models_v2")

# ── Per-task hyperparameters ───────────────────────────────────────────────────
# num_centroids ≈ round_to_power_of_2(n_train / 75) — see docstring of
# train_relgt.py for the rationale.
TASKS_RELGT_SMALL: Dict[str, Dict[str, Any]] = {
    "rel-f1/driver-dnf": {
        # ~11k train rows, ~857 unique drivers → 16 centroids (~54 drivers/centroid)
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.2,
        "attn_dropout":     0.2,
        "num_centroids":    16,
        "batch_size":       64,
    },
    "rel-f1/driver-top3": {
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.2,
        "attn_dropout":     0.2,
        "num_centroids":    16,
        "batch_size":       64,
    },
    "rel-trial/study-outcome": {
        # ~12k train studies → 128 centroids (~94 studies/centroid)
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.1,
        "attn_dropout":     0.1,
        "num_centroids":    128,
        "batch_size":       64,
    },
    "rel-hm/user-churn": {
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.1,
        "attn_dropout":     0.1,
        "num_centroids":    1024,
        "batch_size":       64,
    },
    "rel-arxiv/paper-citation": {
        # ~534k papers → power_of_2(534k/75) ≈ 8192
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.1,
        "attn_dropout":     0.1,
        "num_centroids":    8192,
        "batch_size":       64,
    },
    "rel-arxiv/author-category": {
        # multiclass task — need to know n_authors to set centroids; default
        # 4096 is the relgt-paper default.
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.1,
        "attn_dropout":     0.1,
        "num_centroids":    4096,
        "batch_size":       64,
    },
    "synthetic_cohort/cohort-task": {
        # 5000 R nodes, label depends on 1-hop (S.y or T.z gated by R.x) →
        # 2-hop sampling is more than enough; tiny centroid book.
        "K":                100,
        "channels":         32,
        "global_dim":       32,
        "local_num_layers": 1,
        "heads":            4,
        "ff_dropout":       0.1,
        "attn_dropout":     0.1,
        "num_centroids":    64,
        "batch_size":       64,
    },
}


# Paper-faithful defaults, per the RelGT paper §"We implement RELGT within the
# RDL pipeline…":
#   * lr = 1e-4, 10-20M params (channels=128, global_dim=128 lands in this band)
#   * K = 300 local neighbors,  B = 4096 global centroids
#   * <1M train nodes: tune L ∈ {1,4,8}, dropout ∈ {0.3,0.4,0.5}, batch=256
#   * >1M train nodes: L = 4 fixed, batch=1024
# We pick a single point in the recommended grid (L=4, dropout=0.4) so a single
# run matches the paper without forcing a sweep; users can override via CLI.
TASKS_RELGT_PAPER: Dict[str, Dict[str, Any]] = {
    "rel-f1/driver-dnf": {
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "rel-f1/driver-top3": {
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "rel-trial/study-outcome": {
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "rel-hm/user-churn": {
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "rel-arxiv/paper-citation": {
        # 534k train nodes → still <1M, so batch=256 / tunable L.
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "rel-arxiv/author-category": {
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    4096,
        "batch_size":       256,
    },
    "synthetic_cohort/cohort-task": {
        # Only 5000 R nodes — the paper config's 4096 centroids on 5k entities
        # is degenerate, so we cap at 64. Other params follow paper defaults.
        "K":                300,
        "channels":         128,
        "global_dim":       128,
        "local_num_layers": 4,
        "heads":            4,
        "ff_dropout":       0.4,
        "attn_dropout":     0.4,
        "num_centroids":    64,
        "batch_size":       256,
    },
}

CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "small": TASKS_RELGT_SMALL,
    "paper": TASKS_RELGT_PAPER,
}

# Per-config default learning rate (paper uses 1e-4; small uses 5e-4 to converge
# faster at the smaller model size).
DEFAULT_LRS_BY_CONFIG: Dict[str, List[float]] = {
    "small": [5e-4],
    "paper": [1e-4],
}

DEFAULT_LRS              = [5e-4]  # kept for back-compat; main() picks per-config
DEFAULT_SEEDS            = [42]
DEFAULT_EPOCHS           = 20
PATIENCE                 = 10
MAX_STEPS_PER_EPOCH      = 3000
WEIGHT_DECAY             = 1e-5
NUM_DATALOADER_WORKERS   = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_col_stats(col_stats_dict: dict) -> dict:
    """Columns that are entirely null in train have NaN mean and std, which
    poison the encoder's normalisation. Replace with safe defaults."""
    for col_stats in col_stats_dict.values():
        for stats in col_stats.values():
            if StatType.MEAN in stats and not math.isfinite(stats[StatType.MEAN]):
                stats[StatType.MEAN] = 0.0
            if StatType.STD in stats and not math.isfinite(stats[StatType.STD]):
                stats[StatType.STD] = 1.0
    return col_stats_dict


def load_v2_graph(db_name: str, task_dir_name: str) -> Tuple[HeteroData, dict]:
    """Read the v2 graph_data.pt and return (data, col_stats_dict)."""
    task_dir = os.path.join(OUT_ROOT, db_name, task_dir_name)
    pkg = torch.load(os.path.join(task_dir, "graph_data.pt"), weights_only=False)
    return pkg["data"], _sanitize_col_stats(pkg["col_stats_dict"])


def task_loss_and_metric(task: EntityTask):
    """Returns (out_channels, loss_fn, tune_metric, higher_is_better,
                clamp_min, clamp_max)."""
    clamp_min = clamp_max = None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return 1, BCEWithLogitsLoss(), "roc_auc", True, None, None
    if task.task_type == TaskType.REGRESSION:
        train_df = task.get_table("train").df
        cmin, cmax = np.percentile(train_df[task.target_col].to_numpy(), [2, 98])
        return 1, L1Loss(), "mae", False, float(cmin), float(cmax)
    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return task.num_labels, BCEWithLogitsLoss(), "multilabel_auprc_macro", True, None, None
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return task.num_classes, CrossEntropyLoss(), "multiclass_f1", True, None, None
    raise ValueError(f"Unsupported task type: {task.task_type}")


# ── Training / evaluation ─────────────────────────────────────────────────────

def _batch_to_kwargs(batch: dict, device: torch.device) -> dict:
    """Common kwargs to RelGT.forward extracted from a token-collated batch."""
    return dict(
        neighbor_types  = batch["neighbor_types"].to(device),
        node_indices    = batch["node_indices"].to(device),
        neighbor_hops   = batch["neighbor_hops"].to(device),
        neighbor_times  = batch["neighbor_times"].to(device),
        grouped_tf_dict = {
            "grouped_tfs":     batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx":  batch["flat_batch_idx"],
            "flat_nbr_idx":    batch["flat_nbr_idx"],
        },
        edge_index = batch["edge_index"].to(device),
        batch      = batch["batch"].to(device),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, task, device, max_steps):
    model.train()
    loss_acc = count = 0
    for step, batch in enumerate(loader, 1):
        kwargs = _batch_to_kwargs(batch, device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        pred = model(**kwargs)
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = loss_fn(pred, labels.long())
        else:
            pred_view = pred.view(-1) if pred.size(-1) == 1 else pred
            loss = loss_fn(pred_view.float(), labels.float())
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_acc += loss.item() * (pred.size(0))
        count    += pred.size(0)
        if step >= max_steps:
            break
    return loss_acc / max(count, 1)


@torch.no_grad()
def predict(model, loader, task, clamp_min, clamp_max, device) -> np.ndarray:
    """Return ordered predictions array of shape (N,) for binary/regression,
    (N, C) for multiclass/multilabel."""
    model.eval()
    pred_list, idx_list = [], []
    for batch in loader:
        kwargs = _batch_to_kwargs(batch, device)
        pred   = model(**kwargs)
        if task.task_type == TaskType.REGRESSION and clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in (TaskType.BINARY_CLASSIFICATION,
                              TaskType.MULTILABEL_CLASSIFICATION):
            pred = torch.sigmoid(pred)
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=-1)
        if pred.size(-1) == 1:
            pred = pred.view(-1)
        pred_list.append(pred.cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())

    preds = np.concatenate(pred_list, axis=0)
    idxs  = np.concatenate(idx_list)
    ordered = np.empty_like(preds)
    ordered[idxs] = preds
    return ordered


def save_predictions_parquet(task: EntityTask, split: str, pred: np.ndarray,
                             out_path: str) -> None:
    """Save the original split-table df + prediction column(s) to parquet."""
    import pandas as pd
    df = task.get_table(split).df.copy()
    if pred.ndim == 1:
        df["prediction"] = pred
    else:
        for i in range(pred.shape[1]):
            df[f"prediction_{i}"] = pred[:, i]
    df.to_parquet(out_path, index=False)


def plot_training_curve(curve: Dict[str, list], tune_metric: str, out_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(curve["epoch"], curve["train_loss"], color="tab:red",
             label="train loss", linewidth=1.7)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(curve["epoch"], curve["val_metric"], color="tab:blue",
             label=f"val {tune_metric}", linewidth=1.7, marker="o", markersize=3)
    if curve.get("test_metric"):
        ax2.plot(curve["epoch"], curve["test_metric"], color="tab:green",
                 label=f"test {tune_metric}", linewidth=1.7,
                 marker="s", markersize=3, linestyle="--")
    ax2.set_ylabel(tune_metric, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="lower right", fontsize=8)

    if curve.get("best_epoch") is not None:
        ax1.axvline(curve["best_epoch"], color="black", linestyle=":", alpha=0.5,
                    label=f"best epoch={curve['best_epoch']} (by val)")
        ax1.legend(loc="lower left", fontsize=8)

    fig.suptitle(out_path.split("/runs/")[-1].split("/")[0], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Per-run training ──────────────────────────────────────────────────────────

def train_one_run(
    *,
    data: HeteroData,
    col_stats_dict: dict,
    task: EntityTask,
    cfg: Dict[str, Any],
    relgt_token_cache: str,
    lr: float,
    seed: int,
    epochs: int,
    patience: int,
    max_steps_per_epoch: int,
    out_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single (lr, seed) config; save artifacts; return summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "inference_results"), exist_ok=True)

    seed_everything(seed)
    out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = \
        task_loss_and_metric(task)

    # Build the per-split RelGTTokens datasets (precompute is cached under
    # relgt_token_cache, shared across (lr, seed) combos for this task).
    split_datasets = {
        split: RelGTTokens(
            data=data, task=task,
            K=cfg["K"],
            split=split,
            undirected=True,
            precompute=True,
            precomputed_dir=relgt_token_cache,
            num_workers=NUM_DATALOADER_WORKERS,
        )
        for split in ["train", "val", "test"]
    }
    loaders = {
        split: DataLoader(
            split_datasets[split],
            batch_size=cfg["batch_size"],
            shuffle=(split == "train"),
            collate_fn=split_datasets[split].collate,
            num_workers=NUM_DATALOADER_WORKERS,
            persistent_workers=(NUM_DATALOADER_WORKERS > 0),
            pin_memory=(device.type == "cuda"),
        )
        for split in ["train", "val", "test"]
    }

    # Build RelGT
    train_ds = split_datasets["train"]
    relgt_cfg = RelGTConfig(
        local_num_layers=cfg["local_num_layers"],
        channels=cfg["channels"],
        out_channels=out_channels,
        global_dim=cfg["global_dim"],
        heads=cfg["heads"],
        ff_dropout=cfg["ff_dropout"],
        attn_dropout=cfg["attn_dropout"],
        conv_type="full",
        num_centroids=cfg["num_centroids"],
        sample_node_len=cfg["K"],
    )
    model = RelGT(
        num_nodes=data.num_nodes,
        max_neighbor_hop=train_ds.max_neighbor_hop,
        node_type_map=train_ds.node_type_to_index,
        col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
        col_stats_dict=col_stats_dict,
        **relgt_cfg.model_dump(),
    ).to(device)
    log(f"      RelGT params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    curve = {"epoch": [], "train_loss": [],
             "val_metric": [], "val_all": [],
             "test_metric": [], "test_all": []}
    best_state = None
    best_val   = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    epochs_since_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, loss_fn,
                                     task, device, max_steps_per_epoch)
        val_pred    = predict(model, loaders["val"], task, clamp_min, clamp_max, device)
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_metric  = float(val_metrics[tune_metric])
        # Per-epoch test eval — labels are hidden client-side but task.evaluate has them.
        test_pred    = predict(model, loaders["test"], task, clamp_min, clamp_max, device)
        test_metrics = task.evaluate(test_pred)
        test_metric  = float(test_metrics.get(tune_metric, float('nan')))

        curve["epoch"].append(epoch)
        curve["train_loss"].append(float(train_loss))
        curve["val_metric"].append(val_metric)
        curve["val_all"].append({k: float(v) for k, v in val_metrics.items()})
        curve["test_metric"].append(test_metric)
        curve["test_all"].append({k: float(v) for k, v in test_metrics.items()})

        improved = (higher_is_better and val_metric >= best_val) or \
                   (not higher_is_better and val_metric <= best_val)
        if improved:
            best_val   = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        log(f"      ep {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_{tune_metric}={val_metric:.4f}  "
            f"test_{tune_metric}={test_metric:.4f}"
            f"{'  *' if improved else ''}")

        if epochs_since_improve >= patience:
            log(f"      early stop at epoch {epoch} (patience={patience}, "
                f"best epoch {best_epoch})")
            break

    elapsed = time.time() - t0
    curve["best_epoch"] = best_epoch

    # Final eval with best checkpoint
    assert best_state is not None
    model.load_state_dict(best_state)
    train_pred = predict(model, loaders["train"], task, clamp_min, clamp_max, device)
    val_pred   = predict(model, loaders["val"],   task, clamp_min, clamp_max, device)
    test_pred  = predict(model, loaders["test"],  task, clamp_min, clamp_max, device)

    train_metrics = task.evaluate(train_pred, task.get_table("train"))
    val_metrics   = task.evaluate(val_pred,   task.get_table("val"))
    test_metrics  = task.evaluate(test_pred)

    # Save artifacts
    torch.save(best_state, os.path.join(out_dir, "model_state_dict.pth"))

    relgt_params = {
        **relgt_cfg.model_dump(),
        "K":            cfg["K"],
        "batch_size":   cfg["batch_size"],
        "seed":         seed,
        "learning_rate": lr,
        "weight_decay":  WEIGHT_DECAY,
        "max_steps_per_epoch": max_steps_per_epoch,
    }
    with open(os.path.join(out_dir, "relgt_params.json"), "w") as f:
        json.dump(relgt_params, f, indent=2)

    _best_test_metric = float('nan')
    if curve["test_metric"] and 1 <= best_epoch <= len(curve["test_metric"]):
        _best_test_metric = float(curve["test_metric"][best_epoch - 1])

    meta = {
        "task":             None,
        "lr":               lr,
        "seed":             seed,
        "epochs_run":       len(curve["epoch"]),
        "best_epoch":       best_epoch,
        "best_val_metric":  best_val,
        "best_test_metric": _best_test_metric,
        "tune_metric":      tune_metric,
        "higher_is_better": higher_is_better,
        "elapsed_seconds":  round(elapsed, 1),
        "model":            "RelGT",
        **{k: v for k, v in cfg.items()},
        "patience":         patience,
    }
    with open(os.path.join(out_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    with open(os.path.join(out_dir, "training_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    plot_training_curve(curve, tune_metric, os.path.join(out_dir, "training_curve.png"))

    with open(os.path.join(out_dir, "inference_results", "computed_metrics.json"), "w") as f:
        json.dump({
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val":   {k: float(v) for k, v in val_metrics.items()},
            "test":  {k: float(v) for k, v in test_metrics.items()},
        }, f, indent=2)

    for split, pred in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
        save_predictions_parquet(
            task, split, pred,
            os.path.join(out_dir, "inference_results", f"predictions_{split}.parquet"),
        )

    # Free GPU + worker memory before next run
    del model, optimizer, best_state, loaders, split_datasets
    torch.cuda.empty_cache()

    return {
        "lr":               lr,
        "seed":             seed,
        "best_epoch":       best_epoch,
        "best_val_metric":  best_val,
        "best_test_metric": _best_test_metric,
        "tune_metric":      tune_metric,
        "elapsed_seconds":  round(elapsed, 1),
        "out_dir":          out_dir,
    }


# ── Per-task driver ───────────────────────────────────────────────────────────

def run_task(
    db_task: str,
    *,
    tasks_cfg: Dict[str, Dict[str, Any]],
    config_name: str,
    lrs: List[float],
    seeds: List[int],
    epochs: int,
    patience: int,
    max_steps_per_epoch: int,
    device: torch.device,
    suffix_override: str = None,
) -> None:
    db_name, task_name = db_task.split("/")
    cfg = tasks_cfg[db_task]
    # Use a config-specific suffix so paper vs small results don't clobber each
    # other: paper-config artifacts land in {task}_relgt_paper/.
    if suffix_override is not None:
        suffix = suffix_override
    else:
        suffix = "_relgt" if config_name == "small" else f"_relgt_{config_name}"
    out_dir_task = os.path.join(OUT_ROOT, db_name, f"{task_name}{suffix}")
    os.makedirs(out_dir_task, exist_ok=True)

    log(f"=== {db_task}  (K={cfg['K']}, channels={cfg['channels']}, "
        f"num_centroids={cfg['num_centroids']}, heads={cfg['heads']}) ===")

    # Load the v2 graph (sourced from {task_name} dir, not the _relgt suffix).
    # Convention: training data is shared across model variants — graph_data.pt
    # lives in the original {task_name} v2 dir (where run_train_models_v2 wrote
    # it). The _relgt suffix only differentiates the model+results.
    base_task_dir = task_name
    log(f"  loading graph from {OUT_ROOT}/{db_name}/{base_task_dir}/graph_data.pt …")
    data, col_stats_dict = load_v2_graph(db_name, base_task_dir)
    log(f"  graph: {data}")

    log(f"  loading task ({db_task}) [download=False]…")
    if db_name == "synthetic_cohort":
        # Synthetic task — no relbench registration; use the local adapter
        # built around ground_truth.pt that run_synthetic_cohort.py writes.
        from rdl_explain.synthetic_task import load_task as load_synth_task
        task = load_synth_task(os.path.join(OUT_ROOT, db_name, base_task_dir))
    else:
        task = get_task(db_name, task_name, download=False)

    # Token cache lives under the model output dir — shared across (lr × seed).
    relgt_token_cache = os.path.join(out_dir_task, "relgt_tokens")
    os.makedirs(relgt_token_cache, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for lr in lrs:
        for seed in seeds:
            run_name = f"lr_{lr:g}_seed_{seed}"
            out_dir  = os.path.join(out_dir_task, "runs", run_name)
            if os.path.exists(os.path.join(out_dir, "meta.yaml")):
                log(f"  [{run_name}] already exists; loading summary")
                with open(os.path.join(out_dir, "meta.yaml")) as f:
                    meta = yaml.safe_load(f)
                summaries.append({
                    "lr":               lr,
                    "seed":             seed,
                    "best_epoch":       meta["best_epoch"],
                    "best_val_metric":  meta["best_val_metric"],
                    "best_test_metric": meta.get("best_test_metric", float('nan')),
                    "tune_metric":      meta["tune_metric"],
                    "elapsed_seconds":  meta["elapsed_seconds"],
                    "out_dir":          out_dir,
                })
                continue

            log(f"  [{run_name}]")
            summary = train_one_run(
                data=data,
                col_stats_dict=col_stats_dict,
                task=task,
                cfg=cfg,
                relgt_token_cache=relgt_token_cache,
                lr=lr,
                seed=seed,
                epochs=epochs,
                patience=patience,
                max_steps_per_epoch=max_steps_per_epoch,
                out_dir=out_dir,
                device=device,
            )
            summaries.append(summary)

    # Pick best by val metric
    _, _, _, higher_is_better, *_ = task_loss_and_metric(task)
    summaries_sorted = sorted(
        summaries, key=lambda s: s["best_val_metric"],
        reverse=higher_is_better,
    )
    best = summaries_sorted[0]

    log(f"  best: lr={best['lr']}, seed={best['seed']}, "
        f"val_{best['tune_metric']}={best['best_val_metric']:.4f}  "
        f"test_{best['tune_metric']}={best.get('best_test_metric', float('nan')):.4f}")

    best_dst = os.path.join(out_dir_task, "best_model")
    if os.path.exists(best_dst):
        shutil.rmtree(best_dst)
    shutil.copytree(best["out_dir"], best_dst)

    with open(os.path.join(out_dir_task, "tuning_summary.json"), "w") as f:
        json.dump({
            "db_task":          db_task,
            "model":            "RelGT",
            **{k: v for k, v in cfg.items()},
            "epochs":           epochs,
            "patience":         patience,
            "tune_metric":      best["tune_metric"],
            "higher_is_better": higher_is_better,
            "best": {
                "lr":               best["lr"],
                "seed":             best["seed"],
                "best_val_metric":  best["best_val_metric"],
                "best_test_metric": best.get("best_test_metric", float('nan')),
                "best_epoch":       best["best_epoch"],
                "run_dir":          os.path.relpath(best["out_dir"], out_dir_task),
            },
            "all_runs":         summaries_sorted,
        }, f, indent=2)

    # Mini per-task grid plot
    n     = len(summaries)
    xs    = [f"lr={s['lr']:g}\nseed={s['seed']}" for s in summaries]
    ys_v  = [s["best_val_metric"]                     for s in summaries]
    ys_t  = [s.get("best_test_metric", float('nan'))  for s in summaries]
    x_pos = np.arange(n)
    bw    = 0.4
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * n + 1.5), 4))
    bars_v = ax.bar(x_pos - bw/2, ys_v, bw, label="val",  edgecolor="black", color="tab:blue")
    bars_t = ax.bar(x_pos + bw/2, ys_t, bw, label="test", edgecolor="black", color="tab:green")
    best_idx = next((i for i, s in enumerate(summaries)
                     if s["out_dir"] == best["out_dir"]), None)
    if best_idx is not None:
        for bar in (bars_v[best_idx], bars_t[best_idx]):
            bar.set_edgecolor("red"); bar.set_linewidth(2.0)
    ax.set_xticks(x_pos); ax.set_xticklabels(xs, fontsize=8)
    ax.set_ylabel(best["tune_metric"])
    ax.set_title(f"{db_task} (RelGT) — tuning runs (red border = best by val)")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir_task, "training_summary.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)

    log(f"  artifacts saved → {out_dir_task}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="small",
                        help=("Which hyperparameter preset to use. 'small' = fast/cheap "
                              "defaults matching run_train_models_v2.py scale; 'paper' = "
                              "the RelGT paper's recommended settings (~10-20M params, "
                              "K=300, B=4096, batch=256, lr=1e-4, dropout=0.4, L=4)."))
    parser.add_argument("--task", nargs="*", default=None,
                        help="One or more db/task strings. Defaults to all tasks in the selected config.")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help=("Override the output dir suffix. Default: '_relgt' for "
                              "--config small, '_relgt_paper' for --config paper. Use this "
                              "to keep multiple variants of the same config from clobbering "
                              "each other, e.g. --out-suffix _relgt_paper_L8."))
    parser.add_argument("--lrs", nargs="+", type=float, default=None,
                        help="Override LRs. Default depends on --config (paper=1e-4, small=5e-4).")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max-steps-per-epoch", type=int, default=MAX_STEPS_PER_EPOCH,
                        help="Cap on optimizer steps per training epoch.")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    tasks_cfg = CONFIGS[args.config]
    lrs       = args.lrs if args.lrs is not None else DEFAULT_LRS_BY_CONFIG[args.config]

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    tasks = args.task or list(tasks_cfg.keys())
    for t in tasks:
        if t not in tasks_cfg:
            log(f"WARNING: {t} not in TASKS_RELGT[{args.config}]; skipping")
            continue

    if args.out_suffix is not None:
        suffix = args.out_suffix
    else:
        suffix = "_relgt" if args.config == "small" else f"_relgt_{args.config}"
    log(f"Device={device}  config={args.config}  Tasks={tasks}  lrs={lrs}  "
        f"seeds={args.seeds}  max_steps_per_epoch={args.max_steps_per_epoch}")
    log(f"OUT_ROOT={OUT_ROOT}  suffix={suffix}")

    for t in tasks:
        if t not in tasks_cfg:
            continue
        try:
            run_task(
                t,
                tasks_cfg=tasks_cfg,
                config_name=args.config,
                lrs=lrs,
                seeds=args.seeds,
                epochs=args.epochs,
                patience=args.patience,
                max_steps_per_epoch=args.max_steps_per_epoch,
                device=device,
                suffix_override=args.out_suffix,
            )
        except Exception as e:
            log(f"ERROR on task {t}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    log("done.")


if __name__ == "__main__":
    main()
