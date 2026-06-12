"""
Single-process RelGT training script.

Usage examples:
    # driver-dnf
    python train_relgt.py --dataset rel-f1 --task driver-dnf --epochs 20 --num_centroids 16 --ff_dropout 0.2 --attn_dropout 0.2
    # study-outcome
    python train_relgt.py --dataset rel-trial --task study-outcome --num_centroids 128 --epochs 40


Selecting --num_centroids
-------------------------
Target ~50-100 training entities per centroid.  Too few entities per centroid
means the VQ codebook cold-starts slowly and the log(centroid_count) prior is
noisy, causing erratic validation curves in early epochs.  Too many centroids
relative to training entities wastes codebook capacity and slows EMA convergence.

Rule of thumb:
    num_centroids = round_to_power_of_2(num_training_entities / 75)

Examples:
  rel-f1 / driver-dnf
      857 unique drivers, 11 411 training rows
      857 / 75 ≈ 11  →  num_centroids = 16   (~54 drivers/centroid)

  rel-trial / study-outcome
      11,994 training studies
      11,994 / 75 ≈ 160  →  num_centroids = 128   (~94 studies/centroid)
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.seed import seed_everything
from torch_frame.data.stats import StatType
from tqdm import tqdm

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rdl_explain.model.relgt import RelGT, RelGTConfig, RelGTTokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="rel-f1")
    p.add_argument("--task",     default="driver-top3")
    p.add_argument("--data_dir", default="/home/rissakiagapi/rdl-explain-data/relbench_models")
    p.add_argument("--cache_dir",default="./relgt-cache")
    p.add_argument("--out_dir",  default="./relgt-results")
    # model
    p.add_argument("--K",               type=int,   default=100)
    p.add_argument("--channels",        type=int,   default=32)
    p.add_argument("--global_dim",      type=int,   default=32)
    p.add_argument("--local_num_layers",type=int,   default=1)
    p.add_argument("--heads",           type=int,   default=4)
    p.add_argument("--ff_dropout",      type=float, default=0.1)
    p.add_argument("--attn_dropout",    type=float, default=0.1)
    p.add_argument("--num_centroids",   type=int,   default=128)
    # training
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--lr",                  type=float, default=5e-4)
    p.add_argument("--weight_decay",        type=float, default=1e-5)
    p.add_argument("--max_steps_per_epoch", type=int,   default=3000)
    p.add_argument("--num_workers",         type=int,   default=8)
    p.add_argument("--seed",                type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _sanitize_col_stats(col_stats_dict: dict) -> dict:
    """Replace NaN/inf in numerical column statistics with safe defaults.

    Columns that are entirely null in the training data have NaN mean and std.
    The TorchFrame numerical encoder computes (feat - mean) / std, so NaN stats
    contaminate every row and produce NaN gradients even when feat is finite.
    """
    for col_stats in col_stats_dict.values():
        for stats in col_stats.values():
            if StatType.MEAN in stats and not math.isfinite(stats[StatType.MEAN]):
                stats[StatType.MEAN] = 0.0
            if StatType.STD in stats and not math.isfinite(stats[StatType.STD]):
                stats[StatType.STD] = 1.0
    return col_stats_dict


def load_and_normalize_data(data_dir: str, dataset_name: str):
    """Load data.pt and col_stats_dict.pt, lowercasing all node/edge type names."""
    base = os.path.join(data_dir, dataset_name)
    data = torch.load(os.path.join(base, "data.pt"), weights_only=False)
    col_stats_dict = torch.load(os.path.join(base, "col_stats_dict.pt"), weights_only=False)

    # Lowercase normalization (data.pt uses capitalized names, relbench uses lowercase)
    new_data = HeteroData()
    for node_type in data.node_types:
        for key, val in data[node_type].items():
            new_data[node_type.lower()][key] = val
    for src, rel, dst in data.edge_types:
        for key, val in data[src, rel, dst].items():
            new_data[src.lower(), rel, dst.lower()][key] = val

    col_stats_dict = {nt.lower(): stats for nt, stats in col_stats_dict.items()}
    _sanitize_col_stats(col_stats_dict)
    return new_data, col_stats_dict


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device, max_steps):
    model.train()
    loss_accum = count_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False), 1):
        neighbor_types  = batch["neighbor_types"].to(device)
        node_indices    = batch["node_indices"].to(device)
        neighbor_hops   = batch["neighbor_hops"].to(device)
        neighbor_times  = batch["neighbor_times"].to(device)
        edge_index      = batch["edge_index"].to(device)
        batch_vec       = batch["batch"].to(device)
        labels          = batch["labels"].to(device)
        grouped_tf_dict = {
            "grouped_tfs":     batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx":  batch["flat_batch_idx"],
            "flat_nbr_idx":    batch["flat_nbr_idx"],
        }

        optimizer.zero_grad()
        pred = model(neighbor_types, node_indices, neighbor_hops, neighbor_times,
                     grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        loss = loss_fn(pred.float(), labels.float())
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_accum  += loss.item() * pred.size(0)
        count_accum += pred.size(0)
        if step >= max_steps:
            break

    return loss_accum / count_accum


@torch.no_grad()
def evaluate(model, loader, task, device, clamp_min=None, clamp_max=None):
    """Run inference and return ordered predictions array."""
    model.eval()
    pred_list, idx_list = [], []
    for batch in tqdm(loader, desc="  eval ", leave=False):
        neighbor_types  = batch["neighbor_types"].to(device)
        node_indices    = batch["node_indices"].to(device)
        neighbor_hops   = batch["neighbor_hops"].to(device)
        neighbor_times  = batch["neighbor_times"].to(device)
        edge_index      = batch["edge_index"].to(device)
        batch_vec       = batch["batch"].to(device)
        grouped_tf_dict = {
            "grouped_tfs":     batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx":  batch["flat_batch_idx"],
            "flat_nbr_idx":    batch["flat_nbr_idx"],
        }
        pred = model(neighbor_types, node_indices, neighbor_hops, neighbor_times,
                     grouped_tf_dict, edge_index=edge_index, batch=batch_vec)
        if clamp_min is not None:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1) if pred.size(-1) == 1 else pred
        pred_list.append(pred.cpu().numpy())
        idx_list.append(batch["global_idx"].cpu().numpy())

    preds = np.concatenate(pred_list)
    idxs  = np.concatenate(idx_list)
    ordered = np.empty_like(preds)
    ordered[idxs] = preds
    return ordered


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(history: dict, out_path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    tune_metric = history["tune_metric"]
    ax.plot(history["epoch"], history["val_metric"], label=f"val {tune_metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(tune_metric)
    ax.set_title(f"Validation {tune_metric}")
    ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Learning curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  {args.dataset}/{args.task}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading graph data...")
    data, col_stats_dict = load_and_normalize_data(args.data_dir, args.dataset)

    task = get_task(args.dataset, args.task, download=False)

    # Safeguard: the node-type names in our stored graph (data.pt, normalized to
    # lowercase above) may not match the case of relbench's task.entity_table
    # (e.g. 'AdsInfo' for rel-avito). If they differ only by case, reconcile to
    # the graph's actual key so seed-node sampling doesn't raise KeyError.
    if task.entity_table not in data.node_types:
        ci_node_types = {nt.lower(): nt for nt in data.node_types}
        resolved = ci_node_types.get(task.entity_table.lower())
        if resolved is None:
            raise KeyError(
                f"Entity table '{task.entity_table}' not found among graph node "
                f"types {list(data.node_types)}"
            )
        print(f"Reconciled entity_table '{task.entity_table}' -> '{resolved}'")
        task.entity_table = resolved

    precomputed_dir = os.path.join(args.cache_dir, args.dataset, args.task)
    split_datasets = {
        split: RelGTTokens(
            data=data,
            task=task,
            K=args.K,
            split=split,
            undirected=True,
            precompute=True,
            precomputed_dir=precomputed_dir,
            num_workers=args.num_workers,
        )
        for split in ["train", "val", "test"]
    }

    loaders = {
        split: DataLoader(
            split_datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=split_datasets[split].collate,
            num_workers=2,
            persistent_workers=True,
            pin_memory=(device.type == "cuda"),
        )
        for split in ["train", "val", "test"]
    }

    # ── Task settings ─────────────────────────────────────────────────────
    clamp_min = clamp_max = None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels        = 1
        loss_fn             = BCEWithLogitsLoss()
        tune_metric         = "roc_auc"
        higher_is_better    = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels        = 1
        loss_fn             = L1Loss()
        tune_metric         = "mae"
        higher_is_better    = False
        train_table         = task.get_table("train")
        clamp_min, clamp_max = np.percentile(
            train_table.df[task.target_col].to_numpy(), [2, 98]
        )
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels        = task.num_labels
        loss_fn             = BCEWithLogitsLoss()
        tune_metric         = "multilabel_auprc_macro"
        higher_is_better    = True
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    # ── Model ─────────────────────────────────────────────────────────────
    train_ds = split_datasets["train"]
    cfg = RelGTConfig(
        local_num_layers=args.local_num_layers,
        channels=args.channels,
        out_channels=out_channels,
        global_dim=args.global_dim,
        heads=args.heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        conv_type="full",
        num_centroids=args.num_centroids,
        sample_node_len=args.K,
    )
    model = RelGT(
        num_nodes=data.num_nodes,
        max_neighbor_hop=train_ds.max_neighbor_hop,
        node_type_map=train_ds.node_type_to_index,
        col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
        col_stats_dict=col_stats_dict,
        **cfg.model_dump(),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_state      = None
    history         = {"epoch": [], "train_loss": [], "val_metric": [], "tune_metric": tune_metric}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, loss_fn,
                                 device, args.max_steps_per_epoch)
        val_preds   = evaluate(model, loaders["val"], task, device, clamp_min, clamp_max)
        val_metrics = task.evaluate(val_preds, task.get_table("val"))
        val_score   = val_metrics[tune_metric]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_metric"].append(val_score)

        improved = (higher_is_better and val_score >= best_val_metric) or \
                   (not higher_is_better and val_score <= best_val_metric)
        marker = " *" if improved else ""
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val {tune_metric} {val_score:.4f}{marker}")

        if improved:
            best_val_metric = val_score
            best_state      = copy.deepcopy(model.state_dict())

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(best_state)

    train_preds  = evaluate(model, loaders["train"], task, device, clamp_min, clamp_max)
    val_preds    = evaluate(model, loaders["val"],   task, device, clamp_min, clamp_max)
    test_preds   = evaluate(model, loaders["test"],  task, device, clamp_min, clamp_max)

    train_metrics = task.evaluate(train_preds, task.get_table("train"))
    val_metrics   = task.evaluate(val_preds,   task.get_table("val"))
    test_metrics  = task.evaluate(test_preds)

    print(f"\n{'='*50}")
    print(f"Dataset/Task : {args.dataset} / {args.task}")
    print(f"Train metrics: {train_metrics}")
    print(f"Val   metrics: {val_metrics}")
    print(f"Test  metrics: {test_metrics}")
    print(f"{'='*50}")

    # ── Save outputs ──────────────────────────────────────────────────────
    tag     = f"{args.dataset}_{args.task}"
    fig_dir = os.path.join("./figures", tag)
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    plot_learning_curves(
        history,
        out_path=os.path.join(fig_dir, "relgt_learning_curves.png"),
        title=f"RelGT — {args.dataset} / {args.task}",
    )

    results = {
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics":   val_metrics,
        "test_metrics":  test_metrics,
        "history": {k: v for k, v in history.items()},
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    model_path = os.path.join(out_dir, "best_model.pt")
    torch.save(best_state, model_path)
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
