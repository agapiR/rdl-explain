"""Train new RDL models with mild hyperparameter tuning on tier-1/2 tasks.

Adapted from gnn_entity.py (RelBench commit 74d4c37acb721659d266b3aa8dbdf23bbf841620)
but using the in-codebase Model from `rdl_explain.model.model`. Differences:
  * Uses our Model (no `gnn=`, `gat_heads=`, `gat_dropout=` kwargs).
  * Grid-searches over (lr, seed) per task; runs each combo and picks best by val.
  * Explicit fanouts (not the halving heuristic) — set larger than the existing
    pretrained models to look at bigger subgraphs.
  * Per-task `num_layers` is fixed at max of the existing best_model / explained_model
    settings in /home/rissakiagapi/rdl-explain-data/relbench_models/{db}/{task}/.
  * Channels bumped 32 → 64; epochs 30 with early stopping (patience 5).
  * Saves graph + col_stats once per task, then each run's state_dict + curves +
    parquet predictions + computed metrics + a gnn_params.json compatible with
    rdl_explain.model.config.ModelConfig.

Output layout:
  /home/rissakiagapi/rdl-explain-data/relbench_models_v2/{db}/{task}/
    graph_data.pt              (HeteroData, col_stats_dict, col_to_stype_dict)
    tuning_summary.json
    training_summary.png       (mini grid plot of val metric per run)
    runs/lr_{lr}_seed_{seed}/
      model_state_dict.pth
      gnn_params.json
      meta.yaml
      training_curve.json
      training_curve.png
      inference_results/
        computed_metrics.json
        predictions_{train,val,test}.parquet
    best_model/  (verbatim copy of best run by val metric)

Usage:
  python run_train_models_v2.py                                 # all 3 tasks, GPU 0
  python run_train_models_v2.py --task rel-trial/study-outcome  # single task
  python run_train_models_v2.py --gpu 1                         # different GPU
  python run_train_models_v2.py --lrs 5e-4 --seeds 42           # custom grid
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
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from relbench.base import EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input

# Make the in-project modules importable
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "src")
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdl_explain.model import Model
from rdl_explain.loaders import get_dataset_and_task, build_graph_from_db


# ── Task config ───────────────────────────────────────────────────────────────

DATA_ROOT     = "/home/rissakiagapi/rdl-explain-data"
OUT_ROOT      = os.path.join(DATA_ROOT, "relbench_models_v2")

# Per-task num_layers = max(best_model['gnn_layers'], explained_model['gnn_layers']) from legacy models.
# Per-task fanouts = 2× the existing pretrained fanouts (= "larger subgraphs").
TASKS: Dict[str, Dict[str, Any]] = {
    "rel-f1/driver-dnf": {
        "num_layers":  3,
        "fanouts":     [128, 64, 32],      # 2× of best [64,32] (pad +32), 2× of explained [64,32,16]
        "batch_size":  64,                 # 8 GB GPUs — channels=64 + these fanouts is tight
    },
    "rel-f1/driver-top3": {
        "num_layers":  3,
        "fanouts":     [128, 64, 32],
        "batch_size":  64,
    },
    "rel-trial/study-outcome": {
        # Existing best_model used 4 layers, but rel-trial's schema reaches most
        # substantive tables (outcomes, eligibilities, designs, interventions,
        # conditions, sponsors, facilities) within 2 hops; layers 3-4 mostly
        # cycle back to other studies via shared bridge tables and add noise.
        # 2 layers with bigger fanouts → larger 1-hop coverage, no return-trip.
        "num_layers":  2,
        "fanouts":     [128, 64],
        "batch_size":  128,
    },
    # ── relbench v2 tasks (only available in the relbench_v2 conda env) ──
    "rel-arxiv/paper-citation": {
        # 884k labeled instances, 6-node-type schema (papers, authors, categories,
        # paperAuthors, paperCategories, citations). 2 layers reach direct authors,
        # categories, cited/citing papers — enough relational context. Bigger
        # fanouts would cycle back to other papers via shared authors / categories
        # and add cost without much new signal.
        "num_layers":  2,
        "fanouts":     [128, 64],
        "batch_size":  128,
    },
    "rel-arxiv/author-category": {
        # authors → paperAuthors → papers → paperCategories → categories: the
        # category nodes are 4 bridge-table hops away. 2 layers would only
        # reach the paperAuthors bridge nodes (mostly timestamps/FKs) and
        # never see the actual paper content or categories. 4 layers gets
        # the model to the categories themselves.
        "num_layers":  4,
        "fanouts":     [64, 32, 16, 8],    # taper at deeper layers to keep
                                           # subgraph cost bounded
        "batch_size":  64,                 # 4-layer subgraph is bigger
    },
    "rel-hm/user-churn": {
        "num_layers":  2,
        # attempt 1 was 22 min/epoch at fanouts=[128,64,32,16] batch=64 — infeasible.
        # Smaller fanouts + less layers + bigger batch → much cheaper subgraph per step.
        "fanouts":     [128, 64],
        "batch_size":  128,
    },
    # ── new tasks added 2026-05 for cohort experiments ───────────────────────
    # All chosen for label balance + feature-rich schemas; sizes range from
    # tiny (user-repeat: 3.8k) to medium (user-churn: 374k).
    "rel-trial/studies-has_dmc": {
        # Same DB as study-outcome (15 tables, 140 cols); reuse that config.
        # studies-has_dmc is also study-entity, so the schema reach pattern is
        # identical — 2 hops cover designs / eligibilities / interventions / etc.
        "num_layers":  2,
        "fanouts":     [128, 64],
        "batch_size":  64,        # was 128 → drop to 64
    },
    "rel-event/user-repeat": {
        # Tiny task (3.8k train rows) on a fat DB (5 tables, 131 cols, 41M
        # total rows). Entity is users → user_friends, events, event_attendees.
        # 2 layers covers user → event → other-attendees.
        "num_layers":  2,
        "fanouts":     [128, 64],
        "batch_size":  64,
    },
    # rel-event/event_interest-interested removed — relbench test split has
    # entity ids (e.g. 15372) that exceed the event_interest table size (14978),
    # so the loader errors with IndexError on the test inference pass. Looks
    # like a relbench pkey-vs-local-index mismatch on that specific task.
    "rel-ratebeer/brewer-dormant": {
        # Brewers (98k train). 13 tables, 221 cols — richest schema in our set.
        # 2 hops only reach 11/13 tables; missing `users` and `place_ratings`
        # at hop 3. brewer-dormant's core signal lives in beer_ratings (hop 2)
        # so 2 layers would work, but bumping to 3 captures user-side signal
        # cheaply since this task is small (~99k train).
        # batch=128 OOM'd on 8 GB M10 — dropped to 64.
        "num_layers":  3,
        "fanouts":     [64, 32, 16],
        "batch_size":  64,        # was 128 → drop to 64
    },
    "rel-ratebeer/user-churn": {
        # Users (374k train). Same DB as brewer-dormant.
        # 2 hops miss 5/13 tables, including `brewers` (the core entity users
        # interact with). 3 hops reach brewers via user → beer_ratings → beers
        # → brewers — critical for churn-by-brewer-preference patterns.
        "num_layers":  3,
        "fanouts":     [64, 32, 16],
        "batch_size":  128,
    },
}

DEFAULT_LRS         = [5e-4, 1e-4, 5e-5]  # shifted down from {1e-3,5e-4,1e-4}; old best was at 1e-4
DEFAULT_SEEDS       = [11, 42, 123]
DEFAULT_EPOCHS      = 40                   # 30 → 40
PATIENCE            = 10                   # 5 → 10
MAX_STEPS_PER_EPOCH = None                 # None ⇒ unlimited; cap was starving big tasks
WEIGHT_DECAY        = 1e-5                 # mild L2 against the bigger-model overfit observed in attempt 1
CHANNELS            = 64
AGGR                = "mean"


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_loss_and_metric(task: EntityTask):
    """Returns (loss_fn, tune_metric, higher_is_better, out_channels, clamp_min, clamp_max)."""
    clamp_min = clamp_max = None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return BCEWithLogitsLoss(), "roc_auc", True, 1, None, None
    if task.task_type == TaskType.REGRESSION:
        train_df = task.get_table("train").df
        cmin, cmax = np.percentile(train_df[task.target_col].to_numpy(), [2, 98])
        return L1Loss(), "mae", False, 1, float(cmin), float(cmax)
    if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        return BCEWithLogitsLoss(), "multilabel_auprc_macro", True, task.num_labels, None, None
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return CrossEntropyLoss(), "multiclass_f1", True, task.num_classes, None, None
    raise ValueError(f"Unsupported task type: {task.task_type}")


def build_graph_for_task(
    db_name: str,
    task_name: str,
    cache_dir: str,
    device: torch.device,
) -> Tuple[Any, Dict, EntityTask, Dict]:
    """Materialize the HeteroData graph fresh for this task.

    Notes:
      * `get_dataset` with download=True repopulates ~/.cache/relbench/{db}.
      * `get_task` is called with download=False to avoid the rel-f1/driver-dnf
        pooch SHA256 mismatch (the CDN task ZIP no longer matches the hash
        baked into relbench v1.1.0). The first call falls through to computing
        tables from the DB and caches them locally.
      * Text embedding runs on CPU during materialization to avoid GPU OOM on
        8 GB devices (rel-hm has wide text columns). This is a one-time cost
        per task; everything downstream still runs on `device`.
    """
    # Graph construction now goes through the shared loader layer
    # (rdl_explain.loaders) so train-time and explain-time graphs are built by
    # the SAME code path. Cache locations are kept byte-for-byte identical to the
    # original script so previously-trained models stay consistent:
    #   * materialized features: {cache_dir}/{db}_none/materialized
    #   * stype proposal:        {cache_dir}/{db}/stypes.json
    # get_dataset_and_task applies the t/f label coercion and the
    # download_dataset=True / download_task=False split (see loaders.py).
    log(f"  loading dataset/task ({db_name}/{task_name})…")
    dataset, task = get_dataset_and_task(
        db_name, task_name, download_dataset=True, download_task=False)
    log(f"  materializing graph (text embedder on CPU; cache_dir={cache_dir})…")
    data, col_stats_dict, col_to_stype_dict = build_graph_from_db(
        dataset,
        cache_dir=f"{cache_dir}/{db_name}_none",
        stype_cache_path=f"{cache_dir}/{db_name}/stypes.json",
        text_embedder_device="cpu",
    )
    return data, col_stats_dict, task, col_to_stype_dict


def make_loaders(
    data,
    task: EntityTask,
    fanouts: List[int],
    batch_size: int,
    temporal_strategy: str = "uniform",
    num_workers: int = 0,
) -> Dict[str, NeighborLoader]:
    """Eval loaders for all splits + a separate shuffle=True train loader.

    `temporal_strategy` is one of:
      * "uniform" — neighbours picked uniformly at random from all neighbours
                    whose timestamp ≤ seed_time. Default; broad temporal context.
      * "last"    — neighbours picked from the most-recent ones before seed_time.
                    Sharper signal when recency is informative.
    """
    loaders: Dict[str, NeighborLoader] = {}
    for split in ["train", "val", "test"]:
        table       = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=fanouts,
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            shuffle=False,
            num_workers=num_workers,
        )
    # Separate shuffled loader for training-time
    train_table = task.get_table("train")
    train_input = get_node_train_table_input(table=train_table, task=task)
    loaders["train_shuffle"] = NeighborLoader(
        data,
        num_neighbors=fanouts,
        time_attr="time",
        input_nodes=train_input.nodes,
        input_time=train_input.time,
        transform=train_input.transform,
        batch_size=batch_size,
        temporal_strategy=temporal_strategy,
        shuffle=True,
        num_workers=num_workers,
    )
    return loaders


@torch.no_grad()
def predict(model, loader, task, clamp_min, clamp_max, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, task.entity_table)
        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)
        if task.task_type in (TaskType.BINARY_CLASSIFICATION,
                              TaskType.MULTILABEL_CLASSIFICATION):
            pred = torch.sigmoid(pred)
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = torch.softmax(pred, dim=1)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        out.append(pred.detach().cpu())
    return torch.cat(out, dim=0).numpy()


def train_one_epoch(model, loader, optimizer, loss_fn, task, entity_table, device,
                    max_steps=None) -> float:
    model.train()
    loss_acc = count = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, task.entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            loss = loss_fn(pred, batch[entity_table].y.long())
        else:
            loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()
        loss_acc += loss.detach().item() * pred.size(0)
        count    += pred.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return loss_acc / count


def save_predictions_parquet(
    task: EntityTask, split: str, pred: np.ndarray, out_path: str,
) -> None:
    """Save a parquet with [entity_id, time, prediction(s), label?]."""
    table = task.get_table(split)
    df    = table.df.copy()
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
    data,
    col_stats_dict,
    task: EntityTask,
    cfg: Dict[str, Any],
    channels: int,
    lr: float,
    seed: int,
    epochs: int,
    patience: int,
    out_dir: str,
    device: torch.device,
    max_steps_per_epoch=None,
    temporal_strategy: str = "uniform",
) -> Dict[str, Any]:
    """Train a single (lr, seed) config; save artifacts; return summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "inference_results"), exist_ok=True)

    seed_everything(seed)

    loss_fn, tune_metric, higher_is_better, out_channels, clamp_min, clamp_max = \
        task_loss_and_metric(task)

    loaders = make_loaders(
        data, task,
        fanouts=cfg["fanouts"],
        batch_size=cfg["batch_size"],
        temporal_strategy=temporal_strategy,
    )
    train_loader_input = get_node_train_table_input(table=task.get_table("train"), task=task)
    entity_table       = train_loader_input.nodes[0]

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=cfg["num_layers"],
        channels=channels,
        out_channels=out_channels,
        aggr=AGGR,
        norm="batch_norm",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Pre-flight: detect tasks whose test split has entity ids beyond the
    # graph's entity-table node count. Iterating such a loader hard-crashes
    # the C++ neighbor sampler with a segfault (uncatchable). When detected,
    # skip test eval throughout training; train+val proceed normally.
    _n_entity_nodes = int(data[entity_table].num_nodes)
    try:
        _test_ids = task.get_table("test").df[task.entity_col].astype(int).values
        _test_max = int(_test_ids.max()) if len(_test_ids) else -1
    except Exception:
        _test_max = -1
    _skip_test = _test_max >= _n_entity_nodes
    if _skip_test:
        log(f"      [pre-flight] test split has entity id {_test_max} ≥ "
            f"{_n_entity_nodes} nodes in graph — skipping test eval to avoid "
            f"PyG sampler segfault. See pkey_vs_local_node_index.md.")

    curve = {"epoch": [], "train_loss": [],
             "val_metric": [], "val_all": [],
             "test_metric": [], "test_all": []}
    best_state = None
    best_val   = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    epochs_since_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train_shuffle"], optimizer,
                                     loss_fn, task, entity_table, device,
                                     max_steps=max_steps_per_epoch)
        val_pred    = predict(model, loaders["val"], task, clamp_min, clamp_max, device)
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        val_metric  = float(val_metrics[tune_metric])

        # Per-epoch test eval — labels are hidden client-side but task.evaluate()
        # has them internally, so we get the test metric here too. Diagnoses
        # train/val/test distribution shift visible in the training curve.
        # Skipped entirely (no loader iteration) when the test split contains
        # entity ids beyond the graph node count — that triggers a hard C++
        # segfault in the PyG sampler that try/except cannot catch.
        # See pkey_vs_local_node_index.md for the underlying issue.
        if _skip_test:
            test_pred    = None
            test_metrics = {}
            test_metric  = float('nan')
        else:
            test_pred    = predict(model, loaders["test"], task, clamp_min, clamp_max, device)
            test_metrics = task.evaluate(test_pred)
            test_metric  = float(test_metrics.get(tune_metric, float('nan')))

        curve["epoch"].append(epoch)
        curve["train_loss"].append(float(train_loss))
        curve["val_metric"].append(val_metric)
        curve["val_all"].append({k: float(v) for k, v in val_metrics.items()})
        curve["test_metric"].append(test_metric)
        curve["test_all"].append({k: float(v) for k, v in test_metrics.items()})

        # Best-epoch selection stays on val (test would be peeking).
        improved = (higher_is_better and val_metric >= best_val) or \
                   (not higher_is_better and val_metric <= best_val)
        if improved:
            best_val    = val_metric
            best_epoch  = epoch
            best_state  = copy.deepcopy(model.state_dict())
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

    # Load best, run inference on all splits
    assert best_state is not None
    model.load_state_dict(best_state)
    train_pred = predict(model, loaders["train"], task, clamp_min, clamp_max, device)
    val_pred   = predict(model, loaders["val"],   task, clamp_min, clamp_max, device)
    if _skip_test:
        test_pred    = None
        test_metrics = {}
    else:
        test_pred    = predict(model, loaders["test"],  task, clamp_min, clamp_max, device)
        test_metrics = task.evaluate(test_pred)   # labels hidden — metrics empty/skipped

    train_metrics = task.evaluate(train_pred, task.get_table("train"))
    val_metrics   = task.evaluate(val_pred,   task.get_table("val"))

    # Save artifacts
    torch.save(best_state, os.path.join(out_dir, "model_state_dict.pth"))

    gnn_params = {
        "gnn_layers":   cfg["num_layers"],
        "channels":     channels,
        "aggr":         AGGR,
        "norm":         "batch_norm",
        "fanouts":      list(cfg["fanouts"]),
        "out_channels": out_channels,
        # extras (ignored by ModelConfig but informative)
        "seed":         seed,
        "learning_rate": lr,
        "weight_decay":  WEIGHT_DECAY,
        "train_batch_size":   cfg["batch_size"],
        "test_batch_size":    cfg["batch_size"],
        "gnn_aggregation":    AGGR,
        "temporal_strategy":  temporal_strategy,
    }
    with open(os.path.join(out_dir, "gnn_params.json"), "w") as f:
        json.dump(gnn_params, f, indent=2)

    # Test metric AT best-val epoch (computed below for the return value too).
    _best_test_metric = float('nan')
    if curve["test_metric"] and 1 <= best_epoch <= len(curve["test_metric"]):
        _best_test_metric = float(curve["test_metric"][best_epoch - 1])

    meta = {
        "task":          f"{task.dataset_name}/{task.task_name}" if hasattr(task, "dataset_name") else None,
        "lr":            lr,
        "seed":          seed,
        "epochs_run":    len(curve["epoch"]),
        "best_epoch":    best_epoch,
        "best_val_metric":     best_val,
        # Test metric at the best-val epoch — selection was on val so this is
        # not "test-leaked", it's the held-out performance of the picked model.
        "best_test_metric":    _best_test_metric,
        "tune_metric":         tune_metric,
        "higher_is_better":    higher_is_better,
        "elapsed_seconds":     round(elapsed, 1),
        "channels":            channels,
        "aggr":                AGGR,
        "num_layers":          cfg["num_layers"],
        "fanouts":             list(cfg["fanouts"]),
        "batch_size":          cfg["batch_size"],
        "patience":            patience,
        "temporal_strategy":   temporal_strategy,
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
        if pred is None:   # test was skipped due to pkey/local-index mismatch
            continue
        save_predictions_parquet(
            task, split, pred,
            os.path.join(out_dir, "inference_results", f"predictions_{split}.parquet"),
        )

    # Free GPU memory before next run
    del model, optimizer, best_state, loaders
    torch.cuda.empty_cache()

    # Test metric AT the best-val epoch (peeking-free: selection was on val).
    best_test_metric = float('nan')
    if curve["test_metric"] and 1 <= best_epoch <= len(curve["test_metric"]):
        best_test_metric = float(curve["test_metric"][best_epoch - 1])

    return {
        "lr":               lr,
        "seed":             seed,
        "best_epoch":       best_epoch,
        "best_val_metric":  best_val,
        "best_test_metric": best_test_metric,
        "tune_metric":      tune_metric,
        "elapsed_seconds":  round(elapsed, 1),
        "out_dir":          out_dir,
    }


# ── Per-task driver ───────────────────────────────────────────────────────────

def run_task(
    db_task: str,
    *,
    lrs: List[float],
    seeds: List[int],
    epochs: int,
    patience: int,
    cache_dir: str,
    device: torch.device,
    max_steps_per_epoch=None,
    channels_override=None,
    num_layers_override=None,
    fanouts_override=None,
    batch_size_override=None,
    out_suffix=None,
    temporal_strategy: str = "uniform",
) -> None:
    db_name, task_name = db_task.split("/")

    # Per-task config + apply overrides without mutating the module-level dict.
    cfg      = dict(TASKS[db_task])
    channels = channels_override if channels_override is not None else CHANNELS
    if num_layers_override is not None:
        cfg["num_layers"] = num_layers_override
    if fanouts_override is not None:
        cfg["fanouts"]    = list(fanouts_override)
    if batch_size_override is not None:
        cfg["batch_size"] = batch_size_override
    # Final consistency check — fanouts and num_layers must match after overrides.
    if len(cfg["fanouts"]) != cfg["num_layers"]:
        raise ValueError(
            f"{db_task}: fanouts length {len(cfg['fanouts'])} ≠ "
            f"num_layers {cfg['num_layers']}. "
            f"If overriding one, override the other to match."
        )

    suffix       = f"_{out_suffix}" if out_suffix else ""
    out_dir_task = os.path.join(OUT_ROOT, db_name, f"{task_name}{suffix}")
    os.makedirs(out_dir_task, exist_ok=True)

    log(f"=== {db_task}  (num_layers={cfg['num_layers']}, "
        f"fanouts={cfg['fanouts']}, channels={channels}, "
        f"batch_size={cfg['batch_size']}) ===")
    if out_suffix:
        log(f"  out_dir_task = {out_dir_task}")

    # Build / cache graph once per task
    data, col_stats_dict, task, col_to_stype_dict = build_graph_for_task(
        db_name, task_name, cache_dir=cache_dir, device=device,
    )

    graph_path = os.path.join(out_dir_task, "graph_data.pt")
    if not os.path.exists(graph_path):
        log(f"  saving graph + col_stats → {graph_path}")
        torch.save(
            {
                "data":               data,
                "col_stats_dict":     col_stats_dict,
                "col_to_stype_dict":  {t: {c: s.value for c, s in m.items()}
                                       for t, m in col_to_stype_dict.items()},
                "db_name":            db_name,
                "task_name":          task_name,
            },
            graph_path,
        )

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
                channels=channels,
                lr=lr,
                seed=seed,
                epochs=epochs,
                patience=patience,
                out_dir=out_dir,
                device=device,
                max_steps_per_epoch=max_steps_per_epoch,
                temporal_strategy=temporal_strategy,
            )
            summaries.append(summary)

    # Pick best
    _, _, higher_is_better, *_ = task_loss_and_metric(task)
    summaries_sorted = sorted(
        summaries, key=lambda s: s["best_val_metric"],
        reverse=higher_is_better,
    )
    best = summaries_sorted[0]

    log(f"  best: lr={best['lr']}, seed={best['seed']}, "
        f"val_{best['tune_metric']}={best['best_val_metric']:.4f}  "
        f"test_{best['tune_metric']}={best.get('best_test_metric', float('nan')):.4f}")

    # Copy best → best_model/
    best_dst = os.path.join(out_dir_task, "best_model")
    if os.path.exists(best_dst):
        shutil.rmtree(best_dst)
    shutil.copytree(best["out_dir"], best_dst)

    # Write summary
    with open(os.path.join(out_dir_task, "tuning_summary.json"), "w") as f:
        json.dump(
            {
                "db_task":          db_task,
                "channels":         channels,
                "num_layers":       cfg["num_layers"],
                "fanouts":          cfg["fanouts"],
                "batch_size":       cfg["batch_size"],
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
            },
            f, indent=2,
        )

    # Mini per-task grid plot — val and test side-by-side per run.
    n     = len(summaries)
    xs    = [f"lr={s['lr']:g}\nseed={s['seed']}" for s in summaries]
    ys_v  = [s["best_val_metric"]                     for s in summaries]
    ys_t  = [s.get("best_test_metric", float('nan'))  for s in summaries]
    x_pos = np.arange(n)
    bw    = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * n + 1.5), 4))
    bars_v = ax.bar(x_pos - bw/2, ys_v, bw, label="val",  edgecolor="black",
                    color="tab:blue")
    bars_t = ax.bar(x_pos + bw/2, ys_t, bw, label="test", edgecolor="black",
                    color="tab:green")
    # Highlight the best run (chosen by val) with a thicker edge on both bars.
    best_idx = next((i for i, s in enumerate(summaries)
                     if s["out_dir"] == best["out_dir"]), None)
    if best_idx is not None:
        for bar in (bars_v[best_idx], bars_t[best_idx]):
            bar.set_edgecolor("red")
            bar.set_linewidth(2.0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs, fontsize=8)
    ax.set_ylabel(best['tune_metric'])
    ax.set_title(f"{db_task} — tuning runs (red border = best by val)")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
    parser.add_argument("--task", nargs="*", default=None,
                        help="One or more db/task strings. Defaults to all in TASKS.")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Cap on optimizer steps per training epoch. "
                             "Useful for large datasets where a full epoch is "
                             "too slow (e.g. rel-hm/user-churn). Default: no cap.")
    # Per-task architecture overrides (apply to every task selected by --task).
    # Useful for sweeping width/depth/receptive-field without editing TASKS.
    parser.add_argument("--channels", type=int, default=None,
                        help="Override CHANNELS for this run (applies to all tasks).")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override per-task num_layers for this run.")
    parser.add_argument("--fanouts", nargs="+", type=int, default=None,
                        help="Override per-task fanouts for this run, e.g. "
                             "--fanouts 128 64 32 16. Length must match --num-layers "
                             "if both are given.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override per-task batch_size for this run.")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Append to OUT_ROOT/{db}/{task}/ so different "
                             "sweeps don't clobber each other "
                             "(e.g. --out-suffix ch32_L2 → …/{task}_ch32_L2/).")
    parser.add_argument("--temporal-strategy", choices=["uniform", "last"],
                        default="uniform",
                        help="NeighborLoader temporal strategy. 'uniform' (default) "
                             "samples neighbours uniformly from times ≤ seed_time; "
                             "'last' picks the most recent neighbours before seed_time. "
                             "Try 'last' on tasks where recency is informative.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.expanduser("~/.cache/relbench_examples"))
    args = parser.parse_args()

    # GPU pinning happens via CUDA_VISIBLE_DEVICES; if user already set it,
    # we honour that and just use cuda:0. Otherwise set it from --gpu.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    tasks = args.task or list(TASKS.keys())
    for t in tasks:
        if t not in TASKS:
            log(f"WARNING: {t} not in TASKS config; skipping")
            continue

    # Sanity-check the fanouts/num_layers pair.
    if args.fanouts is not None and args.num_layers is not None \
            and len(args.fanouts) != args.num_layers:
        raise ValueError(
            f"--fanouts has length {len(args.fanouts)} but --num-layers={args.num_layers}; "
            f"they must match."
        )

    overrides = {
        "channels":   args.channels,
        "num_layers": args.num_layers,
        "fanouts":    args.fanouts,
        "batch_size": args.batch_size,
    }
    nonnull_overrides = {k: v for k, v in overrides.items() if v is not None}

    log(f"Device={device}  Tasks={tasks}  lrs={args.lrs}  seeds={args.seeds}  "
        f"max_steps_per_epoch={args.max_steps_per_epoch}  "
        f"temporal_strategy={args.temporal_strategy}")
    if nonnull_overrides:
        log(f"Architecture overrides: {nonnull_overrides}")
    if args.out_suffix:
        log(f"Output suffix: '_{args.out_suffix}'")
    log(f"OUT_ROOT={OUT_ROOT}")

    for t in tasks:
        if t not in TASKS:
            continue
        try:
            run_task(
                t,
                lrs=args.lrs,
                seeds=args.seeds,
                epochs=args.epochs,
                patience=args.patience,
                cache_dir=args.cache_dir,
                device=device,
                max_steps_per_epoch=args.max_steps_per_epoch,
                channels_override=args.channels,
                num_layers_override=args.num_layers,
                fanouts_override=args.fanouts,
                batch_size_override=args.batch_size,
                out_suffix=args.out_suffix,
                temporal_strategy=args.temporal_strategy,
            )
        except Exception as e:
            log(f"ERROR on task {t}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    log("done.")


if __name__ == "__main__":
    main()
