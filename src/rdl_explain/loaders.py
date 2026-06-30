"""Data / model loading layer for the RDL explanation scripts.

Provides functions for database / graph / model loading, used by 
the experiment scripts.

Two RelBench loading modes, selected by ``loader`` in the model config:

  * ``loader: v1`` — load a **pre-built, cached graph** (we do not load the
    original database objects). Reads::

        {graph_dir}/data.pt
        {graph_dir}/col_stats_dict.pt

    (In the v1 layout ``graph_dir`` is the per-*database* dir, e.g.
    ``relbench_models/rel-f1``, since data.pt is shared across that db's tasks.)

  * ``loader: v2`` — **build the graph from the RelBench database** via
    ``make_pkey_fkey_graph`` (``get_db`` → ``get_stype_proposal`` → graph), so the
    graph stays tied to the source data. RelBench materializes features into
    ``cache_dir`` once, so subsequent builds are fast. Construction is
    deterministic given (db, stypes, text-embedder), so the rebuilt graph matches
    the one the checkpoint was trained on. Optionally also reads/writes a
    ``{graph_dir}/graph_data.pt`` bundle (see ``load_graph_bundle`` for the
    reverse — RelGT training reuses the GNN-built bundle this way).

**Cache caveat (important for the tight graph↔DB relation).** RelBench's
``make_pkey_fkey_graph(cache_dir=…)`` caches the *materialized per-table feature
tensors* (one ``.pt`` per table, including the GloVe-embedded text columns) keyed
by the **directory path, not by DB content**. RelBench does NOT auto-invalidate
the cache when the database or the stype proposal changes. So if you bump the
dataset version (e.g. 1.1.0 → 2.1.2) or change stypes but reuse the same
``cache_dir``, you will silently get **stale features**. The tight relation holds
only if a given ``cache_dir`` is dedicated to one (db-version, stypes) pair —
**use a fresh cache_dir (or clear it) whenever the dataset version or stypes
change.** The ``stypes.json`` cache written here has the same property: delete it
if the stype proposal should be recomputed.

The RelBench task API (``get_dataset`` / ``get_task`` / ``make_pkey_fkey_graph``)
is identical across relbench 1.1.0 and 2.1.2; the only version-specific aspect is
which datasets / dataset versions (note that some datasets are updated from 1.1.0 to 2.1.2) 
are available for each relbench version.
"""

import json
import yaml
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype as _stype
from torch_frame.config.text_embedder import TextEmbedderConfig

from rdl_explain.explain.config import ExplainerConfig
from rdl_explain.model import Model
from rdl_explain.model.text_embedder import GloveTextEmbedding


# ── Config ────────────────────────────────────────────────────────────────────

class RunConfig(ExplainerConfig):
    """Unified config for the experiment scripts: model architecture + inference
    (inherited from ExplainerConfig) + loader settings. One object is passed to
    ``construct_graph`` / ``load_model`` / ``RDLExplainer``.
    """
    # Model architecture (for load_model / Model construction)
    channels: int = 64
    out_channels: int = 1
    aggr: Literal["mean", "sum", "max"] = "mean"
    norm: Literal["batch_norm", "none"] = "batch_norm"
    shallow_list: List[str] = []
    id_awareness: bool = False

    # Loader settings
    loader: Literal["v1", "v2"] = "v1"
    graph_dir: Optional[str] = None          # v1: dir with data.pt/col_stats_dict.pt; v2: optional bundle dir
    cache_dir: Optional[str] = None          # v2: relbench feature-materialization cache
    text_embedder_device: str = "cpu"        # v2: where GloVe runs during build
    save_graph_bundle: bool = False          # v2: also write {graph_dir}/graph_data.pt


def _read_config_file(path: str) -> Dict[str, Any]:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_config(model_config_path: str) -> RunConfig:
    """Read a config FILE into a RunConfig.

    USED BY: the legacy CLI experiment scripts (``learn_masks.py`` etc., via
    ``--model_config``). Programmatic code (notebooks / new experiments) does NOT
    need this — construct a ``RunConfig(...)`` directly. (TODO: revisit and
    simplify the file-based config layer.)

    Parses a trained model's ``gnn_params.json`` or a hand-written yaml/json.
    Unknown keys (``seed``, ``learning_rate``, …) are ignored.

    ``fanouts`` and ``num_neighbors`` are the SAME quantity — the number of
    neighbors sampled per GNN layer (``fanouts`` is the term ``gnn_params.json``
    uses; ``num_neighbors`` is PyG's NeighborLoader / ExplainerConfig term).
    Either one populates ``num_neighbors``; if both are present and DISAGREE that
    is a config error. ` 
    Explanation-time inference should using the same ``fanouts`` / ``num_neighbors`` 
    the model was trained with.
    """
    d = dict(_read_config_file(model_config_path))
    fanouts = d.pop("fanouts", None)
    num_neighbors = d.get("num_neighbors", None)
    if fanouts is not None and num_neighbors is not None \
            and list(fanouts) != list(num_neighbors):
        raise ValueError(
            f"config sets both 'fanouts'={fanouts} and "
            f"'num_neighbors'={num_neighbors}, but they are the same quantity "
            f"and disagree — set only one."
        )
    if num_neighbors is None and fanouts is not None:
        d["num_neighbors"] = list(fanouts)
    return RunConfig(**d)


# ── Dataset + task ────────────────────────────────────────────────────────────

# Some RelBench tasks store binary labels as Postgres 't'/'f' strings; relbench's
# get_node_train_table_input casts the target to float and would raise. Coerce.
_TF_MAP = {"t": 1, "true": 1, "T": 1, "True": 1, "TRUE": 1,
           "f": 0, "false": 0, "F": 0, "False": 0, "FALSE": 0,
           "yes": 1, "Yes": 1, "no": 0, "No": 0}


def _patch_tf_labels(task) -> None:
    for split in ("train", "val", "test"):
        try:
            df = task.get_table(split).df
            col = task.target_col
        except Exception:
            continue
        if col in df.columns and df[col].dtype == object:
            mapped = df[col].astype(str).map(_TF_MAP)
            if mapped.notna().all():
                df[col] = mapped.astype(float)


def get_dataset_and_task(
    db_name: str,
    task_name: str,
    download_dataset: bool = False,
    download_task: bool = False,
) -> Tuple[Any, Any]:
    """Programmatic core: return ``(dataset, task)`` for a db/task.

    Download flags are split because RelBench needs them independently: training
    repopulates the dataset (``download_dataset=True``) but loads the task with
    ``download_task=False`` to avoid a stale-hash failure on some tasks. (RelBench
    verifies downloaded task ZIPs against a SHA256 baked into the release; for
    e.g. rel-f1/driver-dnf the CDN file no longer matches, so ``download=True``
    fails the integrity check — ``download_task=False`` computes the task tables
    from the database instead.)

    Attaches ``dataset_name`` / ``task_name`` to the task (RelBench doesn't expose
    them) and coerces 't'/'f' string labels to 0/1.
    """
    dataset = get_dataset(db_name, download=download_dataset)
    task = get_task(db_name, task_name, download=download_task)
    _patch_tf_labels(task)
    try:
        task.dataset_name = db_name
        task.task_name = task_name
    except (AttributeError, TypeError):
        pass
    return dataset, task


def load_dataset_and_task(data_config_path: str) -> Tuple[Any, Any]:
    """Load the RelBench dataset + task named in a small data config FILE.

    USED BY: the legacy CLI experiment scripts (via ``--data_config``).
    Programmatic code should call :func:`get_dataset_and_task` with the
    ``db_name`` / ``task_name`` directly — no file needed. This is just a thin
    file-reading wrapper around it. (TODO: revisit and simplify.)

    Config fields: ``db_name``, ``task_name``, optional ``download`` (default
    False — artifacts are expected to be cached; applies to both dataset and
    task). ``dataset_name`` / ``task_name`` are attached to the returned task
    (RelBench doesn't expose them).
    """
    d = _read_config_file(data_config_path)
    download = bool(d.get("download", False))
    return get_dataset_and_task(
        d["db_name"], d["task_name"],
        download_dataset=download, download_task=download,
    )


# ── Graph ─────────────────────────────────────────────────────────────────────

def _load_graph_v1(graph_dir: str):
    data_path = os.path.join(graph_dir, "data.pt")
    stats_path = os.path.join(graph_dir, "col_stats_dict.pt")
    if not (os.path.exists(data_path) and os.path.exists(stats_path)):
        raise FileNotFoundError(
            f"v1 loader expects {data_path} and {stats_path}. "
            f"Set graph_dir to the per-database dir (e.g. .../relbench_models/rel-f1)."
        )
    data = torch.load(data_path, weights_only=False)
    col_stats_dict = torch.load(stats_path, weights_only=False)
    return data, col_stats_dict


def build_graph_from_db(
    dataset,
    *,
    cache_dir: Optional[str] = None,
    stype_cache_path: Optional[str] = None,
    text_embedder_device: str = "cpu",
) -> Tuple[Any, Dict, Dict]:
    """Build the hetero graph from the RelBench database.

    Single source of truth for v2 graph construction — used both by
    :func:`construct_graph` (explanation side) and by the training scripts
    (``run_train_models_v2.py``) so train-time and explain-time graphs are built
    identically. Deterministic given (db, stypes, text-embedder).

    The materialized feature cache lives at ``{cache_dir}/materialized``. The
    stype proposal is cached at ``stype_cache_path`` if given, else at
    ``{cache_dir}/stypes.json``. (The training scripts pass an explicit
    ``stype_cache_path`` to reuse their pre-existing stype cache byte-for-byte.)
    See the module docstring for the cache-staleness caveat. Returns
    ``(data, col_stats_dict, col_to_stype)``.
    """
    db = dataset.get_db()

    # stype proposal, cached as json (explicit path wins, else under cache_dir)
    stype_cache = stype_cache_path or (
        os.path.join(cache_dir, "stypes.json") if cache_dir else None)
    if stype_cache and os.path.exists(stype_cache):
        col_to_stype = json.load(open(stype_cache))
        col_to_stype = {t: {c: _stype(s) for c, s in m.items()}
                        for t, m in col_to_stype.items()}
    else:
        col_to_stype = get_stype_proposal(db)
        if stype_cache:
            os.makedirs(os.path.dirname(stype_cache), exist_ok=True)
            json.dump({t: {c: s.value for c, s in m.items()}
                       for t, m in col_to_stype.items()},
                      open(stype_cache, "w"), indent=2)

    materialized = os.path.join(cache_dir, "materialized") if cache_dir else None
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(
                device=torch.device(text_embedder_device)),
            batch_size=256,
        ),
        cache_dir=materialized,
    )
    return data, col_stats_dict, col_to_stype


def load_graph_bundle(bundle_path: str) -> Tuple[Any, Dict]:
    """Load a ``graph_data.pt`` bundle written during v2 training.

    Returns ``(data, col_stats_dict)``. Used by RelGT training (which must run on
    the exact graph the GNN training built) and as a fast explanation-time path.
    """
    pkg = torch.load(bundle_path, weights_only=False)
    return pkg["data"], pkg["col_stats_dict"]


def construct_graph(config: RunConfig, dataset) -> Tuple[Any, Dict]:
    """Return ``(data, col_stats_dict)``.

    ``loader == 'v1'`` loads the cached graph from ``config.graph_dir``;
    ``loader == 'v2'`` builds it from the RelBench ``dataset`` database (and
    optionally writes a ``{graph_dir}/graph_data.pt`` bundle).
    """
    if config.loader == "v1":
        return _load_graph_v1(config.graph_dir)
    data, col_stats_dict, _ = build_graph_from_db(
        dataset, cache_dir=config.cache_dir,
        text_embedder_device=config.text_embedder_device,
    )
    if config.save_graph_bundle and config.graph_dir:
        os.makedirs(config.graph_dir, exist_ok=True)
        torch.save({"data": data, "col_stats_dict": col_stats_dict},
                   os.path.join(config.graph_dir, "graph_data.pt"))
    return data, col_stats_dict


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(
    config: RunConfig,
    model_params_path: str,
    construct: bool = True,
    data=None,
    col_stats_dict=None,
    task=None,
) -> Model:
    """Build a Model from the config architecture and load a trained state dict.

    ``construct`` is kept for signature compatibility (always builds). ``task``
    is accepted but unused (out_channels comes from the config).
    """
    if not construct:
        raise NotImplementedError("load_model only supports construct=True.")
    if data is None or col_stats_dict is None:
        raise ValueError("load_model requires `data` and `col_stats_dict`.")

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=config.gnn_layers,
        channels=config.channels,
        out_channels=config.out_channels,
        aggr=config.aggr,
        norm=config.norm,
        shallow_list=config.shallow_list,
        id_awareness=config.id_awareness,
    ).to(config.device)

    state_dict = torch.load(model_params_path, weights_only=False,
                            map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
