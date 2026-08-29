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

import numpy as np
import pandas as pd
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
    # `aggr` is the SAGEConv message aggregation, i.e. `conv_aggregation` /
    # `gnn_aggregation` in the training configs. It is NOT a learned parameter,
    # so a wrong value still passes load_state_dict(strict=True) and silently
    # produces a broken model -- count-based tasks trained with 'sum' collapse
    # from ROC-AUC 1.00 to 0.10 under 'mean'. Always set it from the checkpoint's
    # own config, and confirm with `verify_checkpoint`.
    aggr: Literal["mean", "sum", "max"] = "mean"
    norm: Literal["batch_norm", "layer_norm", "none"] = "batch_norm"
    encoder_layers: int = 2
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


#: Config keys that mean the same thing as a RunConfig field. The training
#: pipelines spell the architecture differently from `Model`'s constructor, and
#: pydantic silently ignores unknown keys -- so an unresolved alias becomes a
#: wrong-but-plausible model rather than an error.
_ARCH_ALIASES = {
    "gnn_aggregation": "aggr",       # gnn_params.json (RelBench models)
    "conv_aggregation": "aggr",      # model_config.yaml (case-study models)
    "num_gnn_layers": "gnn_layers",
    "encoder_num_layers": "encoder_layers",
}

#: Architectural keys that are NOT silently ignorable: if one of these appears
#: with a value the open Model cannot honour, loading must fail loudly.
_ARCH_UNSUPPORTED = {
    # HeteroConv's aggregation is hardcoded to "sum" in model/gnn/nn.py.
    "hetero_conv_aggregation": ("sum",),
}


def _resolve_architecture_aliases(d: Dict[str, Any]) -> None:
    """Rename known architecture aliases in place; reject unsupported values."""
    for alias, field in _ARCH_ALIASES.items():
        if alias not in d:
            continue
        value = d.pop(alias)
        if field in d and d[field] != value:
            raise ValueError(
                f"config sets both {alias!r}={value!r} and {field!r}={d[field]!r}, "
                f"but they are the same quantity and disagree — set only one."
            )
        d[field] = value

    for key, allowed in _ARCH_UNSUPPORTED.items():
        if key in d and d[key] not in allowed:
            raise ValueError(
                f"config sets {key!r}={d[key]!r}, but this implementation only "
                f"supports {allowed[0]!r} (hardcoded in model/gnn/nn.py). "
                f"Loading a checkpoint trained otherwise would silently produce "
                f"a different model."
            )
        d.pop(key, None)


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

    Architecture aliases are resolved rather than ignored. ``gnn_aggregation``
    (gnn_params.json) and ``conv_aggregation`` (training model_config.yaml) both
    mean ``aggr``; these were previously dropped as unknown keys, so a model
    trained with ``sum`` was silently rebuilt with the ``mean`` default. Any
    remaining key that looks architectural raises instead of being ignored.
    """
    d = dict(_read_config_file(model_config_path))
    _resolve_architecture_aliases(d)
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
        encoder_layers=config.encoder_layers,
    ).to(config.device)

    state_dict = torch.load(model_params_path, weights_only=False,
                            map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Checkpoint verification ───────────────────────────────────────────────────

def verify_checkpoint(
    model: Model,
    data,
    entity_table: str,
    reference: pd.DataFrame,
    num_neighbors: List[int],
    node_index_col: str,
    node_index_offset: int = 0,
    time_col: Optional[str] = None,
    temporal_strategy: str = "uniform",
    output_col: str = "output",
    batch_size: int = 512,
    min_correlation: float = 0.99,
    max_abs_diff: Optional[float] = None,
    device: str = "cpu",
    raise_on_mismatch: bool = True,
) -> Dict[str, float]:
    """Re-run inference and check it reproduces the checkpoint's own predictions.

    WHY THIS EXISTS. Most of a model's architecture is NOT recoverable from its
    state dict, because the settings involved have no parameters of their own:
    the message aggregation (``aggr``), the sampling fanouts, and the text
    embedder all leave the tensor shapes untouched. A wrong choice therefore
    passes ``load_state_dict(strict=True)`` without complaint and yields a model
    that runs, produces confident numbers, and is wrong — a count-based task
    trained with ``aggr='sum'`` scores ROC-AUC 0.10 when rebuilt with ``'mean'``.
    Explanations computed on such a model look entirely plausible.

    Comparing against predictions the trained model actually wrote is the only
    check that covers all of these at once, including mismatches nobody
    anticipated. Run it once, right after loading, before explaining anything.

    Args:
        model:            the loaded Model.
        data:             the HeteroData graph it was loaded onto.
        entity_table:     prediction entity node type (graph capitalisation).
        reference:        frame of stored predictions, with ``output_col`` and a
                          node-index column.
        num_neighbors:    fanouts; must match those used at training/inference.
        node_index_col:   column holding the entity id (see NOTE).
        node_index_offset: added to ``node_index_col`` to get the graph node
                          index (see NOTE). Explicit, never guessed.
        time_col:         datetime column of seed timestamps. REQUIRED for a
                          temporal graph (see NOTE); None for an atemporal one.
        output_col:       column holding the raw model output (pre-sigmoid).
        min_correlation:  minimum acceptable Pearson correlation between
                          recomputed and stored outputs. This -- not absolute
                          difference -- is the gate; see NOTE on sampling.
        max_abs_diff:     optional additional strict check, for bundles whose
                          sampling is deterministic (fanouts exceed every node
                          degree, so no truncation happens).
        raise_on_mismatch: raise (default) or just return the measurements.

    Returns:
        ``{'correlation': float, 'max_abs_diff': float}``.

    NOTE on sampling. With fanouts that truncate (rel-f1 uses [64, 32, 16] over
    high-degree tables) and ``temporal_strategy='uniform'``, neighbor sampling is
    STOCHASTIC: two runs of the same correct model differ. Measured on rel-f1,
    recomputed-vs-stored max|diff| is 0.60 while run-to-run max|diff| is 0.60 --
    indistinguishable. So an absolute-difference threshold would either reject a
    correct model or have to be loosened until it accepts a broken one.
    Correlation separates them cleanly: a correct model scores 0.9998, whereas
    the wrong ``aggr`` scores -0.62 and the wrong fanouts 0.96.

    NOTE on node indices. Prediction frames carry the entity key and a
    ``*_mapped`` column, but the relationship between them is NOT the same
    across datasets -- rel-f1 has ``driverId_mapped == driverId`` (offset 0)
    while the synthetic database has ``rid_mapped == rid + 1`` (offset -1).
    A wrong offset does not raise; it silently scores rows against other
    entities' predictions. Record the offset per bundle and let this function
    confirm it.

    NOTE on time. If the graph carries ``time``, predictions were produced with
    time-aware neighbor sampling, and recomputing without it silently gives
    different (future-leaking) subgraphs and different outputs. Pass the seed
    timestamps via ``time_col``.
    """
    from torch_geometric.loader import NeighborLoader
    from relbench.modeling.utils import to_unix_time

    graph_is_temporal = any("time" in data[nt] for nt in data.node_types)
    if graph_is_temporal and time_col is None:
        raise ValueError(
            "this graph has time attributes, so its predictions were produced "
            "with time-aware sampling; pass `time_col` (the seed-timestamp "
            "column of `reference`) or verification will not reproduce them."
        )

    idx = torch.as_tensor(
        reference[node_index_col].to_numpy() + node_index_offset,
        dtype=torch.long)
    loader_kwargs = dict(
        num_neighbors=num_neighbors, input_nodes=(entity_table, idx),
        batch_size=batch_size, shuffle=False,
    )
    if time_col is not None:
        loader_kwargs.update(
            time_attr="time",
            input_time=torch.from_numpy(to_unix_time(reference[time_col])),
            temporal_strategy=temporal_strategy,
        )
    loader = NeighborLoader(data, **loader_kwargs)
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            outs.append(model(batch.to(device), entity_table).view(-1).cpu())
    recomputed = torch.cat(outs).numpy()

    stored = reference[output_col].to_numpy()
    max_diff = float(np.abs(recomputed - stored).max())
    correlation = float(np.corrcoef(recomputed, stored)[0, 1])
    result = {"correlation": correlation, "max_abs_diff": max_diff}

    failures = []
    if correlation < min_correlation:
        failures.append(f"correlation {correlation:.5f} < {min_correlation}")
    if max_abs_diff is not None and max_diff > max_abs_diff:
        failures.append(f"max|diff| {max_diff:.4g} > {max_abs_diff:g}")

    if failures and raise_on_mismatch:
        raise ValueError(
            f"checkpoint verification FAILED: {'; '.join(failures)}.\n"
            f"The weights loaded, so this is almost certainly a "
            f"parameter-free architecture mismatch. Check, in order:\n"
            f"  - aggr: 'sum' vs 'mean' (count-based tasks need 'sum')\n"
            f"  - num_neighbors/fanouts: must match the trained model\n"
            f"  - encoder_layers: torch_frame ResNet depth (1 or 2)\n"
            f"  - the text embedder used to build the graph (GloVe 300-d vs "
            f"DistilBERT 768-d)\n"
            f"  - node_index_offset: 0 for rel-f1, -1 for the synthetic database\n"
            f"  - time_col / temporal_strategy for a temporal graph"
        )
    return result
