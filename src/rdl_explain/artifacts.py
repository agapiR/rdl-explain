"""Locating and loading the downloadable artifact bundles.

The paper's trained models and their constructed graphs are distributed as
bundles (see the README for the download links). Each bundle directory carries a
``manifest.json`` describing exactly how to rebuild the model:

.. code-block:: json

    {
      "name": "rel-f1/driver-dnf",
      "dataset": "rel-f1",
      "task": "driver-dnf",
      "graph": {"data": "data.pt", "col_stats": "col_stats_dict.pt"},
      "model": {
        "state_dict": "model_state_dict.pth",
        "num_layers": 3, "channels": 32, "out_channels": 1,
        "aggr": "mean", "norm": "batch_norm", "encoder_layers": 2,
        "fanouts": [64, 32, 16]
      },
      "predictions": {"dir": "inference_results", "entity_table": "Drivers"}
    }

WHY A MANIFEST. Several architecture settings are parameter-free -- the message
aggregation, the fanouts, the feature-encoder depth, the text embedder used to
build the graph. They cannot be recovered from the state dict, and a wrong guess
loads without error and silently produces a broken model (a count-based task
trained with ``aggr='sum'`` scores ROC-AUC 0.10 under ``'mean'``). The manifest
records them next to the weights, and ``load_bundle`` verifies the result
against the checkpoint's own stored predictions before handing it back.

Bundles are looked up under the root given by ``RDL_EXPLAIN_DATA`` (default:
``./artifacts``), so downloading and unzipping in place is all that is required.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch

from rdl_explain.loaders import RunConfig, verify_checkpoint
from rdl_explain.model import Model

#: Environment variable naming the artifact root.
DATA_ROOT_ENV = "RDL_EXPLAIN_DATA"
DEFAULT_DATA_ROOT = "./artifacts"

MANIFEST_NAME = "manifest.json"


def data_root() -> str:
    """Root directory holding the downloaded artifact bundles."""
    return os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)


def bundle_path(name: str) -> str:
    """Absolute path of a bundle, e.g. ``bundle_path('rel-f1/driver-dnf')``."""
    return os.path.abspath(os.path.join(data_root(), name))


def read_manifest(name: str) -> Dict[str, Any]:
    """Read a bundle's ``manifest.json``, with an actionable error if absent."""
    path = os.path.join(bundle_path(name), MANIFEST_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} at {path}.\n"
            f"Artifacts are looked up under {data_root()!r} "
            f"(set ${DATA_ROOT_ENV} to change this). Download the bundle from "
            f"the link in the README and unzip it there."
        )
    with open(path) as f:
        return json.load(f)


def load_bundle(
    name: str,
    device: str = "cpu",
    verify: bool = True,
    verify_split: str = "test",
) -> Tuple[Model, Any, Dict, RunConfig, Dict[str, Any]]:
    """Load a bundle's graph and trained model, verified against its predictions.

    Args:
        name:         bundle name relative to the artifact root.
        device:       where to put the model.
        verify:       re-run inference and check it reproduces the stored
                      predictions. Leave this on; it is the only check that
                      catches parameter-free architecture mismatches.
        verify_split: which prediction split to verify against.

    Returns:
        ``(model, data, col_stats_dict, config, manifest)``.
    """
    manifest = read_manifest(name)
    root = bundle_path(name)
    g, m = manifest["graph"], manifest["model"]

    data = torch.load(os.path.join(root, g["data"]), map_location="cpu",
                      weights_only=False)
    col_stats_dict = torch.load(os.path.join(root, g["col_stats"]),
                                map_location="cpu", weights_only=False)

    config = RunConfig(
        device=device,
        gnn_layers=m["num_layers"],
        num_neighbors=list(m["fanouts"]),
        channels=m["channels"],
        out_channels=m.get("out_channels", 1),
        aggr=m["aggr"],
        norm=m.get("norm", "batch_norm"),
        encoder_layers=m.get("encoder_layers", 2),
    )

    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=config.gnn_layers,
        channels=config.channels,
        out_channels=config.out_channels,
        aggr=config.aggr,
        norm=config.norm,
        encoder_layers=config.encoder_layers,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(root, m["state_dict"]), map_location=device,
                   weights_only=False)
    )
    model.eval()

    if verify:
        pred = manifest["predictions"]
        reference = pd.read_parquet(
            os.path.join(root, pred["dir"], f"predictions_{verify_split}.parquet"))
        result = verify_checkpoint(
            model=model, data=data, entity_table=pred["entity_table"],
            reference=reference, num_neighbors=config.num_neighbors,
            device=device,
            node_index_col=pred["node_index_col"],
            node_index_offset=pred.get("node_index_offset", 0),
            time_col=pred.get("time_col"),
            temporal_strategy=pred.get("temporal_strategy", "uniform"),
            min_correlation=pred.get("min_correlation", 0.99),
            max_abs_diff=pred.get("max_abs_diff"),
        )
        print(f"[{name}] checkpoint verified against {verify_split} predictions "
              f"(corr = {result['correlation']:.5f}, "
              f"max|diff| = {result['max_abs_diff']:.2e})")

    return model, data, col_stats_dict, config, manifest


def predictions_dir(name: str) -> str:
    """Directory of a bundle's stored predictions (for make_explanation_task)."""
    return os.path.join(bundle_path(name), read_manifest(name)["predictions"]["dir"])
