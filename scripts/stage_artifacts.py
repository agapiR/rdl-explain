"""Assemble the downloadable artifact bundles into ./artifacts, ready to zip.

Each bundle is self-contained: the trained model, its constructed graph, the
predictions the model produced (used to VERIFY the model was rebuilt correctly),
and a manifest.json recording the exact architecture. See
rdl_explain/artifacts.py for how they are consumed.

    python scripts/stage_artifacts.py [--src-root ...] [--out-dir artifacts]

Graphs are shipped rather than rebuilt. The distributed graphs were built by an
internal pipeline whose text features (256-d per column) are not reproduced by
the encoders available here, so a locally rebuilt graph would have different
feature dimensions and the published checkpoints would not load onto it.

Size: rel-f1 and synthetic are small, but each rel-trial graph is 16.6 GB (98%
of it dense text embeddings over 5.8M rows), so this writes ~50 GB and takes a
while. Scenarios sharing a database share one copy of the graph.
"""

import argparse
import json
import os
import shutil

# Architecture per bundle. These are read from each model's own training config;
# they are NOT interchangeable. `aggr` in particular varies by scenario: the
# count-based tasks need 'sum', and a wrong value loads silently and produces a
# broken model (see verify_checkpoint).
GNN_32_L3 = {"num_layers": 3, "channels": 32, "out_channels": 1,
             "encoder_layers": 2, "fanouts": [64, 32, 16]}

#: rel-trial scenarios. `graph` names the shared graph each one uses: the
#: sponsor-count scenario only rewrites task LABELS, leaving the database
#: untouched, so its graph is byte-identical to the original's and is shipped
#: once rather than twice (saving 16.6 GB).
TRIAL_SCENARIOS = [
    # (bundle name,                  source dir,                                    aggr,   graph,            description)
    ("study-outcome", "rel-trial-study-outcome", "mean", "base",
     "Original: predict whether a clinical trial achieves its primary outcome."),
    ("study-outcome-leakage-column", "rel-trial-data-leakage-column-study-outcome", "sum", "leakage-column",
     "Scenario 1: a preliminary_evaluation.rating column leaks the outcome."),
    ("study-outcome-leakage-tuple", "rel-trial-data-leakage-tuple-study-outcome", "sum", "leakage-tuple",
     "Scenario 2: only evaluations in categories A/B/C leak the outcome."),
    ("study-outcome-sponsor-count", "rel-trial-study-sponsor-count-outcome", "sum", "base",
     "Scenario 3: outcome depends purely on the number of linked sponsors."),
]

#: Which source directory each shared rel-trial graph is copied from.
TRIAL_GRAPHS = {
    "base": "rel-trial-study-outcome",
    "leakage-column": "rel-trial-data-leakage-column-study-outcome",
    "leakage-tuple": "rel-trial-data-leakage-tuple-study-outcome",
}


def copy(src, dst):
    if not os.path.exists(src):
        print(f"    MISSING {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_manifest(bundle_dir, manifest):
    with open(os.path.join(bundle_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def stage_rel_f1(src_root, out_dir):
    src = os.path.join(src_root, "relbench_models/rel-f1")
    dst = os.path.join(out_dir, "rel-f1/driver-dnf")
    mdl = os.path.join(src, "driver-dnf/explained_model")
    copy(f"{src}/data.pt", f"{dst}/data.pt")
    copy(f"{src}/col_stats_dict.pt", f"{dst}/col_stats_dict.pt")
    copy(f"{mdl}/model_state_dict.pth", f"{dst}/model_state_dict.pth")
    for split in ("train", "val", "test"):
        copy(f"{mdl}/inference_results/predictions_{split}.parquet",
             f"{dst}/inference_results/predictions_{split}.parquet")
    write_manifest(dst, {
        "name": "rel-f1/driver-dnf",
        "dataset": "rel-f1", "task": "driver-dnf",
        "description": "Paper task D1T1: will a driver fail to finish? "
                       "Test ROC-AUC 0.7593.",
        "graph": {"data": "data.pt", "col_stats": "col_stats_dict.pt"},
        "model": {"state_dict": "model_state_dict.pth", "aggr": "mean",
                  "norm": "batch_norm", **GNN_32_L3},
        "predictions": {"dir": "inference_results", "entity_table": "Drivers",
                        "node_index_col": "driverId",
                        "node_index_offset": 0, "time_col": "date",
                        "temporal_strategy": "uniform"},
        "task": {"entity_table": "Drivers", "entity_col": "driverId",
                 "time_col": "date", "target_col": "did_not_finish",
                 "task_type": "binary_classification",
                 "dataset_name": "rel-f1", "task_name": "driver-dnf"},
    })
    return dst


def stage_synthetic(src_root, out_dir):
    base = os.path.join(src_root, "eval_local/results_case_study")
    staged = []
    for task in ("count-1", "count-2"):
        src = f"{base}/data/r-s-synthetic-{task}"
        inf = f"{base}/inference/r-s-synthetic-{task}"
        dst = os.path.join(out_dir, f"synthetic/{task}")
        copy(f"{src}/data.pt", f"{dst}/data.pt")
        copy(f"{src}/col_stats_dict.pt", f"{dst}/col_stats_dict.pt")
        copy(f"{src}/model.pth", f"{dst}/model_state_dict.pth")
        for split in ("train", "val", "test"):
            copy(f"{inf}/predictions_{split}.parquet",
                 f"{dst}/inference_results/predictions_{split}.parquet")
        rule = ("|{s in S : s.rid = r.rid and s.X_Boolean}| >= 2"
                if task == "count-1" else "|{s in S : s.rid = r.rid}| >= 4")
        write_manifest(dst, {
            "name": f"synthetic/{task}",
            "dataset": "r-s-synthetic", "task": task,
            "description": f"Paper Example 1. Ground truth: label = {rule}.",
            "graph": {"data": "data.pt", "col_stats": "col_stats_dict.pt"},
            "model": {"state_dict": "model_state_dict.pth",
                      "num_layers": 1, "channels": 32, "out_channels": 1,
                      "aggr": "sum", "norm": "layer_norm",
                      "encoder_layers": 1, "fanouts": [64]},
            # Sampling is exhaustive here (fanout 64 exceeds every degree), so
            # verification can demand exact agreement, not just correlation.
            "predictions": {"dir": "inference_results", "entity_table": "R",
                            "node_index_col": "rid",
                            "node_index_offset": 0, "max_abs_diff": 1e-4},
            "task": {"entity_table": "R", "entity_col": "rid", "time_col": None,
                     "target_col": "label",
                     "task_type": "binary_classification",
                     "dataset_name": "r-s-synthetic", "task_name": task},
        })
        staged.append(dst)
    return staged


def stage_rel_trial(src_root, out_dir):
    base = os.path.join(src_root, "eval_local/results_case_study")
    staged = []

    # Shared graphs, copied once each (16.6 GB apiece).
    for graph_name, src_name in TRIAL_GRAPHS.items():
        dst = os.path.join(out_dir, "rel-trial/_graphs", graph_name)
        print(f"    graph {graph_name} (16.6 GB) ...", flush=True)
        copy(f"{base}/data/{src_name}/data.pt", f"{dst}/data.pt")
        copy(f"{base}/data/{src_name}/col_stats_dict.pt", f"{dst}/col_stats_dict.pt")

    for name, src_name, aggr, graph_name, description in TRIAL_SCENARIOS:
        src = f"{base}/data/{src_name}"
        inf = f"{base}/inference/{src_name}"
        dst = os.path.join(out_dir, f"rel-trial/{name}")
        copy(f"{src}/model.pth", f"{dst}/model_state_dict.pth")
        copy(f"{src}/col_stats_dict.pt", f"{dst}/col_stats_dict.pt")
        for split in ("train", "val", "test"):
            copy(f"{inf}/predictions_{split}.parquet",
                 f"{dst}/inference_results/predictions_{split}.parquet")
        write_manifest(dst, {
            "name": f"rel-trial/{name}",
            "dataset": "rel-trial", "task": "study-outcome",
            "description": description,
            # Shared with the other scenario that uses the same database.
            "graph": {"data": f"../_graphs/{graph_name}/data.pt",
                      "col_stats": f"../_graphs/{graph_name}/col_stats_dict.pt"},
            "model": {"state_dict": "model_state_dict.pth", "aggr": aggr,
                      "norm": "layer_norm", **GNN_32_L3},
            "predictions": {"dir": "inference_results",
                            "entity_table": "studies",
                            "node_index_col": "nct_id",
                            "node_index_offset": 0, "time_col": "timestamp",
                            "temporal_strategy": "uniform"},
            "task": {"entity_table": "studies", "entity_col": "nct_id",
                     "time_col": "timestamp", "target_col": "outcome",
                     "task_type": "binary_classification",
                     "dataset_name": "rel-trial", "task_name": "study-outcome"},
        })
        staged.append(dst)

    # The generated intervention tables, small enough to ship as-is.
    db = os.path.join(src_root, "relbench_cache_archive_20260516/rel-trial/db")
    dst = os.path.join(out_dir, "rel-trial/intervention-tables")
    for f in ("preliminary_evaluation_w_column_level_data_leakage.parquet",
              "preliminary_evaluation_w_tuple_level_data_leakage.parquet"):
        copy(os.path.join(db, f), os.path.join(dst, f))
    staged.append(dst)
    return staged


def main(src_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("staging rel-f1 ...");     stage_rel_f1(src_root, out_dir)
    print("staging synthetic ...");  stage_synthetic(src_root, out_dir)
    print("staging rel-trial ...");  stage_rel_trial(src_root, out_dir)

    print(f"\n{'bundle':44s} {'size':>10s}")
    print("-" * 56)
    total = 0
    for group in sorted(os.listdir(out_dir)):
        gpath = os.path.join(out_dir, group)
        if not os.path.isdir(gpath):
            continue
        for bundle in sorted(os.listdir(gpath)):
            bpath = os.path.join(gpath, bundle)
            size = sum(os.path.getsize(os.path.join(r, f))
                       for r, _, fs in os.walk(bpath) for f in fs)
            total += size
            print(f"{group + '/' + bundle:44s} {size / 1e6:9.1f} MB")
    print("-" * 56)
    print(f"{'TOTAL':44s} {total / 1e6:9.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src-root", default=os.environ.get("RDL_EXPLAIN_SRC_ROOT"),
                   help="root holding relbench_models/ and eval_local/ "
                        "(default: $RDL_EXPLAIN_SRC_ROOT)")
    p.add_argument("--out-dir", default="artifacts")
    a = p.parse_args()
    if not a.src_root:
        p.error("--src-root is required (or set $RDL_EXPLAIN_SRC_ROOT). This is "
                "the maintainer-side directory holding the trained models and "
                "graphs the bundles are assembled from.")
    main(a.src_root, a.out_dir)
