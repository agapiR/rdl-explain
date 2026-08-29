"""Generate the synthetic R-S-T database and its two count-based tasks.

This is the toy database of Example 1 in the paper: a relation R of entities, a
relation S linked to R by a foreign key and carrying one informative boolean
attribute, and a relation T also linked to R but carrying nothing informative.
Every other attribute in all three tables is standard normal noise.

Two tasks, both defined purely by counting over the R-S join:

    count-1   label = |{s in S : s.rid = r.rid and s.X_Boolean}| >= 2
    count-2   label = |{s in S : s.rid = r.rid}|                >= 4

The point of the toy database is that the correct explanation is known in
advance, which is what makes it useful for checking the method:

  * count-1 depends on the R-S join AND on the single attribute S.X_Boolean.
    A correct Projection explanation keeps S.X_Boolean and drops the other 20
    columns; a correct FKJoin explanation keeps R-S and drops R-T.
  * count-2 depends on the R-S join ONLY -- it counts rows regardless of their
    values. So no attribute should be important, which is a case feature
    attribution alone cannot express (paper, Example 1).

Usage:
    python scripts/generate_synthetic_rst.py --out-dir data/r-s-synthetic

Reproducibility. At the default seed this regenerates the EXACT database the
`synthetic/*` artifact bundles were trained on -- the three tables come back 
identical, and the R features match those in the bundled graph. So
the toy case study can be reproduced from source.
Change `--seed` and you get a different database, which no longer corresponds to
the bundled checkpoints.
"""

import argparse
import os

import numpy as np
import pandas as pd

# Table sizes and noise-column counts, as used for the paper's toy example.
NUM_R, NUM_S, NUM_T = 1000, 4000, 5000
N_NOISE_R, N_NOISE_S, N_NOISE_T = 10, 5, 5
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / val / test


def generate_database(seed: int = 42):
    """Build the three tables. Only S.X_Boolean carries signal."""
    rng = np.random.RandomState(seed)

    R = pd.DataFrame({
        "rid": range(1, NUM_R + 1),
        **{f"x{i + 1}": rng.randn(NUM_R) for i in range(N_NOISE_R)},
    })
    S = pd.DataFrame({
        "sid": range(1, NUM_S + 1),
        "rid": rng.choice(R["rid"], NUM_S),
        **{f"x{i + 1}": rng.randn(NUM_S) for i in range(N_NOISE_S)},
        "X_Boolean": rng.choice([1, 0], NUM_S),
    })
    T = pd.DataFrame({
        "tid": range(1, NUM_T + 1),
        "rid": rng.choice(R["rid"], NUM_T),
        **{f"x{i + 1}": rng.randn(NUM_T) for i in range(N_NOISE_T)},
    })
    return R, S, T


def make_tasks(R: pd.DataFrame, S: pd.DataFrame):
    """count-1 needs the join AND S.X_Boolean; count-2 needs only the join."""
    count_true = (S[S["X_Boolean"] == 1].groupby("rid")["sid"].count()
                  .reindex(R["rid"], fill_value=0))
    count_all = (S.groupby("rid")["sid"].count()
                 .reindex(R["rid"], fill_value=0))
    return {
        "count-1": pd.DataFrame({"rid": R["rid"],
                                 "label": count_true.values >= 2}),
        "count-2": pd.DataFrame({"rid": R["rid"],
                                 "label": count_all.values >= 4}),
    }


def split(df: pd.DataFrame, seed: int):
    """Shuffle into train/val/test."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(SPLIT_FRACTIONS[0] * len(df))
    n_val = int(SPLIT_FRACTIONS[1] * len(df))
    return {"train": df.iloc[:n_train],
            "val": df.iloc[n_train:n_train + n_val],
            "test": df.iloc[n_train + n_val:]}


def main(out_dir: str, seed: int) -> None:
    R, S, T = generate_database(seed)
    tasks = make_tasks(R, S)

    db_dir = os.path.join(out_dir, "db")
    os.makedirs(db_dir, exist_ok=True)
    for name, table in (("R", R), ("S", S), ("T", T)):
        table.to_parquet(os.path.join(db_dir, f"{name}.parquet"), index=False)
    print(f"database -> {db_dir}")
    print(f"  R {R.shape}  S {S.shape}  T {T.shape}")

    for task_name, table in tasks.items():
        task_dir = os.path.join(out_dir, "tasks", task_name)
        os.makedirs(task_dir, exist_ok=True)
        for split_name, part in split(table, seed).items():
            part.to_parquet(os.path.join(task_dir, f"{split_name}.parquet"),
                            index=False)
        positives = float(table["label"].mean())
        print(f"task {task_name} -> {task_dir}  "
              f"({positives:.1%} positive, {len(table)} instances)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="data/r-s-synthetic",
                        help="where to write db/ and tasks/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.out_dir, args.seed)
