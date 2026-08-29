"""Generate the three interventions on RelBench's rel-trial used in the case study.

The paper's case study (Section 6.3) takes the real `rel-trial` database and the
`study-outcome` task and plants three known problems in it, so that what the
explanation *should* find is known in advance:

  1. column-level leakage  -- a `preliminary_evaluation(reviewer_id, nct_id,
     rating)` table where `rating` correlates with the outcome. Every study gets
     3 reviews. The `Projection` explanation should flag `rating`.

  2. tuple-level leakage   -- the same table plus an `eval_category` column with
     10 values; `rating` correlates with the outcome ONLY for categories A, B
     and C, and is uniform noise elsewhere. 3 reviews per category per study.
     `Projection` flags `rating` AND `eval_category`; `Selection` is then needed
     to say WHICH categories leak.

  3. structural task       -- `study-sponsor-count-outcome`, where the label is
     simply whether a study has more than one sponsor (10% label noise). No
     attribute carries the signal, so `Projection` should come up empty while
     `FKJoin` identifies the sponsors_studies JOIN as important.

REPRODUCIBILITY. The three interventions share one random stream: the seed is set once, 
then they are generated in the fixed order above. 
This reproduces the distributed tables exactly (verified: ratings
match element-for-element, and the sponsor-count labels match on all three
splits). Re-seeding per intervention, or generating only one of them, yields
valid but different data that no longer matches the published checkpoints.

Usage:
    python scripts/generate_clinical_trial_interventions.py \\
        --rel-trial-dir /path/to/rel-trial --out-dir data/rel-trial-interventions

    # check the output against the distributed tables
    python scripts/generate_clinical_trial_interventions.py \\
        --rel-trial-dir /path/to/rel-trial --out-dir /tmp/check \\
        --verify-against /path/to/rel-trial
"""

import argparse
import os
import random

import pandas as pd

#: Rating distributions. A positive outcome skews high, a negative one low --
#: this correlation IS the planted leakage.
WEIGHTS_POSITIVE = [1, 1, 1, 3, 4]
WEIGHTS_NEGATIVE = [4, 3, 1, 1, 1]
RATINGS = [1, 2, 3, 4, 5]

REVIEWS_PER_STUDY = 3
EVAL_CATEGORIES = list("ABCDEFGHIJ")
LEAKING_CATEGORIES = {"A", "B", "C"}
LABEL_NOISE = 0.10

COLUMN_LEVEL_FILE = "preliminary_evaluation_w_column_level_data_leakage.parquet"
TUPLE_LEVEL_FILE = "preliminary_evaluation_w_tuple_level_data_leakage.parquet"
SPONSOR_TASK_DIR = "study-sponsor-count-outcome"


def _rating_for(outcome) -> int:
    weights = WEIGHTS_POSITIVE if outcome == 1 else WEIGHTS_NEGATIVE
    return random.choices(RATINGS, weights=weights)[0]


def make_column_level_leakage(outcome_df: pd.DataFrame) -> pd.DataFrame:
    """Intervention 1: every review's rating correlates with the outcome."""
    rows, reviewer_id = [], 0
    for _, study in outcome_df.iterrows():
        for _ in range(REVIEWS_PER_STUDY):
            rows.append({"reviewer_id": reviewer_id,
                         "nct_id": study["nct_id"],
                         "rating": _rating_for(study["outcome"])})
            reviewer_id += 1
    return pd.DataFrame(rows)


def make_tuple_level_leakage(outcome_df: pd.DataFrame) -> pd.DataFrame:
    """Intervention 2: only categories A/B/C leak; the rest are uniform noise."""
    rows, reviewer_id = [], 0
    for _, study in outcome_df.iterrows():
        for category in EVAL_CATEGORIES:
            for _ in range(REVIEWS_PER_STUDY):
                rating = (_rating_for(study["outcome"])
                          if category in LEAKING_CATEGORIES
                          else random.randint(1, 5))
                rows.append({"reviewer_id": reviewer_id,
                             "nct_id": study["nct_id"],
                             "eval_category": category,
                             "rating": rating})
                reviewer_id += 1
    return pd.DataFrame(rows)


def make_sponsor_count_task(outcome_with_dates: pd.DataFrame,
                            sponsors_studies: pd.DataFrame) -> pd.DataFrame:
    """Intervention 3: label = (study has >1 sponsor), with 10% flipped."""
    sponsor_count = sponsors_studies.groupby("nct_id").size()
    df = outcome_with_dates.copy()
    df["outcome"] = df["nct_id"].map(
        lambda nct_id: 1 if sponsor_count.get(nct_id, 0) > 1 else 0)
    df["outcome"] = df.apply(
        lambda row: 1 - row["outcome"] if random.random() < LABEL_NOISE
        else row["outcome"], axis=1)
    return df


def load_source(rel_trial_dir: str):
    splits = {s: pd.read_parquet(
        os.path.join(rel_trial_dir, "tasks", "study-outcome", f"{s}.parquet"))
        for s in ("train", "val", "test")}
    outcome_with_dates = pd.concat(splits.values(), ignore_index=True)
    sponsors = pd.read_parquet(
        os.path.join(rel_trial_dir, "db", "sponsors_studies.parquet"))
    return splits, outcome_with_dates, sponsors


def verify(generated: dict, reference_dir: str) -> bool:
    """Compare generated tables against the distributed ones, element-wise."""
    ok = True
    checks = [
        (COLUMN_LEVEL_FILE, generated["column_level"],
         os.path.join(reference_dir, "db", COLUMN_LEVEL_FILE), "rating"),
        (TUPLE_LEVEL_FILE, generated["tuple_level"],
         os.path.join(reference_dir, "db", TUPLE_LEVEL_FILE), "rating"),
    ]
    for name, got, path, col in checks:
        if not os.path.exists(path):
            print(f"  {name}: reference not found, skipped"); continue
        ref = pd.read_parquet(path)
        match = (len(got) == len(ref)
                 and bool((got[col].values == ref[col].values).all()))
        ok &= match
        print(f"  {name}: {'MATCH' if match else 'DIFFERS'} "
              f"({len(got)} rows vs {len(ref)})")

    ref_dir = os.path.join(reference_dir, "tasks", SPONSOR_TASK_DIR)
    for split, part in generated["sponsor_task"].items():
        path = os.path.join(ref_dir, f"{split}.parquet")
        if not os.path.exists(path):
            print(f"  {SPONSOR_TASK_DIR}/{split}: reference not found, skipped")
            continue
        ref = pd.read_parquet(path)
        aligned = part.set_index("nct_id").loc[ref["nct_id"], "outcome"].values
        match = bool((aligned == ref["outcome"].values).all())
        ok &= match
        print(f"  {SPONSOR_TASK_DIR}/{split}: {'MATCH' if match else 'DIFFERS'}")
    return ok


def main(rel_trial_dir: str, out_dir: str, seed: int, verify_against: str):
    splits, outcome_with_dates, sponsors = load_source(rel_trial_dir)
    outcome_df = outcome_with_dates[["nct_id", "outcome"]]
    print(f"source: {len(outcome_df)} studies from {rel_trial_dir}")

    # ONE seed, then all three in this order -- see the module docstring.
    random.seed(seed)
    column_level = make_column_level_leakage(outcome_df)
    tuple_level = make_tuple_level_leakage(outcome_df)
    sponsor_all = make_sponsor_count_task(outcome_with_dates, sponsors)

    sponsor_task = {
        split: sponsor_all[sponsor_all["nct_id"].isin(part["nct_id"])]
        for split, part in splits.items()
    }
    generated = {"column_level": column_level, "tuple_level": tuple_level,
                 "sponsor_task": sponsor_task}

    db_dir = os.path.join(out_dir, "db")
    os.makedirs(db_dir, exist_ok=True)
    column_level.to_parquet(os.path.join(db_dir, COLUMN_LEVEL_FILE), index=False)
    tuple_level.to_parquet(os.path.join(db_dir, TUPLE_LEVEL_FILE), index=False)
    print(f"  intervention 1 (column-level leakage): {len(column_level)} reviews")
    print(f"  intervention 2 (tuple-level leakage):  {len(tuple_level)} reviews")

    task_dir = os.path.join(out_dir, "tasks", SPONSOR_TASK_DIR)
    os.makedirs(task_dir, exist_ok=True)
    for split, part in sponsor_task.items():
        part.to_parquet(os.path.join(task_dir, f"{split}.parquet"), index=False)
    positives = float(sponsor_all["outcome"].mean())
    print(f"  intervention 3 (sponsor-count task):   {len(sponsor_all)} studies, "
          f"{positives:.1%} positive")
    print(f"written to {out_dir}")

    if verify_against:
        print(f"\nverifying against {verify_against}")
        print("  ALL MATCH" if verify(generated, verify_against)
              else "  MISMATCH -- output differs from the distributed tables")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rel-trial-dir", required=True,
                   help="rel-trial root containing db/ and tasks/")
    p.add_argument("--out-dir", default="data/rel-trial-interventions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-against", default=None,
                   help="rel-trial root holding the distributed tables, to "
                        "check the generated output against")
    a = p.parse_args()
    main(a.rel_trial_dir, a.out_dir, a.seed, a.verify_against)
