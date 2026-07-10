"""
Assemble a GIFT-Eval leaderboard submission from per-dataset results.

Concatenates the per-config ``results/<alias>/<dataset_config>/results.csv``
files written by ``src.data.evaluator.Evaluator`` into the two files the
official leaderboard repo (SalesforceAIResearch/gift-eval) expects under
``results/<submission_name>/``:

- ``all_results.csv`` — one row per dataset/term config (97 total), same
  column schema the Evaluator already writes.
- ``config.json`` — submission metadata.

Usage (from the repo root):
    python -m pipeline.assemble_submission                # defaults for Toto2
    python -m pipeline.assemble_submission --alias Toto2 \
        --submission-name Toto_2_0_2_5B_FT
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]


def expected_num_configs() -> int:
    """Number of dataset/term configs in the eval sweep (conf/data/dataset.yaml)."""
    with open(ROOT_DIR / "conf" / "data" / "dataset.yaml") as f:
        return len(yaml.safe_load(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alias",
        default="Toto2",
        help="Forecaster alias whose results/<alias>/ tree to assemble.",
    )
    parser.add_argument(
        "--submission-name",
        default="Toto_2_0_2_5B_FT",
        help="Folder name for the submission (results/all_results/<name>/).",
    )
    parser.add_argument(
        "--model-name",
        default="Toto-2.0-2.5B-FT",
        help="Value written to the `model` column of all_results.csv.",
    )
    args = parser.parse_args()

    input_path = ROOT_DIR / "results" / args.alias
    if not input_path.is_dir():
        sys.exit(f"No results directory at {input_path} — run the eval sweep first.")

    csv_files = sorted(input_path.rglob("results.csv"))
    if not csv_files:
        sys.exit(f"No results.csv files found under {input_path}.")

    # Evaluator appends a row per run to each per-config results.csv; keep the
    # most recent row (tail) per config, same as notebooks/results.ipynb.
    df = pd.concat(
        (pd.read_csv(f).tail(1) for f in csv_files),
        ignore_index=True,
    )

    # --- Sanity checks --------------------------------------------------
    problems: list[str] = []

    n_expected = expected_num_configs()
    if len(df) != n_expected:
        problems.append(
            f"expected {n_expected} dataset configs, found {len(df)} "
            f"(missing: incomplete sweep?)"
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append(
            f"column mismatch:\n  got:      {list(df.columns)}\n"
            f"  expected: {EXPECTED_COLUMNS}"
        )

    metric_cols = [c for c in df.columns if c.startswith("eval_metrics/")]
    bad = df[df[metric_cols].isna().any(axis=1)]["dataset"].tolist()
    if bad:
        problems.append(f"NaN metrics in configs: {bad}")

    dupes = df[df["dataset"].duplicated()]["dataset"].tolist()
    if dupes:
        problems.append(f"duplicate dataset configs: {dupes}")

    if problems:
        print("WARNING — submission has issues:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)

    # --- Write submission -----------------------------------------------
    df["model"] = args.model_name
    df = df.sort_values("dataset").reset_index(drop=True)

    output_path = ROOT_DIR / "results" / "all_results" / args.submission_name
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "all_results.csv", index=False)

    config = {
        "model": args.model_name,
        # Datadog's model card describes Toto-2.0-2.5B-FT as a fine-tuned,
        # benchmarking-oriented checkpoint. Review before opening the PR.
        "model_type": "pretrained",
        "model_dtype": "bfloat16",
        "model_link": "https://huggingface.co/Datadog/Toto-2.0-2.5B-FT",
        "code_link": "https://github.com/DataDog/toto",
        "org": "Datadog",
        "testdata_leakage": "No",
        "replication_code_available": "Yes",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print(f"Wrote {len(df)} rows to {output_path / 'all_results.csv'}")
    print(f"Wrote {output_path / 'config.json'}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
