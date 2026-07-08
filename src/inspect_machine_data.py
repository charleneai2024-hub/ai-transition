"""Quick inspection report for the AI4I 2020 predictive maintenance dataset.

The raw CSV lives under data/ and is gitignored (see data/*.csv) — this
script is the reproducible record of what the data looks like.
"""

import pandas as pd

DATA_PATH = "data/Predictive Maintenance Dataset (AI4I 2020).csv"
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("=== Shape ===")
    print(df.shape)
    print()

    print("=== Column dtypes ===")
    print(df.dtypes)
    print()

    print("=== Missing values per column ===")
    print(df.isna().sum())
    print()

    print(f"=== {TARGET_COLUMN} distribution (positive/negative ratio) ===")
    counts = df[TARGET_COLUMN].value_counts()
    pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "pct": pct.round(2)}))
    print()

    # The 5 failure-mode flags are documented sub-causes of Machine failure;
    # check whether the aggregate flag and the sub-flags ever disagree.
    any_mode_flagged = df[FAILURE_MODE_COLUMNS].sum(axis=1) > 0
    mismatch_failure_no_mode = ((df[TARGET_COLUMN] == 1) & (~any_mode_flagged)).sum()
    mismatch_mode_no_failure = ((df[TARGET_COLUMN] == 0) & any_mode_flagged).sum()

    print("=== Consistency: Machine failure vs individual failure-mode flags ===")
    print(f"Machine failure=1 but no mode flagged: {mismatch_failure_no_mode}")
    print(f"Machine failure=0 but a mode IS flagged: {mismatch_mode_no_failure}")
    print()

    print("=== Numeric feature ranges ===")
    print(df[NUMERIC_COLUMNS].describe().T[["min", "max", "mean"]])


if __name__ == "__main__":
    main()
