"""EDA + feature engineering for the AI4I 2020 predictive maintenance dataset.

Produces comparison plots (failure vs no-failure) for the 5 numeric sensor
features, a correlation heatmap, a failure-rate breakdown by machine Type,
and a modeling-ready dataset with two domain-driven engineered features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DATA_PATH = "data/machine_failure.csv"
OUTPUT_PATH = "data/machine_model_ready.parquet"
FIGURES_DIR = Path("figures")

TARGET_COLUMN = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
# Columns that describe the outcome (or an ID with no predictive value) —
# keeping them would leak the label straight into the feature set.
LEAKY_OR_ID_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
TYPE_ORDER = ["L", "M", "H"]

# Two-class comparison palette (categorical slots 1 and 6: blue vs red).
COLOR_NO_FAILURE = "#2a78d6"
COLOR_FAILURE = "#e34948"
FAILURE_PALETTE = {0: COLOR_NO_FAILURE, 1: COLOR_FAILURE}

# Diverging colormap for correlation (blue <-> red, neutral gray midpoint).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"]
)

# Light, recessive chart chrome shared by all figures in this script.
plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "grid.color": "#e1e0d9",
        "axes.grid": True,
        "grid.linewidth": 0.6,
    }
)


def slugify(feature_name: str) -> str:
    """Turn a feature name like 'Air temperature [K]' into 'air_temperature_k'."""
    cleaned = feature_name.lower().replace("[", "").replace("]", "")
    return "_".join(cleaned.split())


def plot_feature_by_failure(df: pd.DataFrame, feature: str) -> None:
    """Save a boxplot comparing one numeric feature across failure vs no-failure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y=feature,
        hue=TARGET_COLUMN,
        palette=FAILURE_PALETTE,
        legend=False,
        ax=ax,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No failure", "Failure"])
    ax.set_title(f"{feature} by failure status")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"dist_{slugify(feature)}.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a correlation heatmap over the numeric features + target."""
    corr = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="#fcfcfb",
        ax=ax,
    )
    ax.set_title("Correlation heatmap (numeric features + target)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def cohens_d(df: pd.DataFrame, feature: str) -> float:
    """Standardized mean difference between the failure and no-failure groups."""
    group_failure = df.loc[df[TARGET_COLUMN] == 1, feature]
    group_ok = df.loc[df[TARGET_COLUMN] == 0, feature]

    n1, n0 = len(group_failure), len(group_ok)
    pooled_std = np.sqrt(
        ((n1 - 1) * group_failure.std() ** 2 + (n0 - 1) * group_ok.std() ** 2) / (n1 + n0 - 2)
    )
    return (group_failure.mean() - group_ok.mean()) / pooled_std


def report_effect_sizes(df: pd.DataFrame) -> None:
    """Print each numeric feature's separation between failure and no-failure, ranked."""
    effect_sizes = {feature: cohens_d(df, feature) for feature in NUMERIC_FEATURES}
    ranked = sorted(effect_sizes.items(), key=lambda item: abs(item[1]), reverse=True)

    print("=== Effect size (Cohen's d) of failure vs no-failure, ranked ===")
    for feature, d in ranked:
        print(f"{feature}: {d:.3f}")


def report_failure_rate_by_type(df: pd.DataFrame) -> None:
    """Print the failure rate for each machine Type (L/M/H)."""
    rate = df.groupby("Type")[TARGET_COLUMN].mean().reindex(TYPE_ORDER)
    print("=== Failure rate by Type ===")
    print((rate * 100).round(2).astype(str) + "%")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features, encode Type ordinally, and drop leaky/ID columns."""
    df = df.copy()

    # Large gap between process and air temperature can indicate abnormal cooling.
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Torque * rotational speed approximates mechanical load/power on the tool.
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    # Type is ordered by machine tier (low -> mid -> high), not a plain category.
    type_rank = {"L": 0, "M": 1, "H": 2}
    df["Type"] = df["Type"].map(type_rank)

    return df.drop(columns=LEAKY_OR_ID_COLUMNS)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    for feature in NUMERIC_FEATURES:
        plot_feature_by_failure(df, feature)
    plot_correlation_heatmap(df)
    print(f"Saved {len(NUMERIC_FEATURES)} distribution plots + 1 heatmap -> {FIGURES_DIR}/")
    print()

    report_failure_rate_by_type(df)
    print()
    report_effect_sizes(df)
    print()

    model_ready = engineer_features(df)
    model_ready.to_parquet(OUTPUT_PATH)
    print(f"Saved modeling-ready data ({model_ready.shape[0]} rows, {model_ready.shape[1]} cols) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
