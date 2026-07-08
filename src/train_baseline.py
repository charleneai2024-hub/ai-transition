"""Baseline model comparison for machine failure prediction.

Compares a dummy (always-predict-no-failure) baseline against Logistic
Regression and Random Forest, both weighted for the severe class imbalance,
to show why plain accuracy is a misleading metric on this dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/machine_model_ready.parquet"
FIGURES_DIR = Path("figures")
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Same categorical blue used across the project's plots (single-series magnitude).
BAR_COLOR = "#2a78d6"


def load_split():
    """Load the modeling-ready data and do a stratified 80/20 train/test split."""
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_models() -> dict:
    """Return the 3 estimators to compare: dummy baseline, logistic regression, random forest."""
    return {
        "Dummy (always predict no failure)": DummyClassifier(strategy="constant", constant=0),
        # Logistic regression is scale-sensitive, so it's wrapped with a scaler.
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        # Tree-based models split on raw thresholds, so no scaling is needed.
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    }


def evaluate(name: str, model, X_test, y_test) -> None:
    """Print accuracy, precision/recall/F1 for the failure class, and the confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"=== {name} ===")
    print(f"accuracy={accuracy:.4f}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print("confusion matrix (rows=actual, cols=predicted, order=[no failure, failure]):")
    print(cm)
    print()


def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> None:
    """Save a horizontal bar chart of Random Forest feature importances, most important first."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

    print("=== Random Forest feature importance (most important first) ===")
    print(importances.sort_values(ascending=False))
    print()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importances.index, importances.values, color=BAR_COLOR)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split()

    models = build_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate(name, model, X_test, y_test)

    plot_feature_importance(models["Random Forest"], list(X_train.columns))
    print(f"Saved feature importance plot -> {FIGURES_DIR}/feature_importance.png")


if __name__ == "__main__":
    main()
