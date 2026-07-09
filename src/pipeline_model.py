"""Cross-validated, leak-safe model selection for machine failure prediction.

Wraps preprocessing + classifier into a single sklearn Pipeline so that
scaling statistics are always fit on training folds only, compares Logistic
Regression vs Random Forest with 5-fold stratified CV (not a single
train/test split), evaluates the better one on a held-out test set, and
persists the fitted pipeline with joblib.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/machine_model_ready.parquet"
MODELS_DIR = Path("models")
TARGET_COLUMN = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power",
]
# Type is already ordinally encoded (L=0/M=1/H=2) — pass it through unscaled.
PASSTHROUGH_FEATURES = ["Type"]


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features; pass the already-ordinal-encoded Type column through."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("type", "passthrough", PASSTHROUGH_FEATURES),
        ]
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Return the two candidate pipelines: preprocessing + classifier bundled together."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def cross_validate_pipelines(pipelines: dict[str, Pipeline], X_train, y_train) -> dict[str, float]:
    """Report mean +/- std F1 (failure class) for each pipeline via 5-fold stratified CV."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mean_scores = {}

    print("=== Cross-validated F1 (failure class), 5-fold StratifiedKFold ===")
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        mean_scores[name] = scores.mean()
        print(f"{name}: mean={scores.mean():.4f}  std={scores.std():.4f}  folds={scores.round(4).tolist()}")
    print()

    return mean_scores


def evaluate_on_test(name: str, pipeline: Pipeline, X_test, y_test) -> None:
    """Print precision/recall/F1 for the failure class and the confusion matrix."""
    y_pred = pipeline.predict(X_test)

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"=== Final held-out test evaluation: {name} ===")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print("confusion matrix (rows=actual, cols=predicted, order=[no failure, failure]):")
    print(cm)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipelines = build_pipelines()
    mean_scores = cross_validate_pipelines(pipelines, X_train, y_train)

    best_name = max(mean_scores, key=mean_scores.get)
    best_pipeline = pipelines[best_name]
    print(f"Best pipeline by mean CV F1: {best_name}\n")

    # Refit the chosen pipeline on the full training set, then evaluate once
    # on the held-out test set — the test set was never touched during CV.
    best_pipeline.fit(X_train, y_train)
    evaluate_on_test(best_name, best_pipeline, X_test, y_test)

    output_path = MODELS_DIR / "best_pipeline.joblib"
    joblib.dump(best_pipeline, output_path)
    print(f"\nSaved fitted pipeline ({best_name}) -> {output_path}")


if __name__ == "__main__":
    main()
