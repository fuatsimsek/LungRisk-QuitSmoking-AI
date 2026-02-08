"""
Training script: clean_lung_cancer.csv -> lung_cancer_model.pkl
- Uses specified feature subset
- Stratified 70/30 split
- Preprocess: age -> StandardScaler, other features -> MinMaxScaler
- Classifier: multinomial LogisticRegression (balanced)
- Reports hold-out metrics (macro) and 5-fold CV
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "clean_lung_cancer.csv"
MODEL_PATH = BASE_DIR / "lung_cancer_model.pkl"
TARGET_COL = "level"

FEATURE_COLUMNS: List[str] = [
    "age",
    "gender",
    "air_pollution",
    "dust_allergy",
    "occupational_hazards",
    "genetic_risk",
    "wheezing",
    "fatigue",
    "alcohol_use",
    "chronic_lung_disease",
    "smoking",
    "passive_smoker",
]

LABEL_NAMES = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned dataset and return X, y."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python merge_and_clean.py` first.")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    """Create preprocessing + classifier pipeline."""
    preprocess = ColumnTransformer(
        [
            ("age_scaler", StandardScaler(), ["age"]),
            ("other_scaler", MinMaxScaler(), [c for c in FEATURE_COLUMNS if c != "age"]),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        C=0.8,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro metrics for multiclass."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=sorted(LABEL_NAMES.keys()))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)


def run_cv(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline) -> None:
    """5-fold Stratified CV with accuracy and macro F1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = Pipeline(pipeline.steps)  # clone-like
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], preds)
        f1 = f1_score(y.iloc[val_idx], preds, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"[CV{fold}] accuracy={acc:.4f} | f1_macro={f1:.4f}")
    print(f"[CV-AVG] accuracy={np.mean(accs):.4f} | f1_macro={np.mean(f1s):.4f}")


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    preds = pipeline.predict(X_test)
    metrics = evaluate(y_test, preds)
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.4f}")
    print_confusion(y_test, preds)

    print("[INFO] 5-fold Stratified CV...")
    run_cv(X, y, pipeline)

    artifact = {
        "model": pipeline,
        "feature_names": FEATURE_COLUMNS,
        "label_names": LABEL_NAMES,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

