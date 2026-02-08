"""
Sigara bırakma olasılığı (QUITYRS > 0) modeli eğitimi.
NHIS missing kodları NaN'a çevrilir, impute edilir; sadece kalibre RF kaydedilir.
Artifact: model_quityrs_rf_calibrated.pkl
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE


# ============================== #
#  PATHS - FILE LOCATIONS
# ============================== #

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nhis_00003.csv"

MODEL_PATH = BASE_DIR / "model_quityrs_rf_calibrated.pkl"


# ============================== #
#  FEATURE DEFINITIONS
# ============================== #

FEATURES: List[str] = [
    "AGE",
    "GENDER",
    "EDUC",
    "ALCSTAT1",
    "ALCDAYSYR",
    "CIGSDAY",
    "CSQTRYYR",
    "CSWANTQUIT",
    "DEPRX",
    "DEPFEELEVL",
    "CIGSLONGFS",     # EKLENDİ
    "QUITNO"          # EKLENDİ
]

LABEL_COLS = ["QUITYRS", "CIGSLONGFS", "QUITNO"]

THRESHOLD = 0.9

CATEGORICAL = ["ALCSTAT1", "DEPRX", "GENDER"]

NUMERIC = [
    "AGE",
    "EDUC",
    "ALCDAYSYR",
    "CIGSDAY",
    "CSQTRYYR",
    "CSWANTQUIT",
    "DEPFEELELV",
    "CIGSLONGFS",
    "QUITNO"
]

NHIS_MISSING = {96,97,98,99,996,997,998,999,9996,9997,9998,9999}


# ============================== #
#  FUNCTIONS
# ============================== #

def replace_missing_codes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(list(NHIS_MISSING), np.nan)
    return df


def load_data(path: Path) -> pd.DataFrame:
    print(f"[INFO] Loading CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["QUITYRS"].notna()].copy()
    print(f"[INFO] Dropped rows with QUITYRS NaN: {before - len(df)}")
    df["target"] = (df["QUITYRS"] > 0).astype(int)
    print("[INFO] Target counts:\n", df["target"].value_counts())
    return df


def drop_never_smokers(df: pd.DataFrame) -> pd.DataFrame:
    mask_never = (
        (df["QUITYRS"] == 0)
        & (df["CIGSDAY"].fillna(0) <= 0)
        & (df["QUITNO"].fillna(0) <= 0)
    )
    before = len(df)
    df = df.loc[~mask_never].reset_index(drop=True)
    print(f"[INFO] Dropped never-smokers: {before - len(df)}")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [c for c in FEATURES if c != "GENDER"] + LABEL_COLS + ["target", "SEX"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[needed].copy()
    df["GENDER"] = df["SEX"]
    df = df.drop(columns=["SEX"])

    ordered = FEATURES + LABEL_COLS + ["target"]
    df = df[[c for c in ordered if c in df.columns]]
    return df


def impute_values(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    imputers = {"median": {}, "mode": {}}
    df = df.copy()

    for col in df.columns:
        if col in LABEL_COLS:
            df[col] = df[col].fillna(0)
            continue

        if df[col].dtype == object or df[col].nunique(dropna=True) <= 20:
            val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(val)
            imputers["mode"][col] = val
        else:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            imputers["median"][col] = med

    return df, imputers


def encode_categoricals(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders = {}
    for col in CATEGORICAL:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[ENCODER] {col} -> {list(le.classes_)}")
    return df, encoders


# ============================== #
#  TRAIN MODELS
# ============================== #

def train_rf_and_calibrated(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["target"].astype(int)

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # ---- RANDOM FOREST ----
    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n==== RF ====")
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred, digits=3))

    # ---- CALIBRATED RF ----
    cal = CalibratedClassifierCV(estimator=rf, method="sigmoid")

    cal.fit(X_train, y_train)

    y_proba_cal = cal.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_proba_cal >= 0.5).astype(int)

    print("\n==== RF + Calibration ====")
    print("ROC AUC:", roc_auc_score(y_test, y_proba_cal))
    print(classification_report(y_test, y_pred_cal, digits=3))

    return rf, cal


def save_model(obj: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] {path}")


# ============================== #
#  MAIN PIPELINE
# ============================== #

def main():
    df = load_data(DATA_PATH)
    df = replace_missing_codes(df, list(df.columns))
    df = build_target(df)
    df = drop_never_smokers(df)
    df = select_columns(df)

    df["CIGSDAY"] = pd.to_numeric(df["CIGSDAY"], errors="coerce").clip(0, 60)
    df["ALCDAYSYR"] = pd.to_numeric(df["ALCDAYSYR"], errors="coerce").clip(0, 365)
    df["CSQTRYYR"] = pd.to_numeric(df["CSQTRYYR"], errors="coerce").clip(0, 12)

    df, imputers = impute_values(df)
    df, encoders = encode_categoricals(df)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    print("[INFO] Dataset shape:", df[FEATURES].shape)

    _, cal = train_rf_and_calibrated(df)

    save_model({
        "model": cal,
        "feature_names": FEATURES,
        "threshold": THRESHOLD,
        "imputers": imputers,
        "encoders": encoders,
        "type": "rf_calibrated"
    }, MODEL_PATH)


if __name__ == "__main__":
    main()
