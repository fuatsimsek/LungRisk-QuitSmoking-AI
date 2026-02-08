import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict


BASE_DIR = os.path.dirname(__file__)


# Uygulama sadece kalibre RF modelini kullanıyor
QUIT_MODEL_PATH = os.path.join(BASE_DIR, "../ml2/model_quityrs_rf_calibrated.pkl")


FEATURES = [
    "AGE",
    "GENDER",
    "EDUC",
    "ALCSTAT1",
    "ALCDAYSYR",
    "CIGSDAY",
    "CSQTRYYR",
    "CSWANTQUIT",
    "DEPRX",
    "DEPFEELELVL",
    "CIGSLONGFS",
    "QUITNO",
]

LABEL_COLS = ["QUITYRS", "CIGSLONGFS", "QUITNO"]


def load_artifact(model_name: str = "rf_calibrated") -> Dict[str, Any]:
    """Tek desteklenen model: rf_calibrated (model_name yok sayılır)."""
    path = QUIT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def apply_impute(df: pd.DataFrame, imputers: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    for col, val in imputers.get("mode", {}).items():
        if col in df:
            df[col] = df[col].fillna(val)

    for col, med in imputers.get("median", {}).items():
        if col in df:
            df[col] = df[col].fillna(med)

    for col in LABEL_COLS:
        if col in df:
            df[col] = df[col].fillna(0)

    return df


def apply_encoders(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        if col in df:
            df[col] = enc.transform(df[col].astype(str))
    return df


def predict_model(input_records, model_name="rf_calibrated"):
    artifact = load_artifact()

    feature_names = artifact["feature_names"]
    model = artifact["model"]
    threshold = artifact.get("threshold", 0.5)
    imputers = artifact.get("imputers", {})
    encoders = artifact.get("encoders", {})

    if isinstance(input_records, dict):
        input_records = [input_records]

    df = pd.DataFrame(input_records)

    # Missing feature tamamlama
    for col in feature_names:
        if col not in df:
            df[col] = np.nan

    # Pipeline
    df = apply_impute(df, imputers)
    df = apply_encoders(df, encoders)

    X = df[feature_names]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    df["probability"] = proba
    df["prediction"] = pred

    return df.to_dict(orient="records")
