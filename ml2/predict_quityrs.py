"""
Kaydedilmiş QUITYRS modelinden tahmin alma.
Kullanılan model: model_quityrs_rf_calibrated.pkl

Örnek kullanım:
python predict_quityrs.py --input sample.json
"""

import argparse
import json
import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model_quityrs_rf_calibrated.pkl")


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
    "DEPFEELELL",
    "CIGSLONGFS",
    "QUITNO",
]

LABEL_COLS = ["QUITYRS", "CIGSLONGFS", "QUITNO"]


# ------------------------------
# Artifact Loader
# ------------------------------
def load_artifact(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------
# Preprocessing Helpers
# ------------------------------
def apply_impute(df: pd.DataFrame, imputers: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col, val in imputers.get("mode", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col, val in imputers.get("median", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def apply_encoders(df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = enc.transform(df[col].astype(str))
    return df


# ------------------------------
# Prediction Logic
# ------------------------------
def predict(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    threshold = artifact.get("threshold", 0.5)
    imputers = artifact.get("imputers", {})
    encoders = artifact.get("encoders", {})

    # Eksik kolonları tamamla
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    df = apply_impute(df, imputers)
    df = apply_encoders(df, encoders)

    X = df[feature_names]

    # predict_proba kontrolü
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"Model predict_proba desteklemiyor: {type(model)}")

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["prob_positive"] = proba
    out["pred_label"] = pred

    return out[feature_names + ["prob_positive", "pred_label"]]


# ------------------------------
# MAIN CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON dosyası (tek kayıt veya liste)")
    args = parser.parse_args()

    model_path = MODEL_PATH

    artifact = load_artifact(model_path)

    with open(args.input, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)
    preds = predict(df, artifact)

    print(preds.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
