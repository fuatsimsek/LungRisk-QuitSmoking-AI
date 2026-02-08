"""
Cleaning script: lung_cancer_reduced.csv -> clean_lung_cancer.csv
- Rename columns to snake_case
- Keep only the required feature columns
- Convert level to numeric (LOW=1, MEDIUM=2, HIGH=3) if needed
- Cast all columns to numeric, fill NaNs with median per column
- Shuffle and save
"""

import re
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "lung_cancer_reduced.csv"
OUTPUT_PATH = BASE_DIR / "clean_lung_cancer.csv"
TARGET_COL = "level"

REQUIRED_COLUMNS: List[str] = [
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
    "level",
]

LEVEL_MAP = {"low": 1, "medium": 2, "high": 3}


def to_snake(name: str) -> str:
    """Convert arbitrary column names to snake_case lowercase."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    name = name.replace("occu_pational", "occupational")
    name = re.sub(r"_+", "_", name)
    return name


def load_raw() -> pd.DataFrame:
    """Load the raw CSV with a few fallback encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(RAW_PATH, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {RAW_PATH}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and retain only required ones."""
    df = df.rename(columns={c: to_snake(c) for c in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after rename: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def normalize_level(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure level is numeric 1/2/3."""
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower().map(LEVEL_MAP)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(float)
    return df


def to_numeric_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all columns to numeric and fill NaNs with column median."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0)
    return df


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = load_raw()
    df = normalize_columns(df)
    df = normalize_level(df)
    df = to_numeric_and_impute(df)

    # Shuffle to avoid any ordering bias
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"[INFO] Clean dataset saved to {OUTPUT_PATH} "
        f"(rows={len(df)}, cols={df.shape[1]})"
    )


if __name__ == "__main__":
    main()

