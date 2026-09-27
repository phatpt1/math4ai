import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def validate_dataset(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset thiếu các cột bắt buộc: " + ", ".join(missing)
        )


def normalize_dataset(df: pd.DataFrame):
    df = df.copy()
    validate_dataset(df)

    if df[TARGET].dtype == bool:
        df[TARGET] = df[TARGET].astype(int)
    else:
        mapping = {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "TRUE": 1,
            "FALSE": 0,
            "1": 1,
            "0": 0,
            1: 1,
            0: 0,
        }

        converted = df[TARGET].map(mapping)

        if converted.isna().any():
            try:
                converted = df[TARGET].astype(int)
            except Exception as e:
                raise ValueError(
                    "Cột Revenue phải là True/False hoặc 1/0."
                ) from e

        df[TARGET] = converted.astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Revenue phải chỉ gồm 0/1.")

    category_levels = {}

    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = (
                df[col]
                .astype("object")
                .where(df[col].notna(), "Missing")
            )

        df[col] = df[col].astype("category")
        category_levels[col] = list(df[col].cat.categories)

    return df, category_levels


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 70 / 15 / 15 split: first carve out test, then validation from the rest
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_defaults(X_train):
    defaults = {}

    for col in NUMERIC_FEATURES:
        defaults[col] = float(
            pd.to_numeric(X_train[col]).median()
        )

    for col in CATEGORICAL_FEATURES:
        mode = X_train[col].mode(dropna=True)
        defaults[col] = mode.iloc[0] if len(mode) else None

    return defaults