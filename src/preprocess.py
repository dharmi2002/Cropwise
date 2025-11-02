"""Preprocessing: cleaning, feature selection and encoding."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Dict

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    # simple NA handling: fill numeric with median, categorical with mode
    for c in df.columns:
        if df[c].dtype.kind in 'biufc':
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
    return df

def select_common_features(train: pd.DataFrame, test: pd.DataFrame, target_names: List[str]=None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    # Determine likely target column
    possible_targets = ["label", "crop", "Crop", "crop_type"]
    tcol = None
    for c in train.columns:
        if c in possible_targets:
            tcol = c
            break
    if tcol is None:
        raise KeyError("No target column found in training data. Expected one of: " + ", ".join(possible_targets))
    # numeric + low-cardinality categorical features common to both
    common = [c for c in train.columns if c in test.columns and c != tcol]
    # keep numeric and small-cardinality categoricals
    features: List[str] = []
    for c in common:
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c]):
            features.append(c)
        else:
            # allow categoricals if unique values reasonably small
            if train[c].nunique() < 50 and test[c].nunique() < 50:
                features.append(c)
    X_train = train[features].copy()
    y_train = train[tcol].copy()
    X_test = test[features].copy()
    y_test = test[tcol].copy() if tcol in test.columns else None
    return X_train, y_train, X_test, y_test, features

def encode_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    # Encode categorical columns with LabelEncoder (fit on train)
    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            le = LabelEncoder()
            X_train[c] = le.fit_transform(X_train[c].astype(str))
            X_test[c] = le.transform(X_test[c].astype(str))
    # Scale numerics
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train.values)
    X_test[X_test.columns] = scaler.transform(X_test.values)
    return X_train, X_test
