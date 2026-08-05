"""
XGBoost Model Training Pipeline
================================
Trains three XGBoost regressors to predict sponsorship deal outcomes:
  1. media_value_ratio — earned media value as a multiplier of spend
  2. brand_lift_pct — predicted brand awareness increase (%)
  3. roi_score — composite ROI score (1-100)

The pipeline handles:
- Feature engineering (log transforms, interaction features, one-hot encoding)
- Train/test split (80/20)
- Fixed, documented hyperparameters
- Evaluation (MAE, R², MAPE)
- Model serialization with joblib

Output:
  model/trained_model.joblib  — dict of fitted models + preprocessor metadata
  model/metrics.json          — evaluation metrics for each target
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ──────────────────────────────────────────────────────────────────────
# Paths — all relative to this script's directory so the project
# can be run from anywhere without path issues
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sponsorship_deals.csv"
MODEL_PATH = Path(__file__).parent / "trained_model.joblib"
METRICS_PATH = Path(__file__).parent / "metrics.json"

# ──────────────────────────────────────────────────────────────────────
# Target columns the model will predict
# ──────────────────────────────────────────────────────────────────────
TARGETS = ["media_value_ratio", "brand_lift_pct", "roi_score"]

# ──────────────────────────────────────────────────────────────────────
# Categorical columns that need one-hot encoding
# ──────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["sport", "market", "deal_type", "brand_category"]

# ──────────────────────────────────────────────────────────────────────
# Binary activation channel columns — already 0/1, no encoding needed
# ──────────────────────────────────────────────────────────────────────
BINARY_COLS = ["on_air", "digital", "social", "experiential", "dooh"]

# ──────────────────────────────────────────────────────────────────────
# Numeric columns that benefit from log transformation
# Spend and reach span several orders of magnitude, so log transform
# compresses the scale and helps tree-based models find better splits
# ──────────────────────────────────────────────────────────────────────
LOG_TRANSFORM_COLS = ["annual_spend", "audience_reach", "social_following"]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw features into model-ready format.

    Steps:
    1. Log-transform high-variance numeric columns (spend, reach, social)
    2. Create interaction features (sport×market combos capture local effects)
    3. Compute activation_count (total active channels — a useful signal)
    4. One-hot encode all categorical columns
    5. Drop original categorical and raw numeric columns

    Returns a DataFrame with only numeric features ready for XGBoost.
    """
    df = df.copy()

    # --- Step 1: Log-transform skewed numeric features ---
    # Adding 1 before log to handle any zero values safely
    for col in LOG_TRANSFORM_COLS:
        df[f"log_{col}"] = np.log1p(df[col])

    # --- Step 2: Interaction features ---
    # Sport × market captures that NFL in New York is different from NFL in Denver
    # This interaction is too sparse for full one-hot, so we encode it as a
    # combined categorical and then one-hot encode the result
    df["sport_market"] = df["sport"] + "_" + df["market"]

    # --- Step 3: Activation channel count ---
    # Total number of active channels — deals with more channels tend to
    # have higher ROI due to cross-channel amplification effects
    df["activation_count"] = df[BINARY_COLS].sum(axis=1)

    # --- Step 4: One-hot encode categoricals ---
    # drop_first=True avoids multicollinearity (the dropped category
    # becomes the implicit reference level)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS + ["sport_market"], drop_first=True)

    # --- Step 5: Drop raw numeric columns (keep log-transformed versions) ---
    df = df.drop(columns=LOG_TRANSFORM_COLS, errors="ignore")

    # --- Step 6: Keep deal_length_years as-is (1-5 is already a reasonable scale) ---
    return df


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Clips denominator to avoid division by near-zero values.
    Returns result as a percentage (0-100 scale).
    """
    y_true_safe = np.clip(np.abs(y_true), 1e-3, None)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100)


def prepare_feature_partitions(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split raw rows first, then fit the encoded feature schema on training rows.

    The test partition is transformed independently and aligned to the training
    columns. Categories seen only in the held-out partition therefore cannot
    influence the schema used to fit a model.
    """
    missing_targets = sorted(set(TARGETS) - set(df.columns))
    if missing_targets:
        raise ValueError(f"dataset is missing target columns: {missing_targets}")

    raw_features = df.drop(columns=TARGETS)
    train_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
    )
    X_train = feature_engineering(raw_features.iloc[train_indices])
    X_test = feature_engineering(raw_features.iloc[test_indices]).reindex(
        columns=X_train.columns,
        fill_value=0,
    )
    return X_train, X_test, train_indices, test_indices


def train_models():
    """
    Main training function. Loads data, engineers features, trains one
    XGBoost regressor per target, evaluates on held-out test set, and
    saves everything needed for inference.
    """
    # ──────────────────────────────────────────────────────────────────
    # Load the synthetic dataset
    # ──────────────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")

    # ──────────────────────────────────────────────────────────────────
    # Separate targets before feature engineering
    # We need the raw target values for evaluation, and we don't want
    # the targets leaking into the feature set
    # ──────────────────────────────────────────────────────────────────
    y_dict = {target: df[target].values for target in TARGETS}

    # ──────────────────────────────────────────────────────────────────
    # Split raw rows first, then derive the feature schema from training data
    # only. The held-out rows cannot influence one-hot column discovery.
    # ──────────────────────────────────────────────────────────────────
    print("Engineering features...")
    X_train, X_test, indices_train, indices_test = prepare_feature_partitions(df)
    feature_cols = list(X_train.columns)

    print(f"  Training feature matrix shape: {X_train.shape}")
    print(f"  Held-out feature matrix shape: {X_test.shape}")
    print(f"  Features: {len(feature_cols)} total")

    # ──────────────────────────────────────────────────────────────────
    # XGBoost hyperparameters — tuned for this dataset size (~5K records)
    # n_estimators=300 with max_depth=6 provides good fit without overfitting
    # learning_rate=0.05 with subsample=0.8 adds regularization
    # ──────────────────────────────────────────────────────────────────
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
    }

    # ──────────────────────────────────────────────────────────────────
    # Train one model per target variable
    # Each target has different characteristics so separate models
    # outperform a single multi-output model here
    # ──────────────────────────────────────────────────────────────────
    models = {}
    metrics = {}
    held_out = {}

    for target in TARGETS:
        print(f"\nTraining model for: {target}")

        y_train = y_dict[target][indices_train]
        y_test = y_dict[target][indices_test]

        # Initialize and fit the XGBoost regressor
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)

        # Generate predictions on the held-out test set
        y_pred = model.predict(X_test)

        # Compute evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = compute_mape(y_test, y_pred)

        metrics[target] = {
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        }
        held_out[target] = {
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist(),
        }

        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        models[target] = model

    # ──────────────────────────────────────────────────────────────────
    # Save everything needed for inference into a single joblib file:
    # - The trained models (one per target)
    # - The feature column names (needed to reconstruct the input matrix)
    # - The list of categorical and binary columns (for the app's preprocessing)
    # ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model artifacts to {MODEL_PATH}...")
    artifact = {
        "models": models,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "binary_cols": BINARY_COLS,
        "log_transform_cols": LOG_TRANSFORM_COLS,
        "targets": TARGETS,
        "held_out": held_out,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"  Model file size: {MODEL_PATH.stat().st_size / 1024:.0f} KB")

    # ──────────────────────────────────────────────────────────────────
    # Save evaluation metrics as JSON for the app's "Under the Hood" section
    # ──────────────────────────────────────────────────────────────────
    print(f"Saving metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete.")
    return models, metrics


if __name__ == "__main__":
    train_models()
