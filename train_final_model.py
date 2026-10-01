from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = Path("data/train-test.csv")
VALIDATION_FILE = Path("data/validation.csv")
OUTPUT_FILE = Path("validation_predictions.csv")
MODEL_FILE = Path("catboost_final_log_model.cbm")

RANDOM_SEED = 42
BEST_ITERATIONS = 282


# ============================================================
# FEATURE CREATION
# ============================================================

FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "route",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


def prepare_features(df, weight_median, market_median):
    """
    Prepare features using the same preprocessing used by
    the selected Random Log1p model.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Clean weight
    # --------------------------------------------------------
    # Negative weights were identified as sign errors.
    df["weight"] = df["weight"].abs()

    # Fill missing values using training-data medians.
    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(market_median)

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------
    df["route"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    # --------------------------------------------------------
    # Return only model features
    # --------------------------------------------------------
    return df[FEATURES]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL FREIGHT RATE MODEL")
    print("Random Split + Log1p CatBoost")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load development data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_df = pd.read_csv(TRAIN_FILE)

    print(f"Training rows: {len(train_df):,}")
    print(f"Training columns: {len(train_df.columns)}")

    if "posted_rate" not in train_df.columns:
        raise ValueError(
            "posted_rate column is missing from training data."
        )

    # --------------------------------------------------------
    # 2. Load validation prediction data
    # --------------------------------------------------------

    print("\nLoading validation data...")

    validation_df = pd.read_csv(VALIDATION_FILE)

    print(f"Validation rows: {len(validation_df):,}")

    if "posted_rate" in validation_df.columns:
        raise ValueError(
            "validation.csv should not contain posted_rate."
        )

    # --------------------------------------------------------
    # 3. Check validation IDs
    # --------------------------------------------------------

    if "load_id" not in validation_df.columns:
        raise ValueError(
            "load_id column is missing from validation.csv."
        )

    if validation_df["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values found in validation.csv."
        )

    # --------------------------------------------------------
    # 4. Calculate preprocessing values ONLY from all
    #    labeled development data
    # --------------------------------------------------------

    weight_median = train_df["weight"].abs().median()
    market_median = train_df["market_index"].median()

    print("\nTraining preprocessing values:")
    print(f"Weight median:       {weight_median:.4f}")
    print(f"Market index median: {market_median:.6f}")

    # --------------------------------------------------------
    # 5. Prepare features
    # --------------------------------------------------------

    print("\nPreparing training features...")

    X_train = prepare_features(
        train_df,
        weight_median,
        market_median,
    )

    print("Preparing validation features...")

    X_validation = prepare_features(
        validation_df,
        weight_median,
        market_median,
    )

    # --------------------------------------------------------
    # 6. Target
    # --------------------------------------------------------

    y = train_df["posted_rate"].astype(float)

    if not np.isfinite(y).all():
        raise ValueError(
            "Training target contains NaN or infinite values."
        )

    if (y <= 0).any():
        raise ValueError(
            "Training target contains non-positive values."
        )

    # Same transformation as the selected $93.15 model.
    y_log = np.log1p(y)

    print("\nTarget:")
    print(f"Mean posted rate:   ${y.mean():,.2f}")
    print(f"Median posted rate: ${y.median():,.2f}")
    print(f"Min posted rate:    ${y.min():,.2f}")
    print(f"Max posted rate:    ${y.max():,.2f}")

    # --------------------------------------------------------
    # 7. Train final CatBoost model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    print(f"Rows used for training: {len(X_train):,}")
    print(f"Iterations:             {BEST_ITERATIONS}")
    print("Learning rate:          0.05")
    print("Depth:                  8")
    print("Target:                 log1p(posted_rate)")
    print("Random seed:            42")

    model = CatBoostRegressor(
        iterations=BEST_ITERATIONS,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=50,
    )

    model.fit(
        X_train,
        y_log,
        cat_features=CATEGORICAL_FEATURES,
    )

    print("\nFinal model training completed.")

    # --------------------------------------------------------
    # 8. Save model
    # --------------------------------------------------------

    model.save_model(MODEL_FILE)

    print(f"Model saved to: {MODEL_FILE}")

    # --------------------------------------------------------
    # 9. Predict validation data
    # --------------------------------------------------------

    print("\nGenerating validation predictions...")

    predictions_log = model.predict(X_validation)

    # Convert predictions back from log scale.
    predictions = np.expm1(predictions_log)

    # Ensure valid predictions.
    predictions = np.asarray(predictions, dtype=float)

    predictions = np.nan_to_num(
        predictions,
        nan=1.0,
        posinf=1e9,
        neginf=1.0,
    )

    # Rates must be positive.
    predictions = np.maximum(predictions, 0.01)

    # --------------------------------------------------------
    # 10. Create required submission file
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": predictions,
    })

    # --------------------------------------------------------
    # 11. Validate output
    # --------------------------------------------------------

    print("\nChecking prediction file...")

    if len(predictions_df) != len(validation_df):
        raise ValueError(
            "Prediction row count does not match validation row count."
        )

    if predictions_df["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load_id values found in predictions."
        )

    if not np.isfinite(
        predictions_df["predicted_rate"]
    ).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    if (
        predictions_df["predicted_rate"] <= 0
    ).any():
        raise ValueError(
            "Predictions contain non-positive values."
        )

    # Make sure columns are exactly as required.
    predictions_df = predictions_df[
        ["load_id", "predicted_rate"]
    ]

    # --------------------------------------------------------
    # 12. Save
    # --------------------------------------------------------

    predictions_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 13. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PREDICTION FILE CREATED")
    print("=" * 70)

    print(f"Output file:       {OUTPUT_FILE}")
    print(f"Rows:              {len(predictions_df):,}")
    print(
        f"Prediction mean:   "
        f"${predictions_df['predicted_rate'].mean():,.2f}"
    )
    print(
        f"Prediction median: "
        f"${predictions_df['predicted_rate'].median():,.2f}"
    )
    print(
        f"Prediction min:    "
        f"${predictions_df['predicted_rate'].min():,.2f}"
    )
    print(
        f"Prediction max:    "
        f"${predictions_df['predicted_rate'].max():,.2f}"
    )

    print("\nFirst 10 predictions:")
    print(predictions_df.head(10).to_string(index=False))

    print("\nRequired columns:")
    print(list(predictions_df.columns))

    print("\nDone.")


if __name__ == "__main__":
    main()