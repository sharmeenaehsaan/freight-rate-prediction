from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


MODEL_FILE = Path("catboost_final_log_model.cbm")
DECEMBER_FILE = Path("data/december-chart-inputs.csv")


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


def prepare_features(df):

    df = df.copy()

    # ---------------------------------------------------------
    # Basic preprocessing
    # ---------------------------------------------------------

    # Same weight handling used during final model training
    df["weight"] = pd.to_numeric(
        df["weight"],
        errors="coerce"
    )

    df["weight"] = df["weight"].abs()

    df["weight"] = df["weight"].fillna(31496.0)

    # ---------------------------------------------------------
    # Features not supplied by December input
    # ---------------------------------------------------------

    # December data does not contain market_index.
    # Use the training median used by the final model.
    df["market_index"] = 1.055800

    # December data does not contain quote_signal.
    # Use the neutral/default value used for missing input.
    df["quote_signal"] = 0.0

    # December data does not contain geographic coordinates.
    # Use neutral values because coordinates are unavailable.
    df["pickup_lat"] = 0.0
    df["pickup_lon"] = 0.0
    df["delivery_lat"] = 0.0
    df["delivery_lon"] = 0.0

    # ---------------------------------------------------------
    # Date features
    # ---------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # ---------------------------------------------------------
    # Route feature
    # ---------------------------------------------------------

    df["route"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    # ---------------------------------------------------------
    # Final feature order
    # ---------------------------------------------------------

    return df[FEATURES]


def main():

    print("=" * 70)
    print("GENERATING DECEMBER PREDICTIONS")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("\nLoading final model...")

    model = CatBoostRegressor()
    model.load_model(MODEL_FILE)

    print("Model loaded:")
    print(f"  {MODEL_FILE}")

    # ---------------------------------------------------------
    # Load December input
    # ---------------------------------------------------------

    print("\nLoading December data...")

    df = pd.read_csv(DECEMBER_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # Prepare features
    # ---------------------------------------------------------

    print("\nPreparing December features...")

    X = prepare_features(df)

    print(f"Feature rows: {len(X):,}")
    print(f"Feature columns: {len(X.columns)}")

    print("\nFinal feature list:")
    for column in X.columns:
        print(f"  - {column}")

    # Check for missing values
    missing = X.isna().sum().sum()

    print(f"\nTotal missing feature values: {missing}")

    if missing > 0:
        raise ValueError(
            "December feature matrix still contains missing values."
        )

    # ---------------------------------------------------------
    # Generate predictions
    # ---------------------------------------------------------

    print("\nGenerating predictions...")

    predicted_log_rate = model.predict(X)

    # Model was trained on log1p(posted_rate)
    predicted_rate = np.expm1(predicted_log_rate)

    predicted_rate = np.asarray(
        predicted_rate,
        dtype=float
    )

    # Safety validation
    if not np.isfinite(predicted_rate).all():
        raise ValueError(
            "Model generated NaN or infinite predictions."
        )

    if (predicted_rate <= 0).any():
        raise ValueError(
            "Model generated non-positive predictions."
        )

    # ---------------------------------------------------------
    # Add predictions
    # ---------------------------------------------------------

    df["predicted_rate"] = predicted_rate

    # ---------------------------------------------------------
    # Validate output
    # ---------------------------------------------------------

    print("\nChecking predictions...")

    print(
        f"Prediction mean:   "
        f"${df['predicted_rate'].mean():,.2f}"
    )

    print(
        f"Prediction median: "
        f"${df['predicted_rate'].median():,.2f}"
    )

    print(
        f"Prediction min:    "
        f"${df['predicted_rate'].min():,.2f}"
    )

    print(
        f"Prediction max:    "
        f"${df['predicted_rate'].max():,.2f}"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        DECEMBER_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("DECEMBER PREDICTIONS CREATED")
    print("=" * 70)

    print(f"Output file: {DECEMBER_FILE}")
    print(f"Rows: {len(df):,}")

    print("\nDecember predictions:")

    print(
        df[
            [
                "pickup",
                "delivery",
                "distance",
                "equipment",
                "weight",
                "date",
                "predicted_rate",
            ]
        ].to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()