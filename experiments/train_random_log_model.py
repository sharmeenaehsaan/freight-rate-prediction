# ============================================================
# Random-Split Log1p CatBoost Model
# Freight Rate Prediction Challenge
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train-test.csv")

MODEL_PATH = Path("catboost_random_log_model.cbm")
ERROR_PATH = Path("random_log_model_error_analysis.csv")

RANDOM_STATE = 42

HIGH_RATE_THRESHOLD = 5000
EXTREME_RATE_THRESHOLD = 7500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_features(df):
    """
    Clean the dataset and create model features.

    Important:
    - geo_distance is intentionally NOT created.
    - rate_per_distance is NOT created because it uses the target
      posted_rate and would cause target leakage.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    # Negative weights appear to be sign errors.
    df["weight"] = df["weight"].abs()

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
    # Route feature
    # --------------------------------------------------------

    df["route"] = (
        df["pickup"].astype(str)
        + " -> "
        + df["delivery"].astype(str)
    )

    return df


def calculate_metrics(y_true, y_pred, name):
    """
    Print MAE, RMSE and R2.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RANDOM-SPLIT LOG1P CATBOOST MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified.")


# ============================================================
# CREATE FEATURES
# ============================================================

df = add_features(df)


# ============================================================
# MODEL FEATURES
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


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df.dropna(subset=["posted_rate"]).copy()

df = df[
    np.isfinite(df["posted_rate"])
].copy()

df = df[
    df["posted_rate"] > 0
].copy()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(f"Rows: {len(df):,}")
print(f"Mean: ${df['posted_rate'].mean():,.2f}")
print(f"Median: ${df['posted_rate'].median():,.2f}")
print(f"Min: ${df['posted_rate'].min():,.2f}")
print(f"Max: ${df['posted_rate'].max():,.2f}")

high_rate_count = (
    df["posted_rate"] >= HIGH_RATE_THRESHOLD
).sum()

extreme_count = (
    df["posted_rate"] >= EXTREME_RATE_THRESHOLD
).sum()

print(
    f"\nHigh-rate >= ${HIGH_RATE_THRESHOLD:,}: "
    f"{high_rate_count:,} "
    f"({high_rate_count / len(df) * 100:.2f}%)"
)

print(
    f"Extreme >= ${EXTREME_RATE_THRESHOLD:,}: "
    f"{extreme_count:,} "
    f"({extreme_count / len(df) * 100:.2f}%)"
)


# ============================================================
# CREATE EXTREME FLAG FOR STRATIFIED SPLIT
# ============================================================

# This flag is ONLY used to make sure the rare extreme
# examples are represented in both train and validation.
#
# It is NOT given to CatBoost as a feature.

df["extreme_flag"] = (
    df["posted_rate"] >= EXTREME_RATE_THRESHOLD
).astype(int)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating random 80/20 split...")

train_df, valid_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["extreme_flag"],
)

print(f"Training rows   : {len(train_df):,}")
print(f"Validation rows : {len(valid_df):,}")


# ============================================================
# CHECK EXTREME RATE DISTRIBUTION
# ============================================================

train_extreme = (
    train_df["posted_rate"] >= EXTREME_RATE_THRESHOLD
).sum()

valid_extreme = (
    valid_df["posted_rate"] >= EXTREME_RATE_THRESHOLD
).sum()

print("\nExtreme-rate distribution:")
print(
    f"Training   >= ${EXTREME_RATE_THRESHOLD:,}: "
    f"{train_extreme:,} "
    f"({train_extreme / len(train_df) * 100:.2f}%)"
)

print(
    f"Validation >= ${EXTREME_RATE_THRESHOLD:,}: "
    f"{valid_extreme:,} "
    f"({valid_extreme / len(valid_df) * 100:.2f}%)"
)


# ============================================================
# TRAIN-ONLY IMPUTATION
# ============================================================

# Important:
# Calculate medians ONLY from training data.
# This avoids preprocessing leakage.

weight_median = train_df["weight"].median()

market_index_median = train_df["market_index"].median()

print("\nTraining medians:")
print(f"Weight median       : {weight_median}")
print(f"Market index median : {market_index_median}")


train_df["weight"] = train_df["weight"].fillna(
    weight_median
)

valid_df["weight"] = valid_df["weight"].fillna(
    weight_median
)

train_df["market_index"] = train_df["market_index"].fillna(
    market_index_median
)

valid_df["market_index"] = valid_df["market_index"].fillna(
    market_index_median
)


# ============================================================
# HANDLE MISSING CATEGORICAL VALUES
# ============================================================

for col in CATEGORICAL_FEATURES:

    train_df[col] = (
        train_df[col]
        .fillna("Unknown")
        .astype(str)
    )

    valid_df[col] = (
        valid_df[col]
        .fillna("Unknown")
        .astype(str)
    )


# ============================================================
# HANDLE REMAINING NUMERIC MISSING VALUES
# ============================================================

numeric_features = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
]

for col in numeric_features:

    median_value = train_df[col].median()

    train_df[col] = train_df[col].fillna(
        median_value
    )

    valid_df[col] = valid_df[col].fillna(
        median_value
    )


# ============================================================
# PREPARE X AND y
# ============================================================

X_train = train_df[FEATURES].copy()
X_valid = valid_df[FEATURES].copy()

y_train = train_df["posted_rate"].copy()
y_valid = valid_df["posted_rate"].copy()


# ============================================================
# LOG1P TARGET
# ============================================================

print("\nTransforming target using log1p...")

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)


# ============================================================
# CATBOOST MODEL
# ============================================================

print("\nTraining CatBoost...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,

    loss_function="RMSE",
    eval_metric="RMSE",

    random_seed=RANDOM_STATE,

    verbose=100,

    # Prevent unnecessary overfitting
    l2_leaf_reg=3,

    # Use validation set for early stopping
    od_type="Iter",
    od_wait=80,

    use_best_model=True,
)


model.fit(
    X_train,
    y_train_log,

    cat_features=CATEGORICAL_FEATURES,

    eval_set=(
        X_valid,
        y_valid_log
    ),
)


# ============================================================
# BEST ITERATION
# ============================================================

best_iteration = model.get_best_iteration()

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest iteration: {best_iteration}"
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating validation predictions...")

valid_pred_log = model.predict(X_valid)

# Convert log predictions back to dollars
valid_pred = np.expm1(valid_pred_log)

# Rates cannot be negative
valid_pred = np.maximum(valid_pred, 0)


# ============================================================
# OVERALL METRICS
# ============================================================

calculate_metrics(
    y_valid,
    valid_pred,
    "OVERALL VALIDATION RESULTS"
)


# ============================================================
# HIGH-RATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RATE ANALYSIS")
print("=" * 70)

high_mask = (
    y_valid >= HIGH_RATE_THRESHOLD
)

extreme_mask = (
    y_valid >= EXTREME_RATE_THRESHOLD
)

print(
    f"\nHigh-rate validation rows "
    f"(>= ${HIGH_RATE_THRESHOLD:,}): "
    f"{high_mask.sum():,}"
)

if high_mask.sum() > 0:

    calculate_metrics(
        y_valid[high_mask],
        valid_pred[high_mask],
        f"High-rate >= ${HIGH_RATE_THRESHOLD:,}"
    )


print(
    f"\nExtreme validation rows "
    f"(>= ${EXTREME_RATE_THRESHOLD:,}): "
    f"{extreme_mask.sum():,}"
)

if extreme_mask.sum() > 0:

    calculate_metrics(
        y_valid[extreme_mask],
        valid_pred[extreme_mask],
        f"Extreme >= ${EXTREME_RATE_THRESHOLD:,}"
    )


# ============================================================
# RATE BANDS
# ============================================================

print("\n" + "=" * 70)
print("RATE-BAND ANALYSIS")
print("=" * 70)

rate_bands = [
    ("<$5k", 0, 5000),
    ("$5k-$7.5k", 5000, 7500),
    ("$7.5k-$10k", 7500, 10000),
    ("$10k-$15k", 10000, 15000),
    ("$15k-$20k", 15000, 20000),
    ("$20k+", 20000, np.inf),
]

band_results = []

for band_name, lower, upper in rate_bands:

    if np.isinf(upper):

        mask = y_valid >= lower

    else:

        mask = (
            (y_valid >= lower)
            & (y_valid < upper)
        )

    count = mask.sum()

    if count == 0:
        continue

    mae = mean_absolute_error(
        y_valid[mask],
        valid_pred[mask]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid[mask],
            valid_pred[mask]
        )
    )

    actual_mean = y_valid[mask].mean()
    predicted_mean = valid_pred[mask].mean()

    band_results.append({
        "rate_band": band_name,
        "count": count,
        "MAE": mae,
        "RMSE": rmse,
        "actual_mean": actual_mean,
        "predicted_mean": predicted_mean,
    })

    print(
        f"\n{band_name}"
        f"\n  Count          : {count:,}"
        f"\n  MAE            : ${mae:,.2f}"
        f"\n  RMSE           : ${rmse:,.2f}"
        f"\n  Actual mean    : ${actual_mean:,.2f}"
        f"\n  Predicted mean : ${predicted_mean:,.2f}"
    )


# ============================================================
# DISTANCE-BAND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DISTANCE-BAND ANALYSIS")
print("=" * 70)

valid_analysis = valid_df.copy()

valid_analysis["actual_rate"] = y_valid.values
valid_analysis["predicted_rate"] = valid_pred

valid_analysis["distance_group"] = pd.cut(
    valid_analysis["distance"],
    bins=[
        0,
        250,
        500,
        750,
        1000,
        1500,
        float("inf"),
    ],
    labels=[
        "0-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1500+",
    ],
)

distance_results = []

for group in valid_analysis["distance_group"].cat.categories:

    mask = (
        valid_analysis["distance_group"]
        == group
    )

    subset = valid_analysis[mask]

    if len(subset) == 0:
        continue

    mae = mean_absolute_error(
        subset["actual_rate"],
        subset["predicted_rate"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            subset["actual_rate"],
            subset["predicted_rate"]
        )
    )

    distance_results.append({
        "distance_group": str(group),
        "count": len(subset),
        "MAE": mae,
        "RMSE": rmse,
        "actual_mean": subset["actual_rate"].mean(),
        "predicted_mean": subset["predicted_rate"].mean(),
    })

    print(
        f"\n{group}"
        f"\n  Count          : {len(subset):,}"
        f"\n  MAE            : ${mae:,.2f}"
        f"\n  RMSE           : ${rmse:,.2f}"
        f"\n  Actual mean    : ${subset['actual_rate'].mean():,.2f}"
        f"\n  Predicted mean : ${subset['predicted_rate'].mean():,.2f}"
    )


# ============================================================
# EQUIPMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EQUIPMENT ANALYSIS")
print("=" * 70)

equipment_results = []

for equipment in sorted(
    valid_analysis["equipment"].unique()
):

    mask = (
        valid_analysis["equipment"]
        == equipment
    )

    subset = valid_analysis[mask]

    mae = mean_absolute_error(
        subset["actual_rate"],
        subset["predicted_rate"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            subset["actual_rate"],
            subset["predicted_rate"]
        )
    )

    equipment_results.append({
        "equipment": equipment,
        "count": len(subset),
        "MAE": mae,
        "RMSE": rmse,
        "actual_mean": subset["actual_rate"].mean(),
        "predicted_mean": subset["predicted_rate"].mean(),
    })

    print(
        f"\n{equipment}"
        f"\n  Count          : {len(subset):,}"
        f"\n  MAE            : ${mae:,.2f}"
        f"\n  RMSE           : ${rmse:,.2f}"
        f"\n  Actual mean    : ${subset['actual_rate'].mean():,.2f}"
        f"\n  Predicted mean : ${subset['predicted_rate'].mean():,.2f}"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

valid_analysis["error"] = (
    valid_analysis["predicted_rate"]
    - valid_analysis["actual_rate"]
)

valid_analysis["absolute_error"] = (
    valid_analysis["error"].abs()
)

valid_analysis["percentage_error"] = (
    valid_analysis["absolute_error"]
    / valid_analysis["actual_rate"]
    * 100
)


# ============================================================
# TOP 30 ERRORS
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 ABSOLUTE ERRORS")
print("=" * 70)

top_errors = (
    valid_analysis
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(30)
)

display_columns = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "date",
    "actual_rate",
    "predicted_rate",
    "error",
    "absolute_error",
]

print(
    top_errors[display_columns]
    .to_string(index=False)
)


# ============================================================
# EXTREME ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXTREME-RATE ERROR ANALYSIS")
print("=" * 70)

extreme_rows = (
    valid_analysis[
        valid_analysis["actual_rate"]
        >= EXTREME_RATE_THRESHOLD
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
)

if len(extreme_rows) > 0:

    print(
        extreme_rows[
            display_columns
        ].to_string(index=False)
    )


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

valid_analysis.to_csv(
    ERROR_PATH,
    index=False
)

print(
    f"\nError analysis saved to:"
    f"\n{ERROR_PATH}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(
    MODEL_PATH
)

print(
    f"\nModel saved to:"
    f"\n{MODEL_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

overall_mae = mean_absolute_error(
    y_valid,
    valid_pred
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        valid_pred
    )
)

overall_r2 = r2_score(
    y_valid,
    valid_pred
)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"\nRandom split : 80% train / 20% validation"
)

print(
    f"Stratified by: Extreme rate >= ${EXTREME_RATE_THRESHOLD:,}"
)

print(
    f"Training rows:   {len(train_df):,}"
)

print(
    f"Validation rows: {len(valid_df):,}"
)

print(
    f"Extreme train:   {train_extreme:,}"
)

print(
    f"Extreme valid:   {valid_extreme:,}"
)

print(
    f"\nOverall MAE  : ${overall_mae:,.2f}"
)

print(
    f"Overall RMSE : ${overall_rmse:,.2f}"
)

print(
    f"Overall R²   : {overall_r2:.4f}"
)

print(
    f"\nBest iteration: {best_iteration}"
)

print("\nFiles created:")

print(
    f"  - {MODEL_PATH}"
)

print(
    f"  - {ERROR_PATH}"
)

print("\nDone.")
print("=" * 70)