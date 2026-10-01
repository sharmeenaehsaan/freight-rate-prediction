# ============================================================
# Random-Split Log1p CatBoost + Historical Route Features
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

MODEL_PATH = Path("catboost_random_log_route_model.cbm")
ERROR_PATH = Path("random_log_route_model_error_analysis.csv")

RANDOM_STATE = 42

HIGH_RATE_THRESHOLD = 5000
EXTREME_RATE_THRESHOLD = 7500


# ============================================================
# BASIC FEATURE CREATION
# ============================================================

def add_basic_features(df):
    """
    Create normal non-target-derived features.

    Important:
    - No geo_distance.
    - No rate_per_distance.
    - No target-derived features are created here.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Clean weight
    # --------------------------------------------------------

    df["weight"] = df["weight"].abs()

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

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

    return df


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    name
):
    """
    Calculate and print MAE, RMSE and R2.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# BUILD HISTORICAL ROUTE FEATURES
# ============================================================

def create_route_statistics(
    train_df,
    target_column="posted_rate"
):
    """
    Calculate route statistics using TRAINING DATA ONLY.

    This is extremely important.

    We do NOT use validation targets to calculate these
    statistics.

    Features created:

    route_mean_rate
    route_median_rate
    route_std_rate
    route_sample_count
    route_high_rate_pct
    route_extreme_rate_pct
    """

    route_stats = (
        train_df
        .groupby("route")[target_column]
        .agg(
            route_mean_rate="mean",
            route_median_rate="median",
            route_std_rate="std",
            route_sample_count="count",
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # High-rate percentage
    # --------------------------------------------------------

    high_rate_by_route = (
        train_df
        .assign(
            high_rate=(
                train_df[target_column]
                >= HIGH_RATE_THRESHOLD
            ).astype(int)
        )
        .groupby("route")["high_rate"]
        .mean()
        .reset_index(
            name="route_high_rate_pct"
        )
    )

    # Convert from proportion to percentage
    high_rate_by_route[
        "route_high_rate_pct"
    ] *= 100

    # --------------------------------------------------------
    # Extreme-rate percentage
    # --------------------------------------------------------

    extreme_rate_by_route = (
        train_df
        .assign(
            extreme_rate=(
                train_df[target_column]
                >= EXTREME_RATE_THRESHOLD
            ).astype(int)
        )
        .groupby("route")["extreme_rate"]
        .mean()
        .reset_index(
            name="route_extreme_rate_pct"
        )
    )

    extreme_rate_by_route[
        "route_extreme_rate_pct"
    ] *= 100

    # --------------------------------------------------------
    # Merge everything
    # --------------------------------------------------------

    route_stats = route_stats.merge(
        high_rate_by_route,
        on="route",
        how="left"
    )

    route_stats = route_stats.merge(
        extreme_rate_by_route,
        on="route",
        how="left"
    )

    return route_stats


# ============================================================
# APPLY ROUTE STATISTICS
# ============================================================

def apply_route_statistics(
    df,
    route_stats,
    fallback_values
):
    """
    Apply training-derived route statistics to any dataset.

    If a route was not seen during training, use fallback
    values derived from the training data.
    """

    df = df.copy()

    df = df.merge(
        route_stats,
        on="route",
        how="left"
    )

    # --------------------------------------------------------
    # Fill unseen-route values
    # --------------------------------------------------------

    for column, value in fallback_values.items():

        df[column] = df[column].fillna(value)

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("RANDOM-SPLIT LOG1P + HISTORICAL ROUTE MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# REQUIRED COLUMNS
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
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print(
    "\nRequired columns verified."
)


# ============================================================
# BASIC FEATURES
# ============================================================

df = add_basic_features(
    df
)


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

df = df.dropna(
    subset=["posted_rate"]
).copy()

df = df[
    np.isfinite(
        df["posted_rate"]
    )
].copy()

df = df[
    df["posted_rate"] > 0
].copy()


# ============================================================
# TARGET INFORMATION
# ============================================================

print("\nTarget distribution:")

print(
    f"Rows   : {len(df):,}"
)

print(
    f"Mean   : ${df['posted_rate'].mean():,.2f}"
)

print(
    f"Median : ${df['posted_rate'].median():,.2f}"
)

print(
    f"Min    : ${df['posted_rate'].min():,.2f}"
)

print(
    f"Max    : ${df['posted_rate'].max():,.2f}"
)


high_rate_count = (
    df["posted_rate"]
    >= HIGH_RATE_THRESHOLD
).sum()

extreme_count = (
    df["posted_rate"]
    >= EXTREME_RATE_THRESHOLD
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
# EXTREME FLAG FOR STRATIFIED SPLIT
# ============================================================

df["extreme_flag"] = (
    df["posted_rate"]
    >= EXTREME_RATE_THRESHOLD
).astype(int)


# ============================================================
# RANDOM 80/20 SPLIT
# ============================================================

print(
    "\nCreating random 80/20 split..."
)

train_df, valid_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["extreme_flag"],
)

train_df = train_df.copy()
valid_df = valid_df.copy()

print(
    f"Training rows   : {len(train_df):,}"
)

print(
    f"Validation rows : {len(valid_df):,}"
)


# ============================================================
# EXTREME DISTRIBUTION
# ============================================================

train_extreme = (
    train_df["posted_rate"]
    >= EXTREME_RATE_THRESHOLD
).sum()

valid_extreme = (
    valid_df["posted_rate"]
    >= EXTREME_RATE_THRESHOLD
).sum()

print(
    "\nExtreme-rate distribution:"
)

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
# TRAIN-ONLY IMPUTATION VALUES
# ============================================================

weight_median = (
    train_df["weight"]
    .median()
)

market_index_median = (
    train_df["market_index"]
    .median()
)

print(
    "\nTraining medians:"
)

print(
    f"Weight median       : {weight_median}"
)

print(
    f"Market index median : {market_index_median}"
)


# ============================================================
# APPLY BASIC IMPUTATION
# ============================================================

train_df["weight"] = (
    train_df["weight"]
    .fillna(weight_median)
)

valid_df["weight"] = (
    valid_df["weight"]
    .fillna(weight_median)
)

train_df["market_index"] = (
    train_df["market_index"]
    .fillna(market_index_median)
)

valid_df["market_index"] = (
    valid_df["market_index"]
    .fillna(market_index_median)
)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

for column in categorical_features:

    train_df[column] = (
        train_df[column]
        .fillna("Unknown")
        .astype(str)
    )

    valid_df[column] = (
        valid_df[column]
        .fillna("Unknown")
        .astype(str)
    )


# ============================================================
# NUMERIC FEATURES
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

for column in numeric_features:

    median_value = (
        train_df[column]
        .median()
    )

    train_df[column] = (
        train_df[column]
        .fillna(median_value)
    )

    valid_df[column] = (
        valid_df[column]
        .fillna(median_value)
    )


# ============================================================
# CREATE ROUTE HISTORY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING HISTORICAL ROUTE FEATURES")
print("=" * 70)

print(
    "\nCalculating route statistics from TRAINING DATA ONLY..."
)

route_stats = create_route_statistics(
    train_df
)

print(
    f"Unique training routes: "
    f"{len(route_stats):,}"
)


# ============================================================
# DISPLAY MOST EXTREME ROUTES
# ============================================================

print(
    "\nRoutes with highest historical extreme-rate percentage:"
)

route_display = (
    route_stats[
        route_stats["route_sample_count"] >= 3
    ]
    .sort_values(
        [
            "route_extreme_rate_pct",
            "route_sample_count",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(15)
)

print(
    route_display[
        [
            "route",
            "route_sample_count",
            "route_mean_rate",
            "route_median_rate",
            "route_high_rate_pct",
            "route_extreme_rate_pct",
        ]
    ].to_string(index=False)
)


# ============================================================
# FALLBACK VALUES
# ============================================================

fallback_values = {
    "route_mean_rate":
        train_df["posted_rate"].mean(),

    "route_median_rate":
        train_df["posted_rate"].median(),

    "route_std_rate":
        train_df["posted_rate"].std(),

    "route_sample_count":
        0,

    "route_high_rate_pct":
        (
            train_df["posted_rate"]
            >= HIGH_RATE_THRESHOLD
        ).mean() * 100,

    "route_extreme_rate_pct":
        (
            train_df["posted_rate"]
            >= EXTREME_RATE_THRESHOLD
        ).mean() * 100,
}


# ============================================================
# APPLY ROUTE FEATURES
# ============================================================

train_df = apply_route_statistics(
    train_df,
    route_stats,
    fallback_values
)

valid_df = apply_route_statistics(
    valid_df,
    route_stats,
    fallback_values
)


# ============================================================
# SHOW UNSEEN ROUTES
# ============================================================

unseen_routes = (
    valid_df["route"]
    .isin(
        set(route_stats["route"])
    )
    == False
)

print(
    f"\nValidation rows with unseen routes: "
    f"{unseen_routes.sum():,}"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [

    # Location
    "pickup",
    "delivery",

    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",

    # Load
    "distance",
    "equipment",
    "weight",

    # Market
    "market_index",
    "quote_signal",

    # Date
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",

    # Original route
    "route",

    # Historical route statistics
    "route_mean_rate",
    "route_median_rate",
    "route_std_rate",
    "route_sample_count",
    "route_high_rate_pct",
    "route_extreme_rate_pct",
]


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train_df[
    FEATURES
].copy()

X_valid = valid_df[
    FEATURES
].copy()

y_train = train_df[
    "posted_rate"
].copy()

y_valid = valid_df[
    "posted_rate"
].copy()


# ============================================================
# VERIFY NO MISSING VALUES
# ============================================================

print(
    "\nChecking model features..."
)

train_missing = (
    X_train.isna()
    .sum()
    .sum()
)

valid_missing = (
    X_valid.isna()
    .sum()
    .sum()
)

print(
    f"Training missing values   : {train_missing}"
)

print(
    f"Validation missing values : {valid_missing}"
)

if train_missing > 0 or valid_missing > 0:

    raise ValueError(
        "Missing values remain in model features."
    )


# ============================================================
# LOG1P TARGET
# ============================================================

print(
    "\nTransforming target using log1p..."
)

y_train_log = np.log1p(
    y_train
)

y_valid_log = np.log1p(
    y_valid
)


# ============================================================
# TRAIN CATBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST")
print("=" * 70)

model = CatBoostRegressor(

    iterations=1000,

    learning_rate=0.05,

    depth=8,

    loss_function="RMSE",

    eval_metric="RMSE",

    random_seed=RANDOM_STATE,

    verbose=100,

    l2_leaf_reg=3,

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
        y_valid_log,
    ),
)


# ============================================================
# BEST ITERATION
# ============================================================

best_iteration = (
    model.get_best_iteration()
)

print(
    "\n" + "=" * 70
)

print(
    "MODEL TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nBest iteration: "
    f"{best_iteration}"
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nGenerating validation predictions..."
)

valid_pred_log = (
    model.predict(X_valid)
)

valid_pred = (
    np.expm1(valid_pred_log)
)

valid_pred = np.maximum(
    valid_pred,
    0
)


# ============================================================
# OVERALL METRICS
# ============================================================

calculate_metrics(
    y_valid,
    valid_pred,
    "OVERALL VALIDATION RESULTS"
)


# ============================================================
# HIGH-RATE METRICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "HIGH-RATE ANALYSIS"
)

print(
    "=" * 70
)


high_mask = (
    y_valid
    >= HIGH_RATE_THRESHOLD
)

extreme_mask = (
    y_valid
    >= EXTREME_RATE_THRESHOLD
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
# RATE BAND ANALYSIS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "RATE-BAND ANALYSIS"
)

print(
    "=" * 70
)


rate_bands = [

    (
        "<$5k",
        0,
        5000
    ),

    (
        "$5k-$7.5k",
        5000,
        7500
    ),

    (
        "$7.5k-$10k",
        7500,
        10000
    ),

    (
        "$10k-$15k",
        10000,
        15000
    ),

    (
        "$15k-$20k",
        15000,
        20000
    ),

    (
        "$20k+",
        20000,
        np.inf
    ),
]


for band_name, lower, upper in rate_bands:

    if np.isinf(upper):

        mask = (
            y_valid
            >= lower
        )

    else:

        mask = (
            (y_valid >= lower)
            &
            (y_valid < upper)
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

    actual_mean = (
        y_valid[mask]
        .mean()
    )

    predicted_mean = (
        valid_pred[mask]
        .mean()
    )

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

print(
    "\n" + "=" * 70
)

print(
    "DISTANCE-BAND ANALYSIS"
)

print(
    "=" * 70
)


valid_analysis = valid_df.copy()

valid_analysis["actual_rate"] = (
    y_valid.values
)

valid_analysis["predicted_rate"] = (
    valid_pred
)


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


for group in (
    valid_analysis[
        "distance_group"
    ].cat.categories
):

    mask = (
        valid_analysis[
            "distance_group"
        ]
        == group
    )

    subset = (
        valid_analysis[mask]
    )

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

print(
    "\n" + "=" * 70
)

print(
    "EQUIPMENT ANALYSIS"
)

print(
    "=" * 70
)


for equipment in sorted(
    valid_analysis[
        "equipment"
    ].unique()
):

    mask = (
        valid_analysis[
            "equipment"
        ]
        == equipment
    )

    subset = (
        valid_analysis[mask]
    )

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
    -
    valid_analysis["actual_rate"]
)

valid_analysis["absolute_error"] = (
    valid_analysis["error"]
    .abs()
)

valid_analysis["percentage_error"] = (
    valid_analysis["absolute_error"]
    /
    valid_analysis["actual_rate"]
    *
    100
)


# ============================================================
# TOP ERRORS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 30 ABSOLUTE ERRORS"
)

print(
    "=" * 70
)


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
    top_errors[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# EXTREME ERROR ANALYSIS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXTREME-RATE ERROR ANALYSIS"
)

print(
    "=" * 70
)


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
        ].to_string(
            index=False
        )
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

print(
    "\n" + "=" * 70
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "=" * 70
)


feature_importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.get_feature_importance(),

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
# ROUTE FEATURE IMPORTANCE SUMMARY
# ============================================================

route_feature_names = [

    "route",

    "route_mean_rate",

    "route_median_rate",

    "route_std_rate",

    "route_sample_count",

    "route_high_rate_pct",

    "route_extreme_rate_pct",

]


print(
    "\n" + "=" * 70
)

print(
    "ROUTE FEATURE IMPORTANCE"
)

print(
    "=" * 70
)


route_importance = (
    feature_importance[
        feature_importance[
            "feature"
        ].isin(
            route_feature_names
        )
    ]
)


print(
    route_importance.to_string(
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


print(
    "\n" + "=" * 70
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 70
)

print(
    "\nRandom split : 80% train / 20% validation"
)

print(
    "Split stratified by "
    f"extreme rate >= ${EXTREME_RATE_THRESHOLD:,}"
)

print(
    f"Training rows   : {len(train_df):,}"
)

print(
    f"Validation rows : {len(valid_df):,}"
)

print(
    f"Extreme train   : {train_extreme:,}"
)

print(
    f"Extreme valid   : {valid_extreme:,}"
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

print(
    "\nFiles created:"
)

print(
    f"  - {MODEL_PATH}"
)

print(
    f"  - {ERROR_PATH}"
)

print(
    "\nDone."
)

print(
    "=" * 70
)