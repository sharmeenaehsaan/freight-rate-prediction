from pathlib import Path
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("data/train-test.csv")

HIGH_RATE_THRESHOLD = 5000

RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TWO-STAGE HIGH-RATE FREIGHT MODEL")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")

# ============================================================
# CLEAN DATA
# ============================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Fix invalid / negative weight values
df.loc[df["weight"] <= 0, "weight"] = np.nan
df["weight"] = df["weight"].fillna(df["weight"].median())

# Market index
df["market_index"] = df["market_index"].fillna(
    df["market_index"].median()
)

# Date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear

# ============================================================
# ROUTE FEATURES
# ============================================================

df["route"] = (
    df["pickup"].astype(str)
    + " -> "
    + df["delivery"].astype(str)
)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["distance_squared"] = df["distance"] ** 2
df["log_distance"] = np.log1p(df["distance"])

df["weight_per_mile"] = (
    df["weight"] / df["distance"].clip(lower=1)
)

df["distance_equipment"] = (
    df["distance"].astype(str)
    + "_"
    + df["equipment"].astype(str)
)

df["pickup_equipment"] = (
    df["pickup"].astype(str)
    + "_"
    + df["equipment"].astype(str)
)

df["delivery_equipment"] = (
    df["delivery"].astype(str)
    + "_"
    + df["equipment"].astype(str)
)

# ============================================================
# HIGH-RATE LABEL
# ============================================================

df["is_high_rate"] = (
    df["posted_rate"] >= HIGH_RATE_THRESHOLD
).astype(int)

print()
print("High-rate distribution:")
print(df["is_high_rate"].value_counts())
print()

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",

    "distance",
    "distance_squared",
    "log_distance",

    "weight",
    "weight_per_mile",

    "market_index",
    "quote_signal",

    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "year",

    "distance_equipment",
    "pickup_equipment",
    "delivery_equipment",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
    "distance_equipment",
    "pickup_equipment",
    "delivery_equipment",
]

X = df[FEATURES].copy()
y = df["posted_rate"].copy()

# CatBoost requires categorical columns to contain strings
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].fillna("Unknown").astype(str)

# Numeric cleanup
for col in FEATURES:
    if col not in CATEGORICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        X[col] = X[col].fillna(X[col].median())

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

print("Train:", X_train.shape)
print("Validation:", X_valid.shape)

# ============================================================
# STAGE 1
# HIGH-RATE CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("STAGE 1: HIGH-RATE CLASSIFIER")
print("=" * 70)

y_class = (y >= HIGH_RATE_THRESHOLD).astype(int)

y_class_train = y_class.loc[X_train.index]
y_class_valid = y_class.loc[X_valid.index]

classifier = CatBoostClassifier(
    iterations=800,
    depth=7,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=100,
    cat_features=CATEGORICAL_FEATURES,
    auto_class_weights="Balanced",
)

classifier.fit(
    X_train,
    y_class_train,
    eval_set=(X_valid, y_class_valid),
    early_stopping_rounds=100,
)

high_probability = classifier.predict_proba(X_valid)[:, 1]

high_prediction = (
    high_probability >= 0.50
).astype(int)

print()
print("Classification results:")
print(confusion_matrix(y_class_valid, high_prediction))
print()
print(
    classification_report(
        y_class_valid,
        high_prediction,
        digits=4,
    )
)

# ============================================================
# STAGE 2A
# NORMAL-RATE REGRESSOR
# ============================================================

print()
print("=" * 70)
print("STAGE 2A: NORMAL-RATE REGRESSOR")
print("=" * 70)

normal_train_mask = y_train < HIGH_RATE_THRESHOLD

X_normal_train = X_train.loc[normal_train_mask]
y_normal_train = y_train.loc[normal_train_mask]

normal_model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=RANDOM_STATE,
    verbose=100,
    cat_features=CATEGORICAL_FEATURES,
)

normal_model.fit(
    X_normal_train,
    y_normal_train,
)

normal_predictions = normal_model.predict(X_valid)

# ============================================================
# STAGE 2B
# HIGH-RATE REGRESSOR
# ============================================================

print()
print("=" * 70)
print("STAGE 2B: HIGH-RATE REGRESSOR")
print("=" * 70)

high_train_mask = y_train >= HIGH_RATE_THRESHOLD

X_high_train = X_train.loc[high_train_mask]
y_high_train = y_train.loc[high_train_mask]

print(f"High-rate training loads: {len(X_high_train)}")

high_model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=RANDOM_STATE,
    verbose=100,
    cat_features=CATEGORICAL_FEATURES,
)

high_model.fit(
    X_high_train,
    y_high_train,
)

high_predictions = high_model.predict(X_valid)

# ============================================================
# COMBINE MODELS
# ============================================================

# Instead of a hard classifier decision, blend the two
# predictions using the probability of being high-rate.

final_predictions = (
    (1 - high_probability) * normal_predictions
    + high_probability * high_predictions
)

# Prevent negative predictions
final_predictions = np.maximum(final_predictions, 0)

# ============================================================
# RESULTS
# ============================================================

actual = y_valid.values

mae = mean_absolute_error(actual, final_predictions)

rmse = np.sqrt(
    mean_squared_error(actual, final_predictions)
)

r2 = r2_score(actual, final_predictions)

print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ============================================================
# DETAILED VALIDATION DATASET
# ============================================================

results = X_valid.copy()

results["posted_rate"] = actual
results["predicted_rate"] = final_predictions
results["high_probability"] = high_probability

results["absolute_error"] = (
    results["posted_rate"]
    - results["predicted_rate"]
).abs()

results["error"] = (
    results["predicted_rate"]
    - results["posted_rate"]
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["posted_rate"].clip(lower=1)
) * 100

results["distance_group"] = pd.cut(
    results["distance"],
    bins=[0, 250, 500, 750, 1000, 1500, np.inf],
    labels=[
        "0-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1500+",
    ],
)

# ============================================================
# HIGH-RATE PERFORMANCE
# ============================================================

high_results = results[
    results["posted_rate"] >= HIGH_RATE_THRESHOLD
]

normal_results = results[
    results["posted_rate"] < HIGH_RATE_THRESHOLD
]

print()
print("=" * 70)
print("HIGH-RATE PERFORMANCE")
print("=" * 70)

if len(high_results) > 0:

    high_mae = mean_absolute_error(
        high_results["posted_rate"],
        high_results["predicted_rate"],
    )

    high_rmse = np.sqrt(
        mean_squared_error(
            high_results["posted_rate"],
            high_results["predicted_rate"],
        )
    )

    print(f"High-rate loads: {len(high_results):,}")
    print(f"High-rate MAE : ${high_mae:,.2f}")
    print(f"High-rate RMSE: ${high_rmse:,.2f}")

print()
print("=" * 70)
print("NORMAL-RATE PERFORMANCE")
print("=" * 70)

normal_mae = mean_absolute_error(
    normal_results["posted_rate"],
    normal_results["predicted_rate"],
)

normal_rmse = np.sqrt(
    mean_squared_error(
        normal_results["posted_rate"],
        normal_results["predicted_rate"],
    )
)

print(f"Normal loads: {len(normal_results):,}")
print(f"Normal MAE : ${normal_mae:,.2f}")
print(f"Normal RMSE: ${normal_rmse:,.2f}")

# ============================================================
# HIGH-RATE BANDS
# ============================================================

print()
print("=" * 70)
print("HIGH-RATE ERROR BY BAND")
print("=" * 70)

high_results["rate_band"] = pd.cut(
    high_results["posted_rate"],
    bins=[
        5000,
        7500,
        10000,
        15000,
        20000,
        np.inf,
    ],
    labels=[
        "$5k-$7.5k",
        "$7.5k-$10k",
        "$10k-$15k",
        "$15k-$20k",
        "$20k+",
    ],
)

band_results = []

for band, group in high_results.groupby(
    "rate_band",
    observed=False,
):

    if len(group) == 0:
        continue

    band_results.append({
        "rate_band": str(band),
        "loads": len(group),
        "mae": mean_absolute_error(
            group["posted_rate"],
            group["predicted_rate"],
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                group["posted_rate"],
                group["predicted_rate"],
            )
        ),
        "mean_actual": group["posted_rate"].mean(),
        "mean_predicted": group["predicted_rate"].mean(),
    })

band_df = pd.DataFrame(band_results)

print(band_df.to_string(index=False))

# ============================================================
# ERROR BY DISTANCE
# ============================================================

print()
print("=" * 70)
print("ERROR BY DISTANCE")
print("=" * 70)

distance_results = []

for group_name, group in results.groupby(
    "distance_group",
    observed=False,
):

    distance_results.append({
        "distance_group": str(group_name),
        "loads": len(group),
        "mae": mean_absolute_error(
            group["posted_rate"],
            group["predicted_rate"],
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                group["posted_rate"],
                group["predicted_rate"],
            )
        ),
        "mean_actual": group["posted_rate"].mean(),
        "mean_predicted": group["predicted_rate"].mean(),
    })

distance_df = pd.DataFrame(distance_results)

print(distance_df.to_string(index=False))

# ============================================================
# TOP 20 ERRORS
# ============================================================

print()
print("=" * 70)
print("TOP 20 LARGEST ERRORS")
print("=" * 70)

top_errors = results.sort_values(
    "absolute_error",
    ascending=False,
).head(20)

print(
    top_errors[
        [
            "posted_rate",
            "predicted_rate",
            "absolute_error",
            "percentage_error",
            "distance",
            "equipment",
            "route",
        ]
    ].to_string()
)

# ============================================================
# SAVE RESULTS
# ============================================================

output_file = "high_rate_model_error_analysis.csv"

results.to_csv(
    output_file,
    index=False,
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Saved detailed results to: {output_file}")
# Extreme-rate diagnostic

df["is_extreme"] = df["posted_rate"] >= 7500

print("\n" + "=" * 70)
print("EXTREME-RATE DIAGNOSTIC")
print("=" * 70)

extreme = df[df["posted_rate"] >= 7500].copy()

print(f"\nExtreme loads (>= $7,500): {len(extreme)}")
print(f"Percentage: {len(extreme) / len(df) * 100:.2f}%")

print("\nExtreme target statistics:")
print(extreme["posted_rate"].describe())

print("\nExtreme loads by equipment:")
print(
    pd.crosstab(
        df["equipment"],
        df["is_extreme"],
        normalize="index"
    ).round(4)
)
df["distance_group"] = pd.cut(
    df["distance"],
    bins=[0, 250, 500, 750, 1000, 1500, float("inf")],
    labels=[
        "0-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1500+"
    ]
)
print("\nExtreme loads by distance:")
print(
    pd.crosstab(
        df["distance_group"],
        df["is_extreme"],
        normalize="index"
    ).round(4)
)

print("\nExtreme loads by month:")
print(
    pd.crosstab(
        df["month"],
        df["is_extreme"],
        normalize="index"
    ).round(4)
)

print("\nExtreme loads by route:")
route_stats = (
    df.groupby("route")
      .agg(
          loads=("posted_rate", "size"),
          extreme_loads=("is_extreme", "sum"),
          avg_rate=("posted_rate", "mean"),
          max_rate=("posted_rate", "max")
      )
)

route_stats["extreme_percentage"] = (
    route_stats["extreme_loads"] /
    route_stats["loads"] * 100
)

print(
    route_stats
    .sort_values("extreme_loads", ascending=False)
    .head(30)
)

print("\nExtreme loads by pickup:")
print(
    df[df["is_extreme"]]
    .groupby("pickup")
    .size()
    .sort_values(ascending=False)
    .head(20)
)

print("\nExtreme loads by delivery:")
print(
    df[df["is_extreme"]]
    .groupby("delivery")
    .size()
    .sort_values(ascending=False)
    .head(20)
)

print("\nExtreme loads:")
print(
    extreme[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(50)
)

print("\n" + "=" * 70)
print("END EXTREME-RATE DIAGNOSTIC")
print("=" * 70)