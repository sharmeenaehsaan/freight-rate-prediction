from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("data/train-test.csv")

HIGH_RATE_THRESHOLD = 5000
HIGH_RATE_WEIGHT = 2.0

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("WEIGHTED LOG1P CATBOOST MODEL")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# DATA CLEANING
# ============================================================

df["date"] = pd.to_datetime(df["date"])

# Negative weights are treated as sign errors
df["weight"] = df["weight"].abs()

# Median imputation
weight_median = df["weight"].median()
market_median = df["market_index"].median()

df["weight"] = df["weight"].fillna(weight_median)
df["market_index"] = df["market_index"].fillna(market_median)


# ============================================================
# DATE FEATURES
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear


# ============================================================
# ROUTE FEATURE
# ============================================================

df["route"] = (
    df["pickup"].astype(str)
    + " -> "
    + df["delivery"].astype(str)
)


# ============================================================
# FEATURES
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
# TIME-BASED SPLIT
# ============================================================

train_mask = (
    (df["date"] >= "2025-01-01")
    & (df["date"] <= "2025-09-30")
)

validation_mask = (
    (df["date"] >= "2025-10-01")
    & (df["date"] <= "2025-10-31")
)

train_df = df.loc[train_mask].copy()
validation_df = df.loc[validation_mask].copy()

print()
print("TIME-BASED SPLIT")
print("-" * 70)
print(f"Training rows   : {len(train_df):,}")
print(f"Validation rows : {len(validation_df):,}")


# ============================================================
# TARGET
# ============================================================

X_train = train_df[FEATURES].copy()
X_valid = validation_df[FEATURES].copy()

y_train = train_df["posted_rate"].copy()
y_valid = validation_df["posted_rate"].copy()

# Log transform
y_train_log = np.log1p(y_train)


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

sample_weights = np.where(
    y_train >= HIGH_RATE_THRESHOLD,
    HIGH_RATE_WEIGHT,
    1.0
)

print()
print("SAMPLE WEIGHTS")
print("-" * 70)

print(
    f"Normal loads (< ${HIGH_RATE_THRESHOLD:,}): "
    f"{(sample_weights == 1.0).sum():,}"
)

print(
    f"High-rate loads (>= ${HIGH_RATE_THRESHOLD:,}): "
    f"{(sample_weights == HIGH_RATE_WEIGHT).sum():,}"
)

print(f"High-rate weight: {HIGH_RATE_WEIGHT}")


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING WEIGHTED LOG1P CATBOOST")
print("=" * 70)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RANDOM_SEED,
    verbose=100,
)

model.fit(
    X_train,
    y_train_log,
    cat_features=CATEGORICAL_FEATURES,
    sample_weight=sample_weights,
    eval_set=(X_valid, np.log1p(y_valid)),
    use_best_model=True,
)


# ============================================================
# PREDICTION
# ============================================================

pred_log = model.predict(X_valid)

# Convert back from log scale
predictions = np.expm1(pred_log)

# Prevent impossible negative predictions
predictions = np.maximum(predictions, 0)


# ============================================================
# OVERALL METRICS
# ============================================================

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
r2 = r2_score(y_valid, predictions)

print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# ERROR DATAFRAME
# ============================================================

results = validation_df[
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
        "posted_rate",
    ]
].copy()

results["predicted_rate"] = predictions

results["absolute_error"] = (
    results["posted_rate"] - results["predicted_rate"]
).abs()

results["error"] = (
    results["posted_rate"] - results["predicted_rate"]
)


# ============================================================
# HIGH-RATE PERFORMANCE
# ============================================================

high_rate_mask = results["posted_rate"] >= HIGH_RATE_THRESHOLD

high_results = results.loc[high_rate_mask].copy()
normal_results = results.loc[~high_rate_mask].copy()

if len(high_results) > 0:

    high_mae = mean_absolute_error(
        high_results["posted_rate"],
        high_results["predicted_rate"]
    )

    high_rmse = np.sqrt(
        mean_squared_error(
            high_results["posted_rate"],
            high_results["predicted_rate"]
        )
    )

    print()
    print("=" * 70)
    print("HIGH-RATE PERFORMANCE")
    print("=" * 70)

    print(f"High-rate loads : {len(high_results):,}")
    print(f"High-rate MAE   : ${high_mae:,.2f}")
    print(f"High-rate RMSE  : ${high_rmse:,.2f}")


# ============================================================
# NORMAL PERFORMANCE
# ============================================================

normal_mae = mean_absolute_error(
    normal_results["posted_rate"],
    normal_results["predicted_rate"]
)

normal_rmse = np.sqrt(
    mean_squared_error(
        normal_results["posted_rate"],
        normal_results["predicted_rate"]
    )
)

print()
print("=" * 70)
print("NORMAL-RATE PERFORMANCE")
print("=" * 70)

print(f"Normal loads : {len(normal_results):,}")
print(f"Normal MAE   : ${normal_mae:,.2f}")
print(f"Normal RMSE  : ${normal_rmse:,.2f}")


# ============================================================
# EXTREME-RATE PERFORMANCE
# ============================================================

EXTREME_THRESHOLD = 7500

extreme_mask = results["posted_rate"] >= EXTREME_THRESHOLD
extreme_results = results.loc[extreme_mask].copy()

print()
print("=" * 70)
print("EXTREME-RATE PERFORMANCE")
print("=" * 70)

print(f"Extreme loads (>= ${EXTREME_THRESHOLD:,}) : {len(extreme_results):,}")

if len(extreme_results) > 0:

    extreme_mae = mean_absolute_error(
        extreme_results["posted_rate"],
        extreme_results["predicted_rate"]
    )

    extreme_rmse = np.sqrt(
        mean_squared_error(
            extreme_results["posted_rate"],
            extreme_results["predicted_rate"]
        )
    )

    print(f"Extreme MAE  : ${extreme_mae:,.2f}")
    print(f"Extreme RMSE : ${extreme_rmse:,.2f}")


# ============================================================
# ERROR BY DISTANCE
# ============================================================

results["distance_group"] = pd.cut(
    results["distance"],
    bins=[0, 250, 500, 750, 1000, 1500, float("inf")],
    labels=[
        "0-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        "1500+",
    ],
)

print()
print("=" * 70)
print("ERROR BY DISTANCE")
print("=" * 70)

distance_summary = (
    results.groupby("distance_group", observed=False)
    .apply(
        lambda x: pd.Series({
            "loads": len(x),
            "mae": mean_absolute_error(
                x["posted_rate"],
                x["predicted_rate"]
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    x["posted_rate"],
                    x["predicted_rate"]
                )
            ),
            "mean_actual": x["posted_rate"].mean(),
            "mean_predicted": x["predicted_rate"].mean(),
        }),
        include_groups=False,
    )
)

print(distance_summary)


# ============================================================
# HIGH-RATE ERROR BANDS
# ============================================================

if len(high_results) > 0:

    high_results["rate_band"] = pd.cut(
        high_results["posted_rate"],
        bins=[5000, 7500, 10000, 15000, 20000, float("inf")],
        labels=[
            "$5k-$7.5k",
            "$7.5k-$10k",
            "$10k-$15k",
            "$15k-$20k",
            "$20k+",
        ],
        include_lowest=True,
    )

    print()
    print("=" * 70)
    print("HIGH-RATE ERROR BY BAND")
    print("=" * 70)

    band_rows = []

    for band, group in high_results.groupby(
        "rate_band",
        observed=False
    ):

        if len(group) == 0:
            continue

        band_rows.append({
            "rate_band": str(band),
            "loads": len(group),
            "mae": mean_absolute_error(
                group["posted_rate"],
                group["predicted_rate"]
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    group["posted_rate"],
                    group["predicted_rate"]
                )
            ),
            "mean_actual": group["posted_rate"].mean(),
            "mean_predicted": group["predicted_rate"].mean(),
        })

    print(pd.DataFrame(band_rows).to_string(index=False))


# ============================================================
# TOP ERRORS
# ============================================================

top_errors = results.sort_values(
    "absolute_error",
    ascending=False
).head(20)

print()
print("=" * 70)
print("TOP 20 LARGEST ERRORS")
print("=" * 70)

print(
    top_errors[
        [
            "load_id",
            "posted_rate",
            "predicted_rate",
            "absolute_error",
            "distance",
            "equipment",
            "pickup",
            "delivery",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "weighted_log_model_error_analysis.csv",
    index=False
)

model.save_model(
    "catboost_weighted_log_model.cbm"
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Saved error analysis to: "
    "weighted_log_model_error_analysis.csv"
)

print(
    "Saved model to: "
    "catboost_weighted_log_model.cbm"
)