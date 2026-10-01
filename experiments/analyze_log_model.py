import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. Load prepared data
# ============================================================

train_df = pd.read_csv("data/prepared_train.csv")
valid_df = pd.read_csv("data/prepared_validation.csv")

target = "posted_rate"


# ============================================================
# 2. Features
#    Distance only - geo_distance excluded
# ============================================================

features = [
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
    "route"
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]


# ============================================================
# 3. Prepare data
# ============================================================

X_train = train_df[features]
X_valid = valid_df[features]

y_train = train_df[target]
y_valid = valid_df[target]

cat_indices = [
    features.index(col)
    for col in categorical_features
]


# ============================================================
# 4. Transform target
# ============================================================

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)


# ============================================================
# 5. Train Log1p CatBoost model
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOG1P MODEL")
print("=" * 70)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train_log,
    cat_features=cat_indices,
    eval_set=(X_valid, y_valid_log),
    use_best_model=True
)


# ============================================================
# 6. Make predictions
# ============================================================

log_predictions = model.predict(X_valid)

predictions = np.expm1(log_predictions)

# Prevent tiny negative floating-point values
predictions = np.maximum(predictions, 0)


# ============================================================
# 7. Overall metrics
# ============================================================

mae = mean_absolute_error(y_valid, predictions)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5

r2 = r2_score(
    y_valid,
    predictions
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"MAE           : ${mae:,.2f}")
print(f"RMSE          : ${rmse:,.2f}")
print(f"R²            : {r2:.4f}")
print(f"Best iteration: {model.get_best_iteration()}")


# ============================================================
# 8. Create error analysis dataframe
# ============================================================

results = valid_df.copy()

results["predicted_rate"] = predictions

results["error"] = (
    results["predicted_rate"] - results["posted_rate"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["posted_rate"]
    * 100
)


# ============================================================
# 9. Largest errors
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 LARGEST ERRORS")
print("=" * 70)

top_errors = results.sort_values(
    "absolute_error",
    ascending=False
).head(20)

columns_to_show = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "posted_rate",
    "predicted_rate",
    "absolute_error",
    "percentage_error"
]

print(
    top_errors[columns_to_show].to_string(
        index=False
    )
)


# ============================================================
# 10. Error by distance
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY DISTANCE")
print("=" * 70)

distance_bins = [
    0,
    250,
    500,
    750,
    1000,
    1500,
    np.inf
]

distance_labels = [
    "0-250",
    "251-500",
    "501-750",
    "751-1000",
    "1001-1500",
    "1500+"
]

results["distance_group"] = pd.cut(
    results["distance"],
    bins=distance_bins,
    labels=distance_labels,
    include_lowest=True
)

distance_analysis = (
    results
    .groupby("distance_group", observed=False)
    .agg(
        loads=("load_id", "count"),
        mae=("absolute_error", "mean"),
        rmse=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        mean_actual=("posted_rate", "mean"),
        mean_predicted=("predicted_rate", "mean")
    )
    .reset_index()
)

print(
    distance_analysis.to_string(
        index=False
    )
)


# ============================================================
# 11. Error by equipment
# ============================================================

print("\n" + "=" * 70)
print("ERROR BY EQUIPMENT")
print("=" * 70)

equipment_analysis = (
    results
    .groupby("equipment")
    .agg(
        loads=("load_id", "count"),
        mae=("absolute_error", "mean"),
        rmse=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        mean_actual=("posted_rate", "mean"),
        mean_predicted=("predicted_rate", "mean")
    )
    .reset_index()
)

print(
    equipment_analysis.to_string(
        index=False
    )
)


# ============================================================
# 12. High-rate loads
# ============================================================

print("\n" + "=" * 70)
print("HIGH-RATE LOADS (>= $5,000)")
print("=" * 70)

high_rate = results[
    results["posted_rate"] >= 5000
]

if len(high_rate) > 0:

    high_rate_mae = (
        high_rate["absolute_error"].mean()
    )

    high_rate_rmse = np.sqrt(
        np.mean(
            high_rate["error"] ** 2
        )
    )

    print(f"Number of loads: {len(high_rate)}")
    print(f"MAE            : ${high_rate_mae:,.2f}")
    print(f"RMSE           : ${high_rate_rmse:,.2f}")

    print("\nHighest actual rates:")

    print(
        high_rate[
            columns_to_show
        ]
        .sort_values(
            "posted_rate",
            ascending=False
        )
        .head(15)
        .to_string(index=False)
    )

else:

    print("No loads with posted_rate >= $5,000")


# ============================================================
# 13. Prediction bias
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION BIAS")
print("=" * 70)

mean_error = results["error"].mean()

median_error = results["error"].median()

print(
    f"Mean error   : ${mean_error:,.2f}"
)

print(
    f"Median error : ${median_error:,.2f}"
)

if mean_error > 0:
    print("Overall tendency: overprediction")
elif mean_error < 0:
    print("Overall tendency: underprediction")
else:
    print("Overall tendency: no average bias")


# ============================================================
# 14. Feature importance
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance()
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 15. Sample predictions
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample = results[
    [
        "load_id",
        "posted_rate",
        "predicted_rate",
        "absolute_error"
    ]
].head(10)

print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# 16. Save complete error analysis
# ============================================================

results.to_csv(
    "log_model_error_analysis.csv",
    index=False
)

print(
    "\nDetailed error analysis saved to:"
)

print(
    "log_model_error_analysis.csv"
)
