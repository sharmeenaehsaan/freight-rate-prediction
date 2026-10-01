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
#    Using distance only, without geo_distance
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
# 3. Prepare X and y
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
# 4. Function to calculate metrics
# ============================================================

def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)

    rmse = mean_squared_error(
        y_true,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_true,
        predictions
    )

    return mae, rmse, r2


# ============================================================
# 5. Model A - Raw target
# ============================================================

print("\n" + "=" * 70)
print("MODEL A - RAW TARGET")
print("=" * 70)

raw_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

raw_model.fit(
    X_train,
    y_train,
    cat_features=cat_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

raw_predictions = raw_model.predict(X_valid)

raw_mae, raw_rmse, raw_r2 = calculate_metrics(
    y_valid,
    raw_predictions
)

print("\nRaw Target Results")
print("-" * 40)
print(f"MAE           : ${raw_mae:,.2f}")
print(f"RMSE          : ${raw_rmse:,.2f}")
print(f"R²            : {raw_r2:.4f}")
print(f"Best iteration: {raw_model.get_best_iteration()}")


# ============================================================
# 6. Model B - Log-transformed target
# ============================================================

print("\n" + "=" * 70)
print("MODEL B - LOG1P TARGET")
print("=" * 70)

# Transform target
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

log_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

log_model.fit(
    X_train,
    y_train_log,
    cat_features=cat_indices,
    eval_set=(X_valid, y_valid_log),
    use_best_model=True
)

# Predict in log space
log_predictions = log_model.predict(X_valid)

# Convert predictions back to dollar/rate scale
log_predictions_original = np.expm1(log_predictions)

# Prevent any tiny negative floating-point values
log_predictions_original = np.maximum(
    log_predictions_original,
    0
)

log_mae, log_rmse, log_r2 = calculate_metrics(
    y_valid,
    log_predictions_original
)

print("\nLog1p Target Results")
print("-" * 40)
print(f"MAE           : ${log_mae:,.2f}")
print(f"RMSE          : ${log_rmse:,.2f}")
print(f"R²            : {log_r2:.4f}")
print(f"Best iteration: {log_model.get_best_iteration()}")


# ============================================================
# 7. Compare models
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Raw target",
        "Log1p target"
    ],
    "MAE": [
        raw_mae,
        log_mae
    ],
    "RMSE": [
        raw_rmse,
        log_rmse
    ],
    "R2": [
        raw_r2,
        log_r2
    ],
    "Best Iteration": [
        raw_model.get_best_iteration(),
        log_model.get_best_iteration()
    ]
})

print(comparison.to_string(index=False))


# ============================================================
# 8. Differences
# ============================================================

print("\nDifference: Log1p - Raw")
print("-" * 40)

print(
    f"MAE difference : "
    f"${log_mae - raw_mae:,.2f}"
)

print(
    f"RMSE difference: "
    f"${log_rmse - raw_rmse:,.2f}"
)

print(
    f"R² difference  : "
    f"{log_r2 - raw_r2:.4f}"
)


# ============================================================
# 9. Save comparison
# ============================================================

comparison.to_csv(
    "log_target_comparison.csv",
    index=False
)

print(
    "\nResults saved to: "
    "log_target_comparison.csv"
)
