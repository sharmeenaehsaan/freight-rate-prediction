from pathlib import Path

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 1. FILES
# =========================================================

DATA_DIR = Path("data")

TRAIN_FILE = DATA_DIR / "prepared_train.csv"
VALID_FILE = DATA_DIR / "prepared_validation.csv"
MODEL_FILE = Path("catboost_baseline.cbm")


# =========================================================
# 2. LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))


# =========================================================
# 3. FEATURES
# =========================================================

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
    "geo_distance",
]


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


X_train = train_df[FEATURES].copy()
X_valid = valid_df[FEATURES].copy()

y_train = train_df["posted_rate"]
y_valid = valid_df["posted_rate"]


for column in CATEGORICAL_FEATURES:
    X_train[column] = X_train[column].astype(str)
    X_valid[column] = X_valid[column].astype(str)


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

model = CatBoostRegressor()

model.load_model(MODEL_FILE)

print("\nLoaded model:", MODEL_FILE)


# =========================================================
# 5. PREDICTIONS
# =========================================================

predictions = model.predict(X_valid)


# =========================================================
# 6. CREATE ERROR TABLE
# =========================================================

results = valid_df[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "posted_rate",
    ]
].copy()

results["predicted_rate"] = predictions

results["error"] = (
    results["predicted_rate"]
    - results["posted_rate"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results["percentage_error"] = (
    results["absolute_error"]
    / results["posted_rate"]
    * 100
)


# =========================================================
# 7. OVERALL ERROR
# =========================================================

mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5


print("\n==============================")
print("OVERALL ERROR")
print("==============================")

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")


# =========================================================
# 8. LARGEST ERRORS
# =========================================================

print("\n==============================")
print("TOP 20 LARGEST ERRORS")
print("==============================")

largest_errors = results.sort_values(
    "absolute_error",
    ascending=False
).head(20)

print(
    largest_errors[
        [
            "load_id",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "posted_rate",
            "predicted_rate",
            "absolute_error",
        ]
    ].to_string(index=False)
)


# =========================================================
# 9. ERROR BY EQUIPMENT
# =========================================================

print("\n==============================")
print("ERROR BY EQUIPMENT")
print("==============================")

equipment_error = (
    results
    .groupby("equipment")
    .agg(
        rows=("load_id", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        mean_actual=("posted_rate", "mean"),
        mean_predicted=("predicted_rate", "mean"),
    )
    .round(2)
)

print(equipment_error)


# =========================================================
# 10. ERROR BY DISTANCE
# =========================================================

print("\n==============================")
print("ERROR BY DISTANCE RANGE")
print("==============================")

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

distance_error = (
    results
    .groupby("distance_group", observed=True)
    .agg(
        rows=("load_id", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "error",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        mean_actual=("posted_rate", "mean"),
        mean_predicted=("predicted_rate", "mean"),
    )
    .round(2)
)

print(distance_error)


# =========================================================
# 11. HIGH-RATE LOADS
# =========================================================

print("\n==============================")
print("HIGH-RATE LOAD ANALYSIS")
print("==============================")

high_rate = results[
    results["posted_rate"] >= 5000
]

if len(high_rate) > 0:

    high_rate_mae = mean_absolute_error(
        high_rate["posted_rate"],
        high_rate["predicted_rate"],
    )

    high_rate_rmse = mean_squared_error(
        high_rate["posted_rate"],
        high_rate["predicted_rate"],
    ) ** 0.5

    print("High-rate rows:", len(high_rate))
    print(f"MAE:  ${high_rate_mae:,.2f}")
    print(f"RMSE: ${high_rate_rmse:,.2f}")

else:
    print("No validation loads above $5,000.")


# =========================================================
# 12. BIAS
# =========================================================

print("\n==============================")
print("PREDICTION BIAS")
print("==============================")

mean_error = results["error"].mean()

print(
    f"Mean error (prediction - actual): "
    f"${mean_error:,.2f}"
)

if mean_error > 0:
    print("Overall tendency: overprediction")
elif mean_error < 0:
    print("Overall tendency: underprediction")
else:
    print("Overall tendency: neutral")


# =========================================================
# 13. FEATURE IMPORTANCE
# =========================================================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

importance = model.get_feature_importance()

feature_importance = (
    pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": importance,
        }
    )
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


# =========================================================
# 14. SAVE ERROR DATA
# =========================================================

results.to_csv(
    "baseline_error_analysis.csv",
    index=False
)

print("\nSaved:")
print("baseline_error_analysis.csv")
