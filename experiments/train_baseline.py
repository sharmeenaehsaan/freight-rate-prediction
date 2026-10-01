from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. FILES
# =========================================================

DATA_DIR = Path("data")

TRAIN_FILE = DATA_DIR / "prepared_train.csv"
VALID_FILE = DATA_DIR / "prepared_validation.csv"


# =========================================================
# 2. LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))


# =========================================================
# 3. TARGET
# =========================================================

TARGET = "posted_rate"


# =========================================================
# 4. FEATURES
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


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET]


# =========================================================
# 5. CATEGORICAL FEATURES
# =========================================================

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


# Convert categorical columns to string.
# This makes sure CatBoost receives them correctly.

X_train = X_train.copy()
X_valid = X_valid.copy()

for column in CATEGORICAL_FEATURES:
    X_train[column] = X_train[column].astype(str)
    X_valid[column] = X_valid[column].astype(str)


# CatBoost wants the column names or column positions
# of categorical features.

cat_feature_indices = [
    X_train.columns.get_loc(column)
    for column in CATEGORICAL_FEATURES
]


# =========================================================
# 6. CREATE MODEL
# =========================================================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
)


# =========================================================
# 7. TRAIN
# =========================================================

print("\nStarting CatBoost training...\n")

model.fit(
    X_train,
    y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
)


# =========================================================
# 8. PREDICT
# =========================================================

print("\nGenerating validation predictions...")

predictions = model.predict(X_valid)


# =========================================================
# 9. EVALUATE
# =========================================================

mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5

r2 = r2_score(
    y_valid,
    predictions
)


print("\n==============================")
print("BASELINE MODEL RESULTS")
print("==============================")

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")


# =========================================================
# 10. SAMPLE PREDICTIONS
# =========================================================

results = pd.DataFrame(
    {
        "actual_rate": y_valid.values,
        "predicted_rate": predictions,
    }
)

print("\nSample predictions:")
print(results.head(10).to_string(index=False))


# =========================================================
# 11. SAVE MODEL
# =========================================================

model.save_model(
    "catboost_baseline.cbm"
)

print("\nModel saved as:")
print("catboost_baseline.cbm")