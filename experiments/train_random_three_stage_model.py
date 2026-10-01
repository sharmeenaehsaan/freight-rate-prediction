

from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train-test.csv")

MODEL_CLASSIFIER_PATH = "catboost_random_three_stage_classifier.cbm"
MODEL_NORMAL_PATH = "catboost_random_three_stage_normal.cbm"
MODEL_HIGH_PATH = "catboost_random_three_stage_high.cbm"
MODEL_EXTREME_PATH = "catboost_random_three_stage_extreme.cbm"

ERROR_ANALYSIS_PATH = "random_three_stage_error_analysis.csv"

RANDOM_STATE = 42

HIGH_THRESHOLD = 5000
EXTREME_THRESHOLD = 7500


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_metrics(title, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse_value = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{title}")
    print("-" * 60)
    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse_value:,.2f}")
    print(f"R²   : {r2:.4f}")

    return mae, rmse_value, r2


def create_features(df, weight_median=None, market_median=None):
    """
    Create model features.

    Important:
    - geo_distance is intentionally NOT used.
    - rate_per_distance is NOT used because it depends on the target.
    - route-history features are NOT used.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    # Negative weights appear to be sign errors.
    df["weight"] = df["weight"].abs()

    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    if weight_median is None:
        weight_median = df["weight"].median()

    if market_median is None:
        market_median = df["market_index"].median()

    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(market_median)

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

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

    return df, weight_median, market_median


# ============================================================
# FEATURE LIST
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("RANDOM-SPLIT THREE-STAGE LOG1P CATBOOST MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified.")


# ============================================================
# TARGET SUMMARY
# ============================================================

y = df["posted_rate"].astype(float)

print("\nTarget distribution:")
print(f"Rows   : {len(df):,}")
print(f"Mean   : ${y.mean():,.2f}")
print(f"Median : ${y.median():,.2f}")
print(f"Min    : ${y.min():,.2f}")
print(f"Max    : ${y.max():,.2f}")

high_count = (y >= HIGH_THRESHOLD).sum()
extreme_count = (y >= EXTREME_THRESHOLD).sum()

print(
    f"\nHigh-rate >= $5,000: "
    f"{high_count:,} ({high_count / len(df) * 100:.2f}%)"
)

print(
    f"Extreme >= $7,500: "
    f"{extreme_count:,} ({extreme_count / len(df) * 100:.2f}%)"
)


# ============================================================
# CREATE THREE CLASSES
# ============================================================

def make_rate_class(rate):
    if rate < HIGH_THRESHOLD:
        return 0
    elif rate < EXTREME_THRESHOLD:
        return 1
    else:
        return 2


df["rate_class"] = df["posted_rate"].apply(make_rate_class)

print("\nRate classes:")
print("0 = Normal   (< $5,000)")
print("1 = High     ($5,000 - $7,500)")
print("2 = Extreme  (>= $7,500)")

print("\nFull dataset class distribution:")

class_counts = df["rate_class"].value_counts().sort_index()

for cls, count in class_counts.items():
    percentage = count / len(df) * 100

    if cls == 0:
        label = "Normal"
    elif cls == 1:
        label = "High"
    else:
        label = "Extreme"

    print(
        f"  {cls} - {label:<8}: "
        f"{count:,} ({percentage:.2f}%)"
    )


# ============================================================
# RANDOM 80/20 SPLIT
# ============================================================

print("\nCreating random 80/20 split...")

train_df, valid_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["rate_class"],
)

train_df = train_df.copy()
valid_df = valid_df.copy()

print(f"Training rows   : {len(train_df):,}")
print(f"Validation rows : {len(valid_df):,}")


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

for name, split_df in [
    ("Training", train_df),
    ("Validation", valid_df),
]:

    print(f"\n{name}:")

    counts = split_df["rate_class"].value_counts().sort_index()

    for cls in [0, 1, 2]:
        count = counts.get(cls, 0)

        if cls == 0:
            label = "Normal"
        elif cls == 1:
            label = "High"
        else:
            label = "Extreme"

        print(
            f"  {label:<8}: "
            f"{count:,} ({count / len(split_df) * 100:.2f}%)"
        )


# ============================================================
# EXTREME DISTRIBUTION
# ============================================================

train_extreme = (
    train_df["posted_rate"] >= EXTREME_THRESHOLD
).sum()

valid_extreme = (
    valid_df["posted_rate"] >= EXTREME_THRESHOLD
).sum()

print("\nExtreme-rate distribution:")
print(
    f"Training   >= $7,500: "
    f"{train_extreme:,} "
    f"({train_extreme / len(train_df) * 100:.2f}%)"
)

print(
    f"Validation >= $7,500: "
    f"{valid_extreme:,} "
    f"({valid_extreme / len(valid_df) * 100:.2f}%)"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating model features...")

train_df, weight_median, market_median = create_features(
    train_df
)

valid_df, _, _ = create_features(
    valid_df,
    weight_median=weight_median,
    market_median=market_median,
)

print("\nTraining medians:")
print(f"Weight median       : {weight_median}")
print(f"Market index median : {market_median}")


# ============================================================
# MODEL MATRICES
# ============================================================

X_train = train_df[FEATURES].copy()
X_valid = valid_df[FEATURES].copy()

y_train = train_df["posted_rate"].astype(float)
y_valid = valid_df["posted_rate"].astype(float)

class_train = train_df["rate_class"].astype(int)
class_valid = valid_df["rate_class"].astype(int)


# ============================================================
# CHECK FEATURES
# ============================================================

print("\nChecking model features...")

print(
    f"Training missing values   : "
    f"{X_train.isna().sum().sum()}"
)

print(
    f"Validation missing values : "
    f"{X_valid.isna().sum().sum()}"
)

if X_train.isna().sum().sum() > 0:
    raise ValueError("Training data still contains missing values.")

if X_valid.isna().sum().sum() > 0:
    raise ValueError("Validation data still contains missing values.")


# ============================================================
# STAGE 1
# THREE-CLASS CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 - THREE-CLASS RATE CLASSIFIER")
print("=" * 70)

classifier = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=RANDOM_STATE,
    verbose=100,
    allow_writing_files=False,
    auto_class_weights="Balanced",
)

print("\nTraining classifier...")

classifier.fit(
    X_train,
    class_train,
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_valid, class_valid),
    use_best_model=True,
    early_stopping_rounds=80,
)

print("\nClassifier training complete.")

print(
    f"Best classifier iteration: "
    f"{classifier.get_best_iteration()}"
)


# ============================================================
# CLASSIFIER VALIDATION
# ============================================================

class_pred = classifier.predict(X_valid)

class_pred = np.asarray(class_pred).reshape(-1).astype(int)

print("\nCLASSIFIER RESULTS")
print("-" * 60)

print(
    classification_report(
        class_valid,
        class_pred,
        target_names=[
            "Normal",
            "High",
            "Extreme",
        ],
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        class_valid,
        class_pred
    )
)


# ============================================================
# STAGE 2
# SEPARATE REGRESSORS
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 - SEPARATE LOG1P REGRESSORS")
print("=" * 70)


def train_regressor(
    class_number,
    class_name,
    model_path,
):
    """
    Train one Log1p CatBoost regressor for one rate class.
    """

    train_mask = (
        train_df["rate_class"] == class_number
    )

    valid_mask = (
        valid_df["rate_class"] == class_number
    )

    X_tr = X_train.loc[train_mask]
    X_va = X_valid.loc[valid_mask]

    y_tr = y_train.loc[train_mask]
    y_va = y_valid.loc[valid_mask]

    print("\n")
    print("-" * 70)
    print(f"TRAINING {class_name.upper()} REGRESSOR")
    print("-" * 70)

    print(f"Training rows   : {len(X_tr):,}")
    print(f"Validation rows : {len(X_va):,}")

    if len(X_tr) == 0:
        raise ValueError(
            f"No training rows found for class {class_number}."
        )

    if len(X_va) == 0:
        raise ValueError(
            f"No validation rows found for class {class_number}."
        )

    print(
        f"Actual training range: "
        f"${y_tr.min():,.2f} - ${y_tr.max():,.2f}"
    )

    print(
        f"Actual validation range: "
        f"${y_va.min():,.2f} - ${y_va.max():,.2f}"
    )

    # --------------------------------------------------------
    # Log1p target
    # --------------------------------------------------------

    y_tr_log = np.log1p(y_tr)
    y_va_log = np.log1p(y_va)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )

    print("\nTraining...")

    model.fit(
        X_tr,
        y_tr_log,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_va, y_va_log),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    print("\nTraining complete.")

    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )

    model.save_model(model_path)

    print(
        f"Model saved to: {model_path}"
    )

    # --------------------------------------------------------
    # Evaluate this individual model
    # --------------------------------------------------------

    pred_log = model.predict(X_va)

    predictions = np.expm1(pred_log)

    predictions = np.maximum(
        predictions,
        0
    )

    print_metrics(
        f"{class_name} REGRESSOR",
        y_va,
        predictions,
    )

    return model


# ============================================================
# TRAIN NORMAL MODEL
# ============================================================

normal_model = train_regressor(
    class_number=0,
    class_name="Normal",
    model_path=MODEL_NORMAL_PATH,
)


# ============================================================
# TRAIN HIGH MODEL
# ============================================================

high_model = train_regressor(
    class_number=1,
    class_name="High-rate",
    model_path=MODEL_HIGH_PATH,
)


# ============================================================
# TRAIN EXTREME MODEL
# ============================================================

extreme_model = train_regressor(
    class_number=2,
    class_name="Extreme",
    model_path=MODEL_EXTREME_PATH,
)


# ============================================================
# GENERATE PREDICTIONS FROM EACH REGRESSOR
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING THREE-STAGE PREDICTIONS")
print("=" * 70)


# ------------------------------------------------------------
# Get predictions from all three models for every validation row
# ------------------------------------------------------------

normal_pred_all = np.expm1(
    normal_model.predict(X_valid)
)

high_pred_all = np.expm1(
    high_model.predict(X_valid)
)

extreme_pred_all = np.expm1(
    extreme_model.predict(X_valid)
)

normal_pred_all = np.maximum(
    normal_pred_all,
    0
)

high_pred_all = np.maximum(
    high_pred_all,
    0
)

extreme_pred_all = np.maximum(
    extreme_pred_all,
    0
)


# ============================================================
# TWO PREDICTION MODES
# ============================================================

# ------------------------------------------------------------
# MODE A
# Hard classification
# ------------------------------------------------------------

hard_predictions = np.where(
    class_pred == 0,
    normal_pred_all,
    np.where(
        class_pred == 1,
        high_pred_all,
        extreme_pred_all,
    )
)


# ------------------------------------------------------------
# MODE B
# Probability-weighted prediction
#
# This is included as a diagnostic.
# It can be smoother than hard classification.
# ------------------------------------------------------------

class_probabilities = classifier.predict_proba(
    X_valid
)

soft_predictions = (
    class_probabilities[:, 0] * normal_pred_all
    + class_probabilities[:, 1] * high_pred_all
    + class_probabilities[:, 2] * extreme_pred_all
)

soft_predictions = np.maximum(
    soft_predictions,
    0
)


# ============================================================
# EVALUATE HARD CLASSIFICATION VERSION
# ============================================================

print("\n")
print("=" * 70)
print("THREE-STAGE HARD CLASSIFICATION RESULTS")
print("=" * 70)

hard_mae, hard_rmse, hard_r2 = print_metrics(
    "OVERALL",
    y_valid,
    hard_predictions,
)


# ============================================================
# EVALUATE SOFT PROBABILITY VERSION
# ============================================================

print("\n")
print("=" * 70)
print("THREE-STAGE SOFT PROBABILITY RESULTS")
print("=" * 70)

soft_mae, soft_rmse, soft_r2 = print_metrics(
    "OVERALL",
    y_valid,
    soft_predictions,
)


# ============================================================
# SELECT BETTER INTERNAL THREE-STAGE VERSION
# ============================================================

if hard_mae <= soft_mae:
    final_predictions = hard_predictions
    selected_mode = "Hard classification"
    selected_mae = hard_mae
    selected_rmse = hard_rmse
    selected_r2 = hard_r2
else:
    final_predictions = soft_predictions
    selected_mode = "Soft probability"
    selected_mae = soft_mae
    selected_rmse = soft_rmse
    selected_r2 = soft_r2


# ============================================================
# HIGH-RATE ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("HIGH-RATE ANALYSIS")
print("=" * 70)

high_mask = y_valid >= HIGH_THRESHOLD

high_actual = y_valid[high_mask]
high_pred = final_predictions[high_mask]

print(
    f"\nHigh-rate validation rows "
    f"(>= $5,000): {len(high_actual):,}"
)

print_metrics(
    "High-rate >= $5,000",
    high_actual,
    high_pred,
)


# ============================================================
# EXTREME ANALYSIS
# ============================================================

extreme_mask = (
    y_valid >= EXTREME_THRESHOLD
)

extreme_actual = y_valid[extreme_mask]
extreme_pred = final_predictions[extreme_mask]

print(
    f"\nExtreme validation rows "
    f"(>= $7,500): {len(extreme_actual):,}"
)

print_metrics(
    "Extreme >= $7,500",
    extreme_actual,
    extreme_pred,
)


# ============================================================
# RATE BAND ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("RATE-BAND ANALYSIS")
print("=" * 70)

rate_bands = [
    ("<$5k", 0, 5000),
    ("$5k-$7.5k", 5000, 7500),
    ("$7.5k-$10k", 7500, 10000),
    ("$10k-$15k", 10000, 15000),
    ("$15k-$20k", 15000, 20000),
    ("$20k+", 20000, float("inf")),
]

for band_name, lower, upper in rate_bands:

    mask = (
        (y_valid >= lower)
        & (y_valid < upper)
    )

    if mask.sum() == 0:
        continue

    actual = y_valid[mask]
    predicted = final_predictions[mask]

    print(f"\n{band_name}")

    print(
        f"  Count          : {mask.sum():,}"
    )

    print(
        f"  MAE            : "
        f"${mean_absolute_error(actual, predicted):,.2f}"
    )

    print(
        f"  RMSE           : "
        f"${rmse(actual, predicted):,.2f}"
    )

    print(
        f"  Actual mean    : "
        f"${actual.mean():,.2f}"
    )

    print(
        f"  Predicted mean : "
        f"${predicted.mean():,.2f}"
    )


# ============================================================
# DISTANCE BAND ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("DISTANCE-BAND ANALYSIS")
print("=" * 70)

distance_bins = [
    (0, 250, "0-250"),
    (250, 500, "251-500"),
    (500, 750, "501-750"),
    (750, 1000, "751-1000"),
    (1000, 1500, "1001-1500"),
    (1500, float("inf"), "1500+"),
]

for lower, upper, band_name in distance_bins:

    mask = (
        (valid_df["distance"] > lower)
        & (valid_df["distance"] <= upper)
    )

    if mask.sum() == 0:
        continue

    actual = y_valid[mask]
    predicted = final_predictions[mask]

    print(f"\n{band_name}")

    print(
        f"  Count          : {mask.sum():,}"
    )

    print(
        f"  MAE            : "
        f"${mean_absolute_error(actual, predicted):,.2f}"
    )

    print(
        f"  RMSE           : "
        f"${rmse(actual, predicted):,.2f}"
    )

    print(
        f"  Actual mean    : "
        f"${actual.mean():,.2f}"
    )

    print(
        f"  Predicted mean : "
        f"${predicted.mean():,.2f}"
    )


# ============================================================
# EQUIPMENT ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("EQUIPMENT ANALYSIS")
print("=" * 70)

for equipment in sorted(
    valid_df["equipment"].dropna().unique()
):

    mask = (
        valid_df["equipment"] == equipment
    )

    actual = y_valid[mask]
    predicted = final_predictions[mask]

    print(f"\n{equipment}")

    print(
        f"  Count          : {mask.sum():,}"
    )

    print(
        f"  MAE            : "
        f"${mean_absolute_error(actual, predicted):,.2f}"
    )

    print(
        f"  RMSE           : "
        f"${rmse(actual, predicted):,.2f}"
    )

    print(
        f"  Actual mean    : "
        f"${actual.mean():,.2f}"
    )

    print(
        f"  Predicted mean : "
        f"${predicted.mean():,.2f}"
    )


# ============================================================
# ERROR DATAFRAME
# ============================================================

error_df = valid_df[
    [
        "load_id",
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "date",
        "posted_rate",
    ]
].copy()

error_df = error_df.rename(
    columns={
        "posted_rate": "actual_rate"
    }
)

error_df["predicted_rate"] = final_predictions

error_df["error"] = (
    error_df["predicted_rate"]
    - error_df["actual_rate"]
)

error_df["absolute_error"] = (
    error_df["error"].abs()
)


# ============================================================
# TOP ERRORS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 30 ABSOLUTE ERRORS")
print("=" * 70)

top_errors = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(30)
)

print(
    top_errors.to_string(
        index=False
    )
)


# ============================================================
# EXTREME ERROR TABLE
# ============================================================

print("\n")
print("=" * 70)
print("EXTREME-RATE ERROR ANALYSIS")
print("=" * 70)

extreme_errors = (
    error_df[
        error_df["actual_rate"] >= EXTREME_THRESHOLD
    ]
    .sort_values(
        "absolute_error",
        ascending=False
    )
)

if len(extreme_errors) > 0:

    print(
        extreme_errors.to_string(
            index=False
        )
    )

else:

    print("No extreme validation rows found.")


# ============================================================
# SAVE ERROR ANALYSIS
# ============================================================

error_df.to_csv(
    ERROR_ANALYSIS_PATH,
    index=False
)

print(
    f"\nError analysis saved to:"
    f"\n{ERROR_ANALYSIS_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE - CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFIER FEATURE IMPORTANCE")
print("=" * 70)

classifier_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": classifier.get_feature_importance(),
})

classifier_importance = (
    classifier_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    classifier_importance.to_string(
        index=False
    )
)


# ============================================================
# FEATURE IMPORTANCE - THREE REGRESSORS
# ============================================================

print("\n")
print("=" * 70)
print("REGRESSOR FEATURE IMPORTANCE")
print("=" * 70)

for name, model in [
    ("Normal", normal_model),
    ("High-rate", high_model),
    ("Extreme", extreme_model),
]:

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance(),
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(f"\n{name} regressor:")
    print(
        importance.head(15).to_string(
            index=False
        )
    )


# ============================================================
# FINAL COMPARISON WITH CURRENT BEST MODEL
# ============================================================

CURRENT_BEST_MAE = 93.15

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("\nCurrent benchmark:")
print("  Random Log1p CatBoost")
print("  MAE : $93.15")

print("\nThree-stage model:")
print(f"  Selected mode : {selected_mode}")
print(f"  MAE           : ${selected_mae:,.2f}")
print(f"  RMSE          : ${selected_rmse:,.2f}")
print(f"  R²            : {selected_r2:.4f}")

difference = selected_mae - CURRENT_BEST_MAE

print(
    f"\nMAE difference vs current benchmark:"
    f" ${difference:+,.2f}"
)

if selected_mae < CURRENT_BEST_MAE:

    print(
        "\nThree-stage model produced a lower MAE "
        "than the current benchmark."
    )

else:

    print(
        "\nThree-stage model did NOT beat the "
        "current benchmark."
    )

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nRandom split : 80% train / 20% validation")
print("Split stratified by: Extreme rate >= $7,500")

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

print("\nThree-stage model:")
print(
    f"  Overall MAE  : ${selected_mae:,.2f}"
)

print(
    f"  Overall RMSE : ${selected_rmse:,.2f}"
)

print(
    f"  Overall R²   : {selected_r2:.4f}"
)

print(
    f"  Prediction mode: {selected_mode}"
)

print("\nModels created:")

print(
    f"  - {MODEL_CLASSIFIER_PATH}"
)

print(
    f"  - {MODEL_NORMAL_PATH}"
)

print(
    f"  - {MODEL_HIGH_PATH}"
)

print(
    f"  - {MODEL_EXTREME_PATH}"
)

print(
    f"  - {ERROR_ANALYSIS_PATH}"
)

print("\nDone.")
print("=" * 70)
