from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. FILES
# =========================================================

DATA_DIR = Path("data")

TRAIN_FILE = DATA_DIR / "train-test.csv"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(TRAIN_FILE)

print("Original shape:", df.shape)


# =========================================================
# 3. BASIC CLEANING
# =========================================================

# Weight should never be negative.
# Negative values appear to be sign errors, so convert them
# to their absolute values.
df["weight"] = df["weight"].abs()


# =========================================================
# 4. DATE CONVERSION
# =========================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isna().any():
    raise ValueError("Invalid dates found in the dataset.")


# =========================================================
# 5. TRAINING MEDIANS
# =========================================================

# IMPORTANT:
# Calculate imputation values from the full development data
# for this first preparation step.
#
# When we build the final modeling pipeline, these values
# should be learned only from the training split.

weight_median = df["weight"].median()
market_index_median = df["market_index"].median()

print("\nWeight median:", weight_median)
print("Market index median:", market_index_median)


# =========================================================
# 6. FILL MISSING VALUES
# =========================================================

df["weight"] = df["weight"].fillna(weight_median)

df["market_index"] = df["market_index"].fillna(
    market_index_median
)


# =========================================================
# 7. DATE FEATURES
# =========================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear


# =========================================================
# 8. ROUTE FEATURE
# =========================================================

df["route"] = (
    df["pickup"].astype(str)
    + " -> "
    + df["delivery"].astype(str)
)


# =========================================================
# 9. GEOGRAPHICAL DISTANCE
# =========================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate straight-line distance between
    two latitude/longitude points.

    Result is approximately in miles.
    """

    earth_radius_miles = 3958.7613

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_miles * c


df["geo_distance"] = haversine_distance(
    df["pickup_lat"],
    df["pickup_lon"],
    df["delivery_lat"],
    df["delivery_lon"],
)


# =========================================================
# 10. CHECK THE RESULT
# =========================================================

print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\nNegative weights after cleaning:")
print((df["weight"] < 0).sum())


print("\nNew columns:")
print(df.columns.tolist())


# =========================================================
# 11. TIME-BASED SPLIT
# =========================================================

# January through September = training
# October = validation

train_df = df[df["date"] < "2025-10-01"].copy()

valid_df = df[
    (df["date"] >= "2025-10-01")
    & (df["date"] < "2025-11-01")
].copy()


# =========================================================
# 12. DISPLAY SPLIT INFORMATION
# =========================================================

print("\n==============================")
print("TIME-BASED SPLIT")
print("==============================")

print("Training rows:", len(train_df))
print("Validation rows:", len(valid_df))

print(
    "Training dates:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)

print(
    "Validation dates:",
    valid_df["date"].min(),
    "to",
    valid_df["date"].max()
)


# =========================================================
# 13. TARGET INFORMATION
# =========================================================

print("\nTraining target rows:")
print(train_df["posted_rate"].describe())

print("\nValidation target rows:")
print(valid_df["posted_rate"].describe())


# =========================================================
# 14. SAVE PREPARED DATA
# =========================================================

train_df.to_csv(
    DATA_DIR / "prepared_train.csv",
    index=False
)

valid_df.to_csv(
    DATA_DIR / "prepared_validation.csv",
    index=False
)

print("\nSaved:")
print(DATA_DIR / "prepared_train.csv")
print(DATA_DIR / "prepared_validation.csv")