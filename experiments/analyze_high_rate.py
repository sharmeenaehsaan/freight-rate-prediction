from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("data/prepared_train.csv")
HIGH_RATE_THRESHOLD = 5000


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("HIGH-RATE LOAD ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# Make sure date is datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Rate per mile
df["rate_per_distance"] = df["posted_rate"] / df["distance"]

# High-rate flag
df["high_rate"] = df["posted_rate"] >= HIGH_RATE_THRESHOLD


# ============================================================
# 1. BASIC HIGH-RATE STATISTICS
# ============================================================

high = df[df["high_rate"]].copy()
normal = df[~df["high_rate"]].copy()

print("\n" + "=" * 70)
print("1. BASIC STATISTICS")
print("=" * 70)

print(f"\nTotal loads: {len(df):,}")
print(f"High-rate loads (>= ${HIGH_RATE_THRESHOLD:,}): {len(high):,}")
print(f"Normal loads: {len(normal):,}")
print(f"High-rate percentage: {len(high) / len(df) * 100:.2f}%")

print("\nHigh-rate target statistics:")
print(high["posted_rate"].describe())

print("\nNormal-rate target statistics:")
print(normal["posted_rate"].describe())


# ============================================================
# 2. HIGH-RATE BY MONTH
# ============================================================

print("\n" + "=" * 70)
print("2. HIGH-RATE LOADS BY MONTH")
print("=" * 70)

monthly = (
    df.groupby(df["date"].dt.month)
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
    )
)

monthly["high_rate_percentage"] = (
    monthly["high_rate_loads"] / monthly["total_loads"] * 100
)

print(monthly.round(2))


# ============================================================
# 3. HIGH-RATE BY EQUIPMENT
# ============================================================

print("\n" + "=" * 70)
print("3. HIGH-RATE LOADS BY EQUIPMENT")
print("=" * 70)

equipment = (
    df.groupby("equipment")
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        average_rate_per_mile=("rate_per_distance", "mean"),
    )
)

equipment["high_rate_percentage"] = (
    equipment["high_rate_loads"] / equipment["total_loads"] * 100
)

print(equipment.round(2))


# ============================================================
# 4. HIGH-RATE BY DISTANCE RANGE
# ============================================================

print("\n" + "=" * 70)
print("4. HIGH-RATE LOADS BY DISTANCE")
print("=" * 70)

bins = [0, 250, 500, 750, 1000, 1500, np.inf]
labels = [
    "0-250",
    "251-500",
    "501-750",
    "751-1000",
    "1001-1500",
    "1500+",
]

df["distance_group"] = pd.cut(
    df["distance"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

distance = (
    df.groupby("distance_group", observed=False)
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"),
        average_rate_per_mile=("rate_per_distance", "mean"),
    )
)

distance["high_rate_percentage"] = (
    distance["high_rate_loads"] / distance["total_loads"] * 100
)

print(distance.round(2))


# ============================================================
# 5. HIGH-RATE BY MARKET INDEX
# ============================================================

print("\n" + "=" * 70)
print("5. MARKET INDEX COMPARISON")
print("=" * 70)

print("\nHigh-rate market index:")
print(high["market_index"].describe())

print("\nNormal-rate market index:")
print(normal["market_index"].describe())

print("\nAverage market index:")
print(f"High-rate: ${high['market_index'].mean():.4f}")
print(f"Normal-rate: ${normal['market_index'].mean():.4f}")


# ============================================================
# 6. HIGH-RATE BY QUOTE SIGNAL
# ============================================================

print("\n" + "=" * 70)
print("6. QUOTE SIGNAL COMPARISON")
print("=" * 70)

print("\nHigh-rate quote signal:")
print(high["quote_signal"].describe())

print("\nNormal-rate quote signal:")
print(normal["quote_signal"].describe())

print("\nAverage quote signal:")
print(f"High-rate: {high['quote_signal'].mean():.4f}")
print(f"Normal-rate: {normal['quote_signal'].mean():.4f}")


# ============================================================
# 7. HIGH-RATE BY ROUTE
# ============================================================

print("\n" + "=" * 70)
print("7. ROUTES WITH MOST HIGH-RATE LOADS")
print("=" * 70)

route_stats = (
    df.groupby("route")
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        max_rate=("posted_rate", "max"),
    )
)

route_stats["high_rate_percentage"] = (
    route_stats["high_rate_loads"] /
    route_stats["total_loads"] * 100
)

route_stats = route_stats.sort_values(
    ["high_rate_loads", "max_rate"],
    ascending=False,
)

print(route_stats.head(20).round(2))


# ============================================================
# 8. LOCATIONS WITH MOST HIGH-RATE LOADS
# ============================================================

print("\n" + "=" * 70)
print("8. PICKUP LOCATIONS")
print("=" * 70)

pickup_stats = (
    df.groupby("pickup")
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        max_rate=("posted_rate", "max"),
    )
)

pickup_stats["high_rate_percentage"] = (
    pickup_stats["high_rate_loads"] /
    pickup_stats["total_loads"] * 100
)

pickup_stats = pickup_stats.sort_values(
    ["high_rate_loads", "max_rate"],
    ascending=False,
)

print(pickup_stats.head(20).round(2))


print("\n" + "=" * 70)
print("9. DELIVERY LOCATIONS")
print("=" * 70)

delivery_stats = (
    df.groupby("delivery")
    .agg(
        total_loads=("posted_rate", "size"),
        high_rate_loads=("high_rate", "sum"),
        average_rate=("posted_rate", "mean"),
        max_rate=("posted_rate", "max"),
    )
)

delivery_stats["high_rate_percentage"] = (
    delivery_stats["high_rate_loads"] /
    delivery_stats["total_loads"] * 100
)

delivery_stats = delivery_stats.sort_values(
    ["high_rate_loads", "max_rate"],
    ascending=False,
)

print(delivery_stats.head(20).round(2))


# ============================================================
# 10. RATE PER MILE
# ============================================================

print("\n" + "=" * 70)
print("10. RATE PER MILE")
print("=" * 70)

print("\nHigh-rate loads:")
print(high["rate_per_distance"].describe())

print("\nNormal-rate loads:")
print(normal["rate_per_distance"].describe())

print("\nHighest rate-per-mile loads:")

columns = [
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
    "rate_per_distance",
]

print(
    high.sort_values(
        "rate_per_distance",
        ascending=False
    )[columns].head(20).to_string(index=False)
)


# ============================================================
# 11. TOP 30 MOST EXPENSIVE LOADS
# ============================================================

print("\n" + "=" * 70)
print("11. TOP 30 HIGHEST POSTED RATES")
print("=" * 70)

top_rates = df.sort_values(
    "posted_rate",
    ascending=False
)[columns]

print(top_rates.head(30).to_string(index=False))


# ============================================================
# 12. CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("12. NUMERIC CORRELATIONS WITH POSTED RATE")
print("=" * 70)

numeric_columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "geo_distance",
    "rate_per_distance",
    "posted_rate",
]

correlations = (
    df[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(ascending=False)
)

print(correlations.round(4))


# ============================================================
# 13. SAVE HIGH-RATE DATA
# ============================================================

output_columns = [
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
    "rate_per_distance",
    "distance_group",
]

high_sorted = high.sort_values(
    "posted_rate",
    ascending=False
)

output_path = Path("high_rate_analysis.csv")
# Make sure distance_group exists in the high-rate output
if "distance_group" not in high_sorted.columns:
    high_sorted["distance_group"] = pd.cut(
        high_sorted["distance"],
        bins=[0, 250, 500, 750, 1000, 1500, float("inf")],
        labels=[
            "0-250",
            "251-500",
            "501-750",
            "751-1000",
            "1001-1500",
            "1500+"
        ],
        include_lowest=True
    )
high_sorted[output_columns].to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nSaved high-rate loads to:")
print(output_path)

print("\nNext step:")
print("Send me the terminal output from this script.")