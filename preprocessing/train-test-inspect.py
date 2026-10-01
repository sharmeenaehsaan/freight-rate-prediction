import pandas as pd

train = pd.read_csv("data/train-test.csv")

print("Shape:", train.shape)

print("\nMissing values:")
print(train.isnull().sum())

print("\nDuplicate rows:")
print(train.duplicated().sum())

print("\nDuplicate load IDs:")
print(train["load_id"].duplicated().sum())

print("\nData types:")
print(train.dtypes)

print("\nTarget (`posted_rate`) statistics:")
print(train["posted_rate"].describe())
print("\nUnique pickup locations:")
print(train["pickup"].nunique())
print(train["pickup"].value_counts().head(20))

print("\nUnique delivery locations:")
print(train["delivery"].nunique())
print(train["delivery"].value_counts().head(20))

print("\nEquipment types:")
print(train["equipment"].value_counts())

print("\nDate range:")
print(train["date"].min(), "to", train["date"].max())
print("Negative values:")

print("\nDistance < 0:")
print((train["distance"] < 0).sum())

print("\nWeight < 0:")
print((train["weight"] < 0).sum())

print("\nMarket index < 0:")
print((train["market_index"] < 0).sum())

print("\nQuote signal < 0:")
print((train["quote_signal"] < 0).sum())
print("\nZero values:")

print("Distance == 0:", (train["distance"] == 0).sum())
print("Weight == 0:", (train["weight"] == 0).sum())
print("Market index == 0:", (train["market_index"] == 0).sum())
print("Quote signal == 0:", (train["quote_signal"] == 0).sum())
print("\nMissing weight examples:")
print(train[train["weight"].isna()].head())

print("\nMissing market_index examples:")
print(train[train["market_index"].isna()].head())
negative_weight = train[train["weight"] < 0]

print("Number of negative weights:", len(negative_weight))

print("\nNegative weight statistics:")
print(negative_weight["weight"].describe())

print("\nNegative weight examples:")
print(
    negative_weight[
        ["load_id", "pickup", "delivery", "distance",
         "equipment", "weight", "date", "posted_rate"]
    ].head(20).to_string(index=False)
)
print("\nAbsolute values of negative weights:")
print(negative_weight["weight"].abs().describe())
print(
    train[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate"
        ]
    ].corr()
)
print("\nAverage posted rate by equipment:")
print(
    train.groupby("equipment")["posted_rate"]
    .agg(["count", "mean", "median"])
)
print("\nRate per mile:")
train["rate_per_distance"] = train["posted_rate"] / train["distance"]

print(train["rate_per_distance"].describe())
print("\nRate per mile by equipment:")
print(
    train.groupby("equipment")["rate_per_distance"]
    .agg(["count", "mean", "median"])
)