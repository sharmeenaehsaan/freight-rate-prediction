import pandas as pd
validation = pd.read_csv("data/validation.csv")

print("Validation shape:", validation.shape)

print("\nValidation missing values:")
print(validation.isnull().sum())

print("\nValidation negative weights:")
print((validation["weight"] < 0).sum())

print("\nValidation zero weights:")
print((validation["weight"] == 0).sum())

print("\nValidation negative distance:")
print((validation["distance"] < 0).sum())
