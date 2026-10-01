import pandas as pd

train = pd.read_csv("train-test.csv")
validation = pd.read_csv("validation.csv")

print("TRAIN COLUMNS:")
print(train.columns.tolist())

print("\nVALIDATION COLUMNS:")
print(validation.columns.tolist())

print("\nCOLUMNS ONLY IN TRAIN:")
print(set(train.columns) - set(validation.columns))

print("\nCOLUMNS ONLY IN VALIDATION:")
print(set(validation.columns) - set(train.columns))