import pandas as pd
df = pd.read_csv("vr_training_data.csv")

print(df.isna().sum())  # Check for missing values
print(df.describe())  # See numerical ranges
print(df.head())  # Display the first few rows
