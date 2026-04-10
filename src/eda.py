import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv("data/raw/ds_salaries.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("UNIQUE VALUES IN CATEGORICAL COLUMNS")
print("=" * 50)
categorical_cols = ['experience_level', 'employment_type', 'job_title', 'company_size', 'remote_ratio']
for col in categorical_cols:
    print(f"\n{col}: {df[col].unique()[:10]}")

print("\n" + "=" * 50)
print("TARGET VARIABLE: salary_in_usd")
print("=" * 50)
print(f"Min: {df['salary_in_usd'].min()}")
print(f"Max: {df['salary_in_usd'].max()}")
print(f"Mean: {df['salary_in_usd'].mean():.2f}")
print(f"Median: {df['salary_in_usd'].median():.2f}")

print("\n" + "=" * 50)
print("SANITY CHECKS")
print("=" * 50)
# remote_ratio should be 0, 50, or 100
invalid_remote = df[~df['remote_ratio'].isin([0, 50, 100])]
print(f"Invalid remote_ratio values: {invalid_remote.shape[0]} rows")
if invalid_remote.shape[0] > 0:
    print(invalid_remote['remote_ratio'].unique())