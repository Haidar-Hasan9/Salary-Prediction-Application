import kagglehub
import pandas as pd
from pathlib import Path

# Create data/raw directory if it doesn't exist
raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Download the dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("ruchi798/data-science-job-salaries")
print(f"Dataset downloaded to: {path}")

# Find the CSV file in the downloaded folder
csv_files = list(Path(path).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV file found in downloaded dataset")

csv_path = csv_files[0]
print(f"Loading CSV: {csv_path}")

# Read into pandas DataFrame
df = pd.read_csv(csv_path)

# Save a copy to data/raw for easy access
output_path = raw_dir / "ds_salaries.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved to: {output_path}")

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Show basic info
print("\nDataset info:")
print(df.info())