import pandas as pd
from pathlib import Path

# File paths
raw_file = Path("data/raw/stock_data.csv")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(raw_file)

print("Original Shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Convert Date column to datetime (if present)
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

print("Cleaned Shape:", df.shape)
print("\nMissing Values:\n")
print(df.isnull().sum())

# Save cleaned data
output_file = processed_dir / "clean_stock_data.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Cleaned data saved to: {output_file}")