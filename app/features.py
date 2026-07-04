import pandas as pd
from pathlib import Path
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

# Load cleaned data
input_file = Path("data/processed/clean_stock_data.csv")
df = pd.read_csv(input_file)

# -------------------------------
# Basic technical indicators
# -------------------------------
df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
df["EMA_10"] = EMAIndicator(close=df["Close"], window=10).ema_indicator()
df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

macd = MACD(close=df["Close"])
df["MACD"] = macd.macd()

# -------------------------------
# Lag features (important for forecasting)
# -------------------------------
df["Close_Lag_1"] = df["Close"].shift(1)
df["Close_Lag_2"] = df["Close"].shift(2)
df["Close_Lag_3"] = df["Close"].shift(3)

df["Volume_Lag_1"] = df["Volume"].shift(1)

# -------------------------------
# Optional return feature
# -------------------------------
df["Daily_Return"] = df["Close"].pct_change()

# -------------------------------
# Drop NaNs created by indicators + lagging
# -------------------------------
df = df.dropna().reset_index(drop=True)

# Save feature engineered data
output_file = Path("data/processed/featured_stock_data.csv")
df.to_csv(output_file, index=False)

print("✅ Feature Engineering Completed")
print(df.head())
print("\nColumns:")
print(df.columns)
