import yfinance as yf
import os

# Stock ticker
ticker = "AAPL"

# Download historical data
stock_data = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)

# ⭐ MultiIndex fix
if hasattr(stock_data.columns, "nlevels") and stock_data.columns.nlevels > 1:
    stock_data.columns = stock_data.columns.get_level_values(0)

# Date ko normal column banao
stock_data.reset_index(inplace=True)

# Folder create karo
os.makedirs("data/raw", exist_ok=True)

# CSV save karo
stock_data.to_csv("data/raw/stock_data.csv", index=False)

print("✅ Stock data downloaded successfully!")
print(stock_data.head())
