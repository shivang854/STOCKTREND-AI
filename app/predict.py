import joblib
import pandas as pd

# Load featured dataset
df = pd.read_csv("data/processed/featured_stock_data.csv")

# Same feature columns used during training
feature_cols = [
    "Close",
    "Volume",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "RSI",
    "MACD",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Volume_Lag_1",
    "Daily_Return"

    
]

# Latest row for prediction
latest_data = df[feature_cols].iloc[[-1]]

# Load best model
model = joblib.load("models/random_forest.pkl")

# Predict next value
predicted_price = model.predict(latest_data)[0]

# Current close price
current_close = df["Close"].iloc[-1]

# Percentage change
change_pct = ((predicted_price - current_close) / current_close) * 100

# Signal logic
if change_pct > 1:
    signal = "BUY"
elif change_pct < -1:
    signal = "SELL"
else:
    signal = "HOLD"

print("\n========== NEXT DAY STOCK PREDICTION ==========")
print(f"Current Close Price   : {current_close:.2f}")
print(f"Predicted Next Close  : {predicted_price:.2f}")
print(f"Expected Change       : {change_pct:.2f}%")
print(f"Signal                : {signal}")
