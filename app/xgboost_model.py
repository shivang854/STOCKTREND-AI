import joblib
import os
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/processed/featured_stock_data.csv")

# Features
X = df[[
    "Open",
    "High",
    "Low",
    "Volume",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "RSI",
    "MACD"
]]

# Target
y = df["Close"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
print("\n========== XGBoost ==========")
print(f"MAE : {mean_absolute_error(y_test,pred):.4f}")
print(f"MSE : {mean_squared_error(y_test,pred):.4f}")
print(f"R² Score : {r2_score(y_test,pred):.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost.pkl")

print("\n✅ XGBoost model saved successfully!")