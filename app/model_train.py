from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import joblib
import os

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== Model Performance ==========")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R² Score : {r2:.4f}")
# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/linear_regression.pkl")

print("\n✅ Model saved successfully!")

# Plot Actual vs Predicted

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label="Actual Price")
plt.plot(predictions[:100], label="Predicted Price")

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Samples")
plt.ylabel("Price")

plt.legend()

plt.show()

# -----------------------------
# Random Forest Model
# -----------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n========== Random Forest ==========")
print(f"MAE : {rf_mae:.4f}")
print(f"MSE : {rf_mse:.4f}")
print(f"R² Score : {rf_r2:.4f}")