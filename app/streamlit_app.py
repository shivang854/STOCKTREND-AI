import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="StockTrend AI", page_icon="📈", layout="wide")

st.title("📈 StockTrend AI")
st.subheader("AI Powered Stock Prediction Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
DATA_PATH = "data/processed/featured_stock_data.csv"
MODEL_PATH = "models/random_forest.pkl"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    for col in ["Date", "date", "Datetime", "datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break

    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at `{DATA_PATH}`.")
    st.stop()

df = load_data(DATA_PATH)

feature_cols = [
    "Close", "Volume", "SMA_10", "SMA_20", "EMA_10", "RSI", "MACD",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Volume_Lag_1", "Daily_Return",
]

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing expected columns in data: {missing_cols}")
    st.stop()

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# -------------------------
# BUILD TARGET (next day's Close) FOR COMPARISON/BACKTEST
# -------------------------
data = df.copy()
data["Target"] = data["Close"].shift(-1)
data = data.dropna(subset=["Target"])

X = data[feature_cols]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # keep time order for time series
)

# -------------------------
# TRAIN / COMPARE MODELS (cached so this doesn't retrain every rerun)
# -------------------------
@st.cache_resource
def train_and_compare(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)

    results = []
    trained_models = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({"Model": name, "MAE": mae, "MSE": mse, "R2 Score": r2})
        trained_models[name] = m

    results_df = pd.DataFrame(results)
    return results_df, trained_models


results_df, trained_models = train_and_compare(X_train, y_train, X_test, y_test)

# Pick best model by lowest MAE (change to r2 if you prefer)
best_model_name = results_df.loc[results_df["MAE"].idxmin(), "Model"]
best_model = trained_models[best_model_name]

# Use the best trained model for predictions if no external model.pkl was loaded
if model is None:
    model = best_model

# -------------------------
# PREDICTION SECTION
# -------------------------
st.subheader("🔮 Make Prediction")

option = st.radio("Choose input mode", ["Latest Data", "Manual Row Selection"])

if option == "Manual Row Selection":
    row_index = st.slider("Select Data Row", 0, len(df) - 1, len(df) - 1)
    input_data = df[feature_cols].iloc[[row_index]]
else:
    input_data = df[feature_cols].iloc[[-1]]

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if st.button("Predict Next Price 🚀"):
    prediction = model.predict(input_data)[0]
    current_price = df["Close"].iloc[-1]
    change = ((prediction - current_price) / current_price) * 100

    if change > 1:
        signal = "🟢 BUY"
    elif change < -1:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    st.session_state.prediction_result = {
        "current_price": current_price,
        "prediction": prediction,
        "change": change,
        "signal": signal,
        "time": datetime.datetime.now(),
    }

result = st.session_state.prediction_result

if result:
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{result['current_price']:.2f}")
    col2.metric("Predicted Price", f"{result['prediction']:.2f}")
    col3.metric("Change %", f"{result['change']:.2f}%")

    st.subheader("Trading Signal")
    if result["signal"] == "🟢 BUY":
        st.success(result["signal"])
    elif result["signal"] == "🔴 SELL":
        st.error(result["signal"])
    else:
        st.warning(result["signal"])

    report = pd.DataFrame([{
        "Current Price": result["current_price"],
        "Predicted Price": result["prediction"],
        "Change %": result["change"],
        "Signal": result["signal"],
        "Time": result["time"],
    }])
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Report", data=csv,
        file_name="prediction_report.csv", mime="text/csv",
        key="download_report_btn",
    )
else:
    st.info("Click **Predict Next Price 🚀** to generate a prediction.")

# -------------------------
# 🏆 MODEL COMPARISON
# -------------------------
st.subheader("🏆 Model Comparison")

st.dataframe(
    results_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "R2 Score": "{:.4f}"}),
    use_container_width=True,
)

st.success(f"Best Model : {best_model_name}")

# -------------------------
# 📊 MODEL BACKTESTING
# -------------------------
st.subheader("📊 Model Backtesting")

if st.button("Run Backtest 🔥"):
    test_index = X_test.index
    test_preds = best_model.predict(X_test)

    bt = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": test_preds,
    }, index=test_index)

    # Simple strategy: go long if predicted next close > current close, else flat
    bt["PrevClose"] = data.loc[test_index, "Close"].values
    bt["Signal"] = np.where(bt["Predicted"] > bt["PrevClose"], 1, 0)
    bt["MarketReturn"] = bt["Actual"].pct_change().fillna(0)
    bt["StrategyReturn"] = bt["Signal"].shift(1).fillna(0) * bt["MarketReturn"]

    bt["BuyHoldEquity"] = (1 + bt["MarketReturn"]).cumprod()
    bt["StrategyEquity"] = (1 + bt["StrategyReturn"]).cumprod()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["BuyHoldEquity"], name="Buy & Hold"))
    fig_bt.add_trace(go.Scatter(x=bt.index, y=bt["StrategyEquity"], name="Strategy"))
    fig_bt.update_layout(title="Backtest: Strategy vs Buy & Hold", yaxis_title="Equity (normalized)")

    st.plotly_chart(fig_bt, use_container_width=True)

    final_return = (bt["StrategyEquity"].iloc[-1] - 1) * 100
    bh_return = (bt["BuyHoldEquity"].iloc[-1] - 1) * 100
    c1, c2 = st.columns(2)
    c1.metric("Strategy Return", f"{final_return:.2f}%")
    c2.metric("Buy & Hold Return", f"{bh_return:.2f}%")

# -------------------------
# 📈 ACTUAL vs PREDICTED (full history)
# -------------------------
st.subheader("📈 Actual vs Predicted")

full_preds = model.predict(df[feature_cols])

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual Price"))
fig1.add_trace(go.Scatter(x=df.index, y=full_preds, mode="lines", name="Predicted Price"))
st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# 🕯 CANDLESTICK CHART
# -------------------------
st.subheader("🕯 Candlestick Chart")

ohlc_cols = ["Open", "High", "Low", "Close"]
if all(c in df.columns for c in ohlc_cols):
    fig2 = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )])
    fig2.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Candlestick chart needs Open/High/Low/Close columns.")

# -------------------------
# INDICATORS
# -------------------------
st.subheader("📊 Latest Technical Indicators")
indicator_cols = [c for c in ["Close", "SMA_10", "SMA_20", "EMA_10", "RSI", "MACD"] if c in df.columns]
st.dataframe(df[indicator_cols].tail(1))

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.success("Dashboard Running Successfully 🚀")
