import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from textblob import TextBlob
import datetime
import os
import requests
import xml.etree.ElementTree as ET


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# NEW: yfinance import for the Live Stock Search feature.
# If it's not installed, the app still runs — the search box just shows a friendly message.
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="StockTrend AI", page_icon="📈", layout="wide")

st.title("📈 StockTrend AI")
st.subheader("AI Powered Stock Prediction Dashboard")

# =========================================================
# 📂 PORTFOLIO TRACKER / WATCHLIST — SIDEBAR INPUT  (NEW FEATURE)
# ---------------------------------------------------------
# Lets the user type multiple tickers (comma-separated) in the sidebar.
# The list is parsed here and used later by the "Portfolio Tracker /
# Watchlist" section further down the page, which builds a compact
# summary table (price, RSI, MACD, sentiment, AI signal) for each ticker.
# =========================================================
st.sidebar.subheader("📂 Portfolio Tracker")
watchlist_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)",
    value="AAPL, TSLA, MSFT",
    key="watchlist_input",
    help="Example: AAPL, TSLA, MSFT, NVDA, RELIANCE.NS, TCS.NS",
)
# Clean + de-duplicate the ticker list while keeping order
watchlist_tickers = []
for _t in watchlist_input.split(","):
    _t = _t.strip().upper()
    if _t and _t not in watchlist_tickers:
        watchlist_tickers.append(_t)


# =========================================================
# 🧠 NEWS SENTIMENT HELPER  (FIX: moved to module/top level)
# ---------------------------------------------------------
# This used to be defined *inside* fetch_stock_news(), which meant it
# went out of scope the moment that function returned — calling it later
# in the display loop raised a `NameError: name 'get_news_sentiment' is
# not defined`. It now lives at the top level so both fetch_stock_news()
# and the display code below can use it safely.
# =========================================================
def get_news_sentiment(text: str) -> str:
    """Returns a sentiment label ('🟢 Positive' / '🔴 Negative' / '🟡 Neutral')
    for a piece of text (usually a news headline) using TextBlob polarity."""
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "🟢 Positive"
        elif polarity < -0.1:
            return "🔴 Negative"
        else:
            return "🟡 Neutral"
    except Exception:
        return "🟡 Neutral"


def get_average_sentiment_score(news_items: list) -> float:
    """
    Averages the TextBlob polarity score (-1 to +1) across a list of news
    items (each a dict with a 'title' key). Used to feed a single numeric
    sentiment_score into generate_ai_recommendation() below.
    Returns 0.0 (neutral) if there are no news items to score.
    """
    if not news_items:
        return 0.0
    try:
        scores = [TextBlob(n["title"]).sentiment.polarity for n in news_items]
        return sum(scores) / len(scores)
    except Exception:
        return 0.0


# =========================================================
# 🤖 AI RECOMMENDATION HELPER  (NEW FEATURE)
# ---------------------------------------------------------
# Combines four existing signals you already compute elsewhere in the app:
#   1. predicted price vs current price   (from the Prediction section)
#   2. RSI                                (from Technical Indicators)
#   3. MACD                                (from Technical Indicators)
#   4. news sentiment score               (from Live Stock Search news)
# into one simple BUY / HOLD / SELL recommendation, plus a short
# human-readable explanation of why.
#
# This is a lightweight, rule-based scoring system (not a trained model) —
# each signal casts a "vote" of +1 (bullish), -1 (bearish), or 0 (neutral),
# and the total vote decides the final call.
# =========================================================
def generate_ai_recommendation(prediction, current_price, rsi, macd, sentiment_score):
    """
    Combines prediction, RSI, MACD, and news sentiment into a final
    recommendation.

    Parameters
    ----------
    prediction : float or None      -> model's predicted next price
    current_price : float or None   -> latest known price
    rsi : float or None              -> latest RSI value
    macd : float or None             -> latest MACD value
    sentiment_score : float          -> average news sentiment polarity (-1 to +1)

    Returns
    -------
    (signal, reason) : tuple(str, str)
        signal -> "🟢 BUY", "🟡 HOLD", or "🔴 SELL"
        reason -> short explanation combining all four signals
    """
    score = 0
    reasons = []

    # ---- 1) Predicted price vs current price ----
    if prediction is not None and current_price:
        price_change_pct = ((prediction - current_price) / current_price) * 100
        if price_change_pct > 1:
            score += 1
            reasons.append(f"predicted price is {price_change_pct:.2f}% above the current price")
        elif price_change_pct < -1:
            score -= 1
            reasons.append(f"predicted price is {abs(price_change_pct):.2f}% below the current price")
        else:
            reasons.append("predicted price is close to the current price")
    else:
        reasons.append("no prediction available yet")

    # ---- 2) RSI (Relative Strength Index) ----
    if rsi is not None:
        if rsi > 70:
            score -= 1
            reasons.append(f"RSI ({rsi:.1f}) shows overbought conditions")
        elif rsi < 30:
            score += 1
            reasons.append(f"RSI ({rsi:.1f}) shows oversold conditions")
        else:
            reasons.append(f"RSI ({rsi:.1f}) is in a neutral range")
    else:
        reasons.append("RSI unavailable")

    # ---- 3) MACD (Moving Average Convergence Divergence) ----
    if macd is not None:
        if macd > 0:
            score += 1
            reasons.append("MACD is positive, suggesting upward momentum")
        elif macd < 0:
            score -= 1
            reasons.append("MACD is negative, suggesting downward momentum")
        else:
            reasons.append("MACD is flat")
    else:
        reasons.append("MACD unavailable")

    # ---- 4) News sentiment score ----
    if sentiment_score > 0.1:
        score += 1
        reasons.append("recent news sentiment is positive")
    elif sentiment_score < -0.1:
        score -= 1
        reasons.append("recent news sentiment is negative")
    else:
        reasons.append("recent news sentiment is neutral")

    # ---- Final decision based on total score ----
    if score >= 2:
        signal = "🟢 BUY"
    elif score <= -2:
        signal = "🔴 SELL"
    else:
        signal = "🟡 HOLD"

    reason = "Because " + "; ".join(reasons) + "."
    return signal, reason


# =========================================================
# 📂 PORTFOLIO TRACKER HELPERS  (NEW FEATURE)
# ---------------------------------------------------------
# The rest of the dashboard gets RSI/MACD from your local
# featured_stock_data.csv. The Portfolio Tracker works on ARBITRARY
# tickers typed in by the user (not just the one stock in your CSV), so
# it needs to compute RSI/MACD itself from live yfinance price history.
# These are standard, lightweight implementations using pandas only.
# =========================================================
def compute_rsi(close_series: pd.Series, period: int = 14):
    """Computes the latest RSI value from a series of closing prices."""
    try:
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        latest = rsi.iloc[-1]
        return float(latest) if pd.notna(latest) else None
    except Exception:
        return None


def compute_macd(close_series: pd.Series, fast: int = 12, slow: int = 26):
    """Computes the latest MACD line value (fast EMA - slow EMA) from closing prices."""
    try:
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        latest = macd_line.iloc[-1]
        return float(latest) if pd.notna(latest) else None
    except Exception:
        return None


# =========================================================
# 🔍 LIVE STOCK SEARCH  (NEW FEATURE)
# ---------------------------------------------------------
# This section lets the user type ANY stock ticker (e.g. AAPL, MSFT, TSLA,
# RELIANCE.NS for NSE-listed stocks, etc.) and pull LIVE data straight from
# Yahoo Finance via the `yfinance` library.
#
# It is completely independent from the rest of the dashboard below
# (which uses your local CSV + trained model), so it cannot break your
# existing prediction / chart / backtest / report features.
# =========================================================
st.markdown("## 🔍 Live Stock Search")

if not YF_AVAILABLE:
    # If yfinance isn't installed, tell the user how to fix it instead of crashing.
    st.warning(
        "`yfinance` is not installed. Run `pip install yfinance` and restart the app "
        "to use Live Stock Search."
    )
else:
    # Small helper functions kept local to this section so they don't
    # interfere with anything else in the file.

    @st.cache_data(ttl=60)  # cache for 60 seconds so we don't hammer Yahoo Finance
    def fetch_live_stock(ticker_symbol: str, period: str = "6mo"):
        """
        Fetches live price history + basic company info for a given ticker.
        Returns (history_dataframe, info_dict). Raises on invalid ticker.
        """
        ticker_obj = yf.Ticker(ticker_symbol)
        hist = ticker_obj.history(period=period)
        info = ticker_obj.info  # dict with company name, market cap, PE ratio, etc.
        return hist, info

    # ---------------------------------------------------------
    # 📰 FIXED: fetch_stock_news()
    # ---------------------------------------------------------
    # Fixes applied:
    #   1. Sentiment is now computed HERE (using the top-level
    #      get_news_sentiment helper) and stored on each news item,
    #      instead of being computed by an out-of-scope nested function
    #      later in the display loop.
    #   2. Title / link / published-time extraction is made more robust
    #      to yfinance's different news payload shapes (older vs newer
    #      `ticker.news` formats), with safe fallbacks at every step.
    #   3. Any single malformed news item is skipped instead of crashing
    #      the whole news list (try/except per item).
    # ---------------------------------------------------------
    @st.cache_data(ttl=300)
    def fetch_stock_news(ticker_symbol: str):
        """
        Fetches recent news for a given ticker using yfinance.
        Returns a list of dicts: {title, link, published, sentiment}
        """
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            raw_news = ticker_obj.news
        except Exception:
            return []

        if not raw_news:
            return []

        news_items = []

        for item in raw_news[:8]:
            try:
                content = item.get("content", {}) or {}

                # ---------- TITLE ----------
                title = (
                    item.get("title")
                    or item.get("headline")
                    or content.get("title")
                    or "No title"
                )

                # ---------- LINK ----------
                canonical = item.get("canonicalUrl") or {}
                content_canonical = content.get("canonicalUrl") or {}
                link = (
                    item.get("link")
                    or canonical.get("url")
                    or content_canonical.get("url")
                    or "#"
                )

                # ---------- PUBLISHED TIME ----------
                published_raw = (
                    item.get("providerPublishTime")
                    or item.get("pubDate")
                    or item.get("published")
                    or content.get("pubDate")
                )

                published = "Unknown time"
                if isinstance(published_raw, (int, float)):
                    try:
                        published = datetime.datetime.fromtimestamp(
                            published_raw
                        ).strftime("%d %b %Y %I:%M %p")
                    except Exception:
                        published = "Unknown time"
                elif isinstance(published_raw, str) and published_raw.strip():
                    published = published_raw

                # ---------- SENTIMENT (computed once, stored on the item) ----------
                sentiment = get_news_sentiment(title)

                news_items.append({
                    "title": title,
                    "link": link,
                    "published": published,
                    "sentiment": sentiment,
                })
            except Exception:
                # Skip this single malformed news item, keep the rest
                continue

        return news_items

    # --- Search input row ---
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_ticker = st.text_input(
            "Enter a stock ticker symbol",
            placeholder="e.g. AAPL, MSFT, TSLA, RELIANCE.NS",
            key="live_search_ticker",
        ).strip().upper()
    with search_col2:
        search_period = st.selectbox(
            "Chart period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2,
            key="live_search_period",
        )

    search_clicked = st.button("Search 🔎", key="live_search_btn")

    if search_clicked and search_ticker:
        try:
            with st.spinner(f"Fetching live data for {search_ticker}..."):
                live_hist, live_info = fetch_live_stock(search_ticker, search_period)

            if live_hist.empty:
                st.error(
                    f"No data found for `{search_ticker}`. "
                    "Double check the symbol (e.g. use `RELIANCE.NS` for NSE stocks)."
                )
            else:
                company_name = live_info.get("longName", search_ticker)
                current_live_price = live_hist["Close"].iloc[-1]
                prev_close = live_info.get(
                    "previousClose",
                    live_hist["Close"].iloc[-2] if len(live_hist) > 1 else current_live_price,
                )
                day_change_pct = ((current_live_price - prev_close) / prev_close) * 100 if prev_close else 0

                st.markdown(f"### {company_name} ({search_ticker})")

                # --- Live price + key stats ---
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("Live Price", f"{current_live_price:.2f}", f"{day_change_pct:.2f}%")
                stat_col2.metric(
                    "Market Cap",
                    f"{live_info.get('marketCap', 'N/A'):,}"
                    if isinstance(live_info.get("marketCap"), (int, float)) else "N/A",
                )
                stat_col3.metric(
                    "P/E Ratio",
                    f"{live_info.get('trailingPE', 'N/A'):.2f}"
                    if isinstance(live_info.get("trailingPE"), (int, float)) else "N/A",
                )
                stat_col4.metric(
                    "52W High / Low",
                    f"{live_info.get('fiftyTwoWeekHigh', 'N/A')} / {live_info.get('fiftyTwoWeekLow', 'N/A')}",
                )

                # --- Mini live price chart ---
                fig_live = go.Figure()
                fig_live.add_trace(go.Scatter(
                    x=live_hist.index, y=live_hist["Close"],
                    mode="lines", name="Close Price"
                ))
                fig_live.update_layout(
                    title=f"{search_ticker} — Last {search_period}",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    height=350,
                )
                st.plotly_chart(fig_live, use_container_width=True)

                # --- Latest News for the searched stock (FIXED SECTION) ---
                st.markdown("### 📰 Latest News")

                news_items = fetch_stock_news(search_ticker)

                # Save to session_state so the AI Recommendation section
                # (further down the page) can reuse this ticker's sentiment
                # without needing another search.
                st.session_state["live_news_items"] = news_items
                st.session_state["live_sentiment_score"] = get_average_sentiment_score(news_items)
                st.session_state["live_search_ticker_used"] = search_ticker

                if news_items:
                    for i, news in enumerate(news_items, start=1):
                        # Sentiment was already computed inside fetch_stock_news,
                        # so we just read it off the item here — no out-of-scope call.
                        st.markdown(f"**{i}. {news['title']}**")
                        st.caption(f"🕒 {news['published']}")
                        st.write(f"**Sentiment:** {news['sentiment']}")

                        if news.get("link") and news["link"] != "#":
                            st.markdown(f"[Read full article]({news['link']})")
                        else:
                            st.write("No article link available")

                        st.markdown("---")
                else:
                    st.info("No recent news found for this stock.")

                # --- Optional: download the fetched live data as CSV ---
                live_csv = live_hist.to_csv().encode("utf-8")
                st.download_button(
                    "📥 Download Live Data (CSV)",
                    data=live_csv,
                    file_name=f"{search_ticker}_live_data.csv",
                    mime="text/csv",
                    key="download_live_search_btn",
                )

        except Exception as e:
            st.error(f"Couldn't fetch data for `{search_ticker}`. Error: {e}")
    elif search_clicked and not search_ticker:
        st.warning("Please enter a ticker symbol first.")

st.markdown("---")
# =========================================================
# END OF LIVE STOCK SEARCH SECTION
# =========================================================

# -------------------------
# LOAD DATA  (EXISTING FEATURE — UNCHANGED)
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
# BUILD TARGET (next day's Close) FOR COMPARISON/BACKTEST (EXISTING — UNCHANGED)
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
# TRAIN / COMPARE MODELS (EXISTING — UNCHANGED, cached so this doesn't retrain every rerun)
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
# PREDICTION SECTION (EXISTING — UNCHANGED)
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
# 🏆 MODEL COMPARISON (EXISTING — UNCHANGED)
# -------------------------
st.subheader("🏆 Model Comparison")

st.dataframe(
    results_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "R2 Score": "{:.4f}"}),
    use_container_width=True,
)

st.success(f"Best Model : {best_model_name}")

# -------------------------
# 📊 MODEL BACKTESTING (EXISTING — UNCHANGED)
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
# 📈 ACTUAL vs PREDICTED (full history)  (EXISTING — UNCHANGED)
# -------------------------
st.subheader("📈 Actual vs Predicted")

full_preds = model.predict(df[feature_cols])

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual Price"))
fig1.add_trace(go.Scatter(x=df.index, y=full_preds, mode="lines", name="Predicted Price"))
st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# 🕯 CANDLESTICK CHART (EXISTING — UNCHANGED)
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
# INDICATORS (EXISTING — UNCHANGED)
# -------------------------
st.subheader("📊 Latest Technical Indicators")
indicator_cols = [c for c in ["Close", "SMA_10", "SMA_20", "EMA_10", "RSI", "MACD"] if c in df.columns]
st.dataframe(df[indicator_cols].tail(1))

# =========================================================
# 🤖 AI RECOMMENDATION  (NEW FEATURE)
# ---------------------------------------------------------
# Combines everything the dashboard already knows about the stock:
#   - the model's predicted next price vs the current price
#   - the latest RSI and MACD values
#   - the average news sentiment score from the Live Stock Search section
# into one simple, human-readable BUY / HOLD / SELL call.
#
# Notes:
#   - Prediction/current price come from the Prediction section above —
#     click "Predict Next Price 🚀" first if you haven't yet.
#   - RSI/MACD come from your local featured_stock_data.csv (same data
#     used everywhere else in the dashboard).
#   - News sentiment comes from whichever ticker you last looked up in
#     "Live Stock Search" above. If you haven't searched a ticker yet,
#     sentiment defaults to neutral (0.0) and the reason text notes this.
# =========================================================
st.subheader("🤖 AI Recommendation")

# Pull the four inputs from what's already been computed elsewhere in the app
rec_prediction = result["prediction"] if result else None
rec_current_price = result["current_price"] if result else df["Close"].iloc[-1]

latest_rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else None
latest_macd = df["MACD"].iloc[-1] if "MACD" in df.columns else None

sentiment_score = st.session_state.get("live_sentiment_score", 0.0)
sentiment_ticker = st.session_state.get("live_search_ticker_used")

if rec_prediction is None:
    st.info(
        "Click **Predict Next Price 🚀** in the Prediction section above to include "
        "the model's prediction in this recommendation."
    )

if sentiment_ticker is None:
    st.caption(
        "ℹ️ No ticker searched yet in Live Stock Search — using neutral (0.0) news sentiment. "
        "Search a ticker above to factor in real news sentiment."
    )
else:
    st.caption(f"📰 News sentiment based on your last search: **{sentiment_ticker}**")

signal, reason = generate_ai_recommendation(
    prediction=rec_prediction,
    current_price=rec_current_price,
    rsi=latest_rsi,
    macd=latest_macd,
    sentiment_score=sentiment_score,
)

if signal == "🟢 BUY":
    st.success(f"### {signal}")
elif signal == "🔴 SELL":
    st.error(f"### {signal}")
else:
    st.warning(f"### {signal}")

st.write(reason)

# =========================================================
# 📂 PORTFOLIO TRACKER / WATCHLIST  (NEW FEATURE)
# ---------------------------------------------------------
# Builds a compact summary table for every ticker the user typed into
# the sidebar "📂 Portfolio Tracker" box: live price, day change %,
# RSI, MACD, news sentiment, and an AI signal for each one.
#
# This re-uses the same building blocks as the rest of the app
# (fetch_live_stock, fetch_stock_news, get_average_sentiment_score,
# generate_ai_recommendation) so behavior stays consistent — it just
# runs them in a loop, once per ticker, instead of one at a time.
#
# Note: prediction=None is passed to generate_ai_recommendation here,
# since your trained ML model is built on your local CSV's features and
# doesn't apply to arbitrary tickers typed into the watchlist. The AI
# Recommendation section above still uses the real model prediction.
# =========================================================
st.subheader("📂 Portfolio Tracker / Watchlist")

if not YF_AVAILABLE:
    st.warning(
        "`yfinance` is not installed. Run `pip install yfinance` and restart the app "
        "to use the Portfolio Tracker."
    )
elif not watchlist_tickers:
    st.info("Add tickers in the sidebar under **📂 Portfolio Tracker** to see your watchlist here.")
else:
    watchlist_rows = []

    with st.spinner("Updating watchlist..."):
        for wl_ticker in watchlist_tickers:
            try:
                wl_hist, wl_info = fetch_live_stock(wl_ticker, "3mo")

                if wl_hist.empty:
                    watchlist_rows.append({
                        "Ticker": wl_ticker, "Company": "No data found", "Price": None,
                        "Change %": None, "RSI": None, "MACD": None,
                        "Sentiment": "N/A", "AI Signal": "N/A",
                    })
                    continue

                wl_price = wl_hist["Close"].iloc[-1]
                wl_prev_close = wl_info.get(
                    "previousClose",
                    wl_hist["Close"].iloc[-2] if len(wl_hist) > 1 else wl_price,
                )
                wl_change_pct = ((wl_price - wl_prev_close) / wl_prev_close) * 100 if wl_prev_close else 0

                wl_rsi = compute_rsi(wl_hist["Close"])
                wl_macd = compute_macd(wl_hist["Close"])

                wl_news = fetch_stock_news(wl_ticker)
                wl_sentiment_score = get_average_sentiment_score(wl_news)
                if wl_sentiment_score > 0.1:
                    wl_sentiment_label = "🟢 Positive"
                elif wl_sentiment_score < -0.1:
                    wl_sentiment_label = "🔴 Negative"
                else:
                    wl_sentiment_label = "🟡 Neutral"

                wl_signal, _ = generate_ai_recommendation(
                    prediction=None,  # no ML prediction for arbitrary watchlist tickers
                    current_price=wl_price,
                    rsi=wl_rsi,
                    macd=wl_macd,
                    sentiment_score=wl_sentiment_score,
                )

                watchlist_rows.append({
                    "Ticker": wl_ticker,
                    "Company": wl_info.get("longName", wl_ticker),
                    "Price": round(float(wl_price), 2),
                    "Change %": round(float(wl_change_pct), 2),
                    "RSI": round(wl_rsi, 2) if wl_rsi is not None else None,
                    "MACD": round(wl_macd, 2) if wl_macd is not None else None,
                    "Sentiment": wl_sentiment_label,
                    "AI Signal": wl_signal,
                })

            except Exception as e:
                # Keep going even if one ticker fails (bad symbol, network hiccup, etc.)
                watchlist_rows.append({
                    "Ticker": wl_ticker, "Company": f"Error: {e}", "Price": None,
                    "Change %": None, "RSI": None, "MACD": None,
                    "Sentiment": "N/A", "AI Signal": "N/A",
                })

    watchlist_df = pd.DataFrame(watchlist_rows)
    st.dataframe(watchlist_df, use_container_width=True)

    # --- Download the watchlist summary as CSV ---
    watchlist_csv = watchlist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Watchlist (CSV)",
        data=watchlist_csv,
        file_name="portfolio_watchlist.csv",
        mime="text/csv",
        key="download_watchlist_btn",
    )

# -------------------------
# FOOTER (EXISTING — UNCHANGED)
# -------------------------
st.markdown("---")
st.success("Dashboard Running Successfully 🚀")
