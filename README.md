# 📈 StockTrend AI

An **AI-powered stock market analysis dashboard** built with **Streamlit**, designed to help users explore live stock prices, analyze market sentiment, compare ML models, and generate stock price predictions with interactive visualizations.

---

## 🚀 Live Demo
👉 Add your deployed Streamlit app link here  
Example: `https://your-stocktrend-ai.streamlit.app`

---

## 📌 Project Overview

**StockTrend AI** is a stock prediction and analysis dashboard that combines:

- **Live stock market data**
- **Machine learning price prediction**
- **News sentiment analysis**
- **Technical indicators**
- **Portfolio/watchlist tracking**
- **Interactive charts and backtesting**

The goal of this project is to provide a **single intelligent dashboard** where users can search stocks, understand market trends, view recent news, and get AI-driven insights like **BUY / HOLD / SELL recommendations**.

---

# ✨ Features

## 🔍 1. Live Stock Search
- Search any stock ticker such as:
  - `AAPL`
  - `TSLA`
  - `MSFT`
  - `NVDA`
  - `RELIANCE.NS`
- Fetches **live historical stock data** using `yfinance`

---

## 📊 2. Live Stock Price Chart
- Interactive stock price chart
- Visualizes recent stock movement over time
- Helps users quickly understand trend direction

---

## 📰 3. Latest News Fetching
- Fetches recent news related to the selected stock
- Displays article titles, published time, and source links

---

## 😊 4. News Sentiment Analysis
- Uses **TextBlob** to analyze stock news sentiment
- Classifies news as:
  - 🟢 Positive
  - 🟡 Neutral
  - 🔴 Negative

---

## 🤖 5. AI Stock Price Prediction
- Predicts the **next stock price** using machine learning
- Supports:
  - **Latest Data prediction**
  - **Manual row selection prediction**

---

## 💡 6. AI Recommendation Engine
Based on predicted price vs current price, the app generates:
- 🟢 **BUY**
- 🟡 **HOLD**
- 🔴 **SELL**

---

## 🏆 7. Model Comparison
Compares multiple machine learning models such as:
- **Random Forest Regressor**
- **Linear Regression**
- **XGBoost** *(if installed)*

Evaluation metrics shown:
- MAE
- MSE
- R² Score

---

## 🔥 8. Backtesting
Runs a simple strategy backtest to compare:
- **AI strategy return**
- **Buy & Hold return**

This helps evaluate how the prediction strategy performs historically.

---

## 🕯 9. Candlestick Chart
- Interactive candlestick chart for OHLC stock data
- Useful for technical analysis and price action visualization

---

## 📉 10. Technical Indicators
Displays the latest values of indicators like:
- SMA 10
- SMA 20
- EMA 10
- RSI
- MACD

---

## 📂 11. Portfolio Tracker / Watchlist
Users can enter multiple stocks and analyze them together in a summary table.

Example watchlist:
AAPL, TSLA, MSFT, NVDA

The watchlist can show:
- current price
- predicted price
- recommendation
- sentiment

---

## 📥 12. CSV Report Downloads
Download prediction results and/or stock data in CSV format.

---

# 🛠 Tech Stack

## Frontend / App Framework
- **Streamlit**

## Data & Visualization
- **Pandas**
- **NumPy**
- **Plotly**
- **Matplotlib**

## Machine Learning
- **Scikit-learn**
- **XGBoost**
- **Joblib**

## Finance / Data Source
- **yfinance**

## NLP / Sentiment
- **TextBlob**

## Parsing / Web Utilities
- **Requests**
- **BeautifulSoup4**

---

# 🧠 Machine Learning Models Used

This project experiments with multiple regression models for stock price prediction:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

The app compares these models and identifies the best-performing one using evaluation metrics.

---

# 📂 Project Structure

```bash
stocktrend-ai/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit dashboard
│
├── data/
│   └── processed/
│       └── featured_stock_data.csv
│
├── models/
│   └── random_forest.pkl
│
├── notebooks/                    # Experiments / training notebooks
├── utils/                        # Helper scripts
├── assets/                       # Images / plots / screenshots
│
├── requirements.txt
├── README.md
└── .gitignore