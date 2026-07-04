import pandas as pd

def load_featured_data(file_path="data/processed/featured_stock_data.csv"):
    df = pd.read_csv(file_path)

    # Next day target
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    # Feature set for next-day forecasting
    X = df[[
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
    ]]

    y = df["Target"]

    return X, y, df
