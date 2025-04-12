import yfinance as yf
import pandas as pd

def get_stock_data(ticker="TSLA"):
    df = yf.download(ticker, interval="1d")
    df.columns = df.columns.droplevel(-1)  # Remove random second header row
    df = df.rename(columns={
        "Close": "close",
        "High": "high",
        "Low": "low",
        "Open": "open",
        "Volume": "volume"
    })
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    # Min-max normalization
    df["close_minmax"] = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())
    df["open_minmax"] = (df["open"] - df["open"].min()) / (df["open"].max() - df["open"].min())

    # Daily % change
    df["change_open_to_close"] = (df["close"] - df["open"]) / df["open"]
    df["change_close_to_close"] = df["close"] / df["close"].shift(1) - 1

    return df

if __name__ == "__main__":
    df = get_stock_data()
    df.to_csv("data/stock/tsla.csv")