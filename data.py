from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import torch

import yfinance as yf
import pandas as pd


def get_stock_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance for a given ticker symbol between start_date and end_date.
    Applies min-max normalization to the 'Open', 'Close', and 'Volume' columns.

    Parameters:
        ticker_symbol (str): The ticker symbol (e.g., 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format (e.g., '2023-01-01').
        end_date (str): The end date in 'YYYY-MM-DD' format (e.g., '2023-06-01').

    Returns:
        pd.DataFrame: A DataFrame with normalized 'Open', 'Close', and 'Volume' data.
    """
    # Create a Ticker object for the given symbol
    ticker = yf.Ticker(ticker_symbol)
    
    # Download historical data between the specified dates
    data = ticker.history(start=start_date, end=end_date)
    
    # Select only the columns we need: 'Open', 'Close', and 'Volume'
    data = data[['Open', 'Close', 'Volume']]
    
    # Apply min-max normalization to each column: normalized_x = (x - min(X)) / (max(X) - min(X))
    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        # Avoid division by zero if max equals min:
        if max_val != min_val:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0.0

    return data


def create_dataset(tweet_data_src='data/sentiment/tweets_with_finbert_sentiment.csv', lookback=10):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """

    # TODO:
    #   1. get the # of tweets for each day 
    #   2. get the sentiment score for each day (average all the scores for the day?)
    #   3. compile this to one dataset

    # Stock data
    stock_data = get_stock_data('TSLA', start_date='2023-01-01', end_date='2024-01-01')
    close_prices = stock_data['Close']
    open_prices = stock_data['Open']
    trade_volumes = stock_data['Volume'] # don't need this
    # Tweet data (sentiment)
    polarities = []
    daily_tweet_counts = []



    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)

def get_tweet_dataset(filename):
    tweets = pd.read_csv(filename)
    tweets = tweets[["date", "text"]]
    tweets = _append_sentiment_analysis(tweets)
    return tweets

def _append_sentiment_analysis(dataset):
    analyzer = SentimentIntensityAnalyzer(lexicon_file="data/sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt")
    for t in dataset["text"]:
        # Hashtags are not taken into consideration by analyzer.polarity_scores
        t = t.replace("#", "")
        dataset["scores_pos"] = analyzer.polarity_scores(t)["pos"]
        dataset["scores_neg"] = analyzer.polarity_scores(t)["neg"]
        dataset["scores_neu"] = analyzer.polarity_scores(t)["neu"]
        dataset["scores_compound"] = analyzer.polarity_scores(t)["compound"]
    return dataset

def main():
    tweets = get_tweet_dataset("data/tweets/elon_musk_tweets.csv")
    print(tweets.head())

if __name__ == "__main__":
    main()