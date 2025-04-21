from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import torch

import yfinance as yf
import pandas as pd
import stock

from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class LSTMDataset(Dataset):
    def __init__(self, dates, X, y, sequence_length=50, stock_close_prices=[], logger=None):
        """
        data: all the data (for the entire time period)
        """
        self.dates = dates
        self.sequence_length = sequence_length
        self.X = X
        self.y = y
        self.stock_close_prices = stock_close_prices
        self.logger = logger

    def __len__(self):
        # number of sequences = total days - window size
        # 
        length = len(self.X) - self.sequence_length
        if length < 0:
            if self.logger is not None:
                self.logger.warning(f"Dataset length ({len(self.X)}) is less than window size ({self.sequence_length})")
            print(f"Dataset length ({len(self.X)}) is less than window size ({self.sequence_length})")
            return 0

        return length

    def __getitem__(self, idx):
        """
        For each index, returns:
          - x: a sequence of days of shape (sequence_length, num_features)
          - y: the label for the day immediately following the sequence. For example,
               using the closing price (assumed to be the first feature) as the label.
        """
        # Extract sequence length datapoints at a time
        x = self.X[idx : idx + self.sequence_length]
        # Label is the closing price of the day after the sequence.
        y = self.y[idx + self.sequence_length]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y

    def get_input_dim(self):
        return self.X.shape[1]


def DEPRECATED_get_stock_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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

def DEPRECATED_create_dataset(tweet_data_src='data/sentiment/tweets_with_finbert_sentiment.csv', sequence_length=50, data_splits=(70,20,10)):
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
    # TODO: update this to use mingkai's stock.py version
    stock_data = stock.get_stock_data('TSLA', start_date='2023-01-01', end_date='2024-01-01')
    close_prices = stock_data['Close']
    open_prices = stock_data['Open']
    trade_volumes = stock_data['Volume'] # don't need this (?)
    
    # Tweet data (sentiment)
    daily_averaged_sentiment_scores = []
    daily_tweet_counts = []

    dataset = np.stack((close_prices, open_prices, daily_averaged_sentiment_scores, daily_tweet_counts), axis=1)

    data_len = len(dataset) 
    split_train = int(data_len * (data_splits[0]/100)) # end index of train split
    split_valid = split_train+int(data_len * (data_splits[1]/100)) # end index of validation split
    return LSTMDataset(dataset[:split_train], sequence_length), LSTMDataset(dataset[split_train:split_valid], sequence_length), LSTMDataset(dataset[split_valid:], sequence_length)

def create_dataset(tweet_data_src='data/sentiment/daily_sentiment_summary.csv', sequence_length=50, data_splits=(70,20,10)):
    # Define file paths (adjust these paths as necessary)
    stock_csv = 'data/stock/tsla.csv'

    # Load the sentiment CSV file, parsing the 'date' column as datetime objects
    sentiment_df = pd.read_csv(tweet_data_src, parse_dates=['day'])
    sentiment_df.rename(columns={'day': 'date'}, inplace=True)

    # Load the stock CSV file, also parsing the 'date' column
    stock_df = pd.read_csv(stock_csv, parse_dates=['date'])

    # Extract only the 'date' and 'change_close_to_close' columns from the stock data
    stock_subset = stock_df[['date', 'open_minmax', 'close_minmax']]
    y_values = stock_df[['date', 'close']]

    # # Merge the sentiment data with the selected stock data on the 'date' column.
    # # A left merge ensures all rows from sentiment_df are kept, and the corresponding 
    # # change_close_to_close value is appended.
    # dataset = pd.merge(sentiment_df, stock_subset, on='date', how='left')

    # Merge using an inner join so that only rows with matching dates in both datasets remain.
    # NOTE: no stock data on weekends
    # TODO: consider averaging sentiment data over the weekend so that we don't completely ignore them.
    dataset = pd.merge(stock_subset, sentiment_df, on='date', how='inner')
    dataset = pd.merge(dataset, y_values, on='date', how='inner')

    # Get close prices (for plotting predictions later on)
    stock_close_prices = dataset['close'].to_numpy()

    date_array = dataset['date'].to_numpy()

    # Get rid of 'date' column
    dataset.drop('date', axis=1, inplace=True)
    # Get rid of 'close' column (not used as input)
    dataset.drop('close', axis=1, inplace=True)

    # TODO: Temporarily don't use the prob_negative,prob_neutral,prob_positive columns
    dataset.drop('prob_negative', axis=1, inplace=True)
    dataset.drop('prob_neutral', axis=1, inplace=True)
    dataset.drop('prob_positive', axis=1, inplace=True)

    # Save so we can check whether the formatting is correct
    dataset.to_csv("current_dataset.csv", index=False)

    # Convert to tensor
    data_tensor = torch.tensor(dataset.values, dtype=torch.float32)
    # Split data and labels
    X = data_tensor[:, :-1]
    y = data_tensor[:, -1]

    data_len = len(X) 
    split_train = int(data_len * (data_splits[0]/100)) # end index of train split
    split_valid = split_train+int(data_len * (data_splits[1]/100)) # end index of validation split
    
    train_data = LSTMDataset(date_array[:split_train], X[:split_train], y[:split_train], sequence_length)
    val_data = LSTMDataset(date_array[split_train-sequence_length:split_valid], X[split_train-sequence_length:split_valid], y[split_train-sequence_length:split_valid], sequence_length)
    test_data = LSTMDataset(date_array[split_valid-sequence_length:], X[split_valid-sequence_length:], y[split_valid-sequence_length:], sequence_length)

    return train_data, val_data, test_data

def create_dataset_vader(tweet_data_src='data/sentiment/daily_sentiment_summary_vader.csv', sequence_length=50, data_splits=(70,20,10)):
    # Define file paths (adjust these paths as necessary)
    stock_csv = 'data/stock/tsla.csv'

    # Load the sentiment CSV file, parsing the 'date' column as datetime objects
    sentiment_df = pd.read_csv(tweet_data_src, parse_dates=['day'])
    sentiment_df.rename(columns={'day': 'date'}, inplace=True)

    # Load the stock CSV file, also parsing the 'date' column
    stock_df = pd.read_csv(stock_csv, parse_dates=['date'])

    # Extract 'date', 'open', 'close' columns from the stock data
    stock_subset = stock_df[['date', 'open', 'close', 'open_minmax', 'close_minmax']]
    y_values = stock_df[['date', 'close']]  # We are predicting 'close' prices

    # Merge using an inner join so that only rows with matching dates in both datasets remain.
    dataset = pd.merge(stock_subset, sentiment_df, on='date', how='inner')
    dataset = pd.merge(dataset, y_values, on='date', how='inner')

    # Handle missing dates in the stock data (weekends, holidays, etc.)
    final_dataset = []
    last_sentiment = None  # To store the last valid sentiment for missing days
    last_stock_data = None  # To store the last valid stock data for missing days
    sentiment_count = 0  # To track the number of days without stock data

    for idx, row in dataset.iterrows():
        # If it's the first row or if stock data is available for this date, just append the row
        if last_sentiment is None or not pd.isna(row['open_minmax']):
            final_dataset.append(row)
            last_sentiment = row[['sentiment_score', 'prob_negative', 'prob_neutral', 'prob_positive']].copy()
            last_stock_data = row[['open', 'close', 'open_minmax', 'close_minmax']].copy()  # Include open and close prices
            sentiment_count = 1  # Reset the count as we've encountered valid stock data
        else:
            # Average sentiment scores if the current day doesn't have stock data (weekend/holiday)
            sentiment_count += 1
            last_sentiment = (last_sentiment * (sentiment_count - 1) + row[['sentiment_score', 'prob_negative', 'prob_neutral', 'prob_positive']]) / sentiment_count
            row[['sentiment_score', 'prob_negative', 'prob_neutral', 'prob_positive']] = last_sentiment
            row[['open', 'close', 'open_minmax', 'close_minmax']] = last_stock_data  # Use the last available stock data
            final_dataset.append(row)

    # Convert final_dataset back to a DataFrame
    final_dataset_df = pd.DataFrame(final_dataset)

    # Get close prices (for plotting predictions later on)
    stock_close_prices = final_dataset_df['close'].to_numpy()
    date_array = final_dataset_df['date'].to_numpy()

    # Get rid of 'date' column
    final_dataset_df.drop('date', axis=1, inplace=True)
    # Get rid of 'close' column (not used as input)
    # final_dataset_df.drop('close', axis=1, inplace=True)

    # Save so we can check whether the formatting is correct
    final_dataset_df.to_csv("current_dataset.csv", index=False)

    # Split data and labels
    y = torch.tensor(final_dataset_df['close'].values, dtype=torch.float32)
    X = torch.tensor(final_dataset_df.values, dtype=torch.float32)  # All columns

    data_len = len(X)
    split_train = int(data_len * (data_splits[0] / 100))  # end index of train split
    split_valid = split_train + int(data_len * (data_splits[1] / 100))  # end index of validation split

    train_data = LSTMDataset(date_array[:split_train], X[:split_train], y[:split_train], sequence_length=sequence_length, stock_close_prices=stock_close_prices)
    val_data = LSTMDataset(date_array[split_train - sequence_length:split_valid], X[split_train - sequence_length:split_valid], y[split_train - sequence_length:split_valid], sequence_length=sequence_length, stock_close_prices=stock_close_prices)
    test_data = LSTMDataset(date_array[split_valid - sequence_length:], X[split_valid - sequence_length:], y[split_valid - sequence_length:], sequence_length=sequence_length, stock_close_prices=stock_close_prices)

    return train_data, val_data, test_data

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

def get_close_prices_from_percent_changes(y_true_close, y_pred_percent_change):
    """
    the dates for y_true_close should lag y_pred_percent_change by 1 day
    """
    assert len(y_true_close) == len(y_pred_percent_change)

    y_pred_close = []
    for i, y in enumerate(y_true_close):
        y_pred_close.append(round(y*(1+y_pred_percent_change[i]), 2))

    return y_pred_close


def main():
    tweets = get_tweet_dataset("data/tweets/elon_musk_tweets.csv")
    print(tweets.head())

if __name__ == "__main__":
    main()