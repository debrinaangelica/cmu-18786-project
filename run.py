import numpy as np
import torch

from data import *
import model

def main():
    tweets = get_tweet_dataset("data/tweets/elon_musk_tweets.csv")
    print(tweets.head())
    
    input_dim = 0
    hidden_dim = 0
    output_dim = 0
    num_layers = 0

    model = model.LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    # TODO: 
    #   - how should be process the data prior to passing it into the RNN?
    #   - what data features should we be passing in?

if __name__ == "__main__":
    main()