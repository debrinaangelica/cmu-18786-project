import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch

from data import *
import model

def main():
    tweets = get_tweet_dataset("data/tweets/elon_musk_tweets.csv")
    print(tweets.head())
    
    # Hyperparameters
    input_dim = 0
    hidden_dim = 0
    output_dim = 0
    num_layers = 0

    batch_size = 8

    model = model.LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    # TODO: 
    #   - how should we process the data prior to passing it into the RNN?
    #   - what data features should we be passing in?

    # TODO:
    #   - normalization: min max
    #   - loss function: l2, mae, 
    #   - optimizer: adam?

    train_data, val_data, test_data = None, None, None
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    
    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))

        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

if __name__ == "__main__":
    main()