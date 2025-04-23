import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import data as dataset
importlib.reload(dataset)

from model import LSTM

def main():
    # === Hyperparameters ===
    input_dim = 4
    hidden_dim = 64
    output_dim = 1
    num_layers = 4

    num_epochs = 2000
    batch_size = 16
    learning_rate = 0.001

    data_splits = (70,20,10) # train/validation/test :: 70/20/10
    sequence_length = 50
    # =======================

    # Architecture:
    #   - normalization: min max
    #   - loss function: l2, mae,
    #   - optimizer: adam?

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()

    train_data, valid_data, test_data = dataset.create_dataset(tweet_data_src='data/sentiment/daily_sentiment_summary.csv', sequence_length=sequence_length, data_splits=data_splits, new_dataset=True)
    print(train_data.dates[0])
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    train_rmse = 0.0
    train_predictions = []

    model.load_state_dict(torch.load("model_params/model.params", map_location=device))
    model.eval()
    total_train_loss = 0.0
    with torch.no_grad():
        for data, y_true in train_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            y_pred = model(data)
            train_loss = criterion(y_pred.squeeze(-1), y_true)

            total_train_loss += train_loss.item() * data.size(0)

            train_predictions.append(y_pred.cpu().numpy())
        
        train_rmse = np.sqrt(total_train_loss / len(train_data))

    predictions = np.concatenate(train_predictions, axis=0)

    plot_predictions(y_true=train_data.y[sequence_length:], y_pred=predictions, test_rmse=train_rmse, start_date=train_data.dates[sequence_length], end_date=train_data.dates[-1], set_type="Train")

    valid_rmse = 0.0
    valid_predictions = []
    
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for data, y_true in valid_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            y_pred = model(data)
            valid_loss = criterion(y_pred.squeeze(-1), y_true)
            total_valid_loss += valid_loss.item() * data.size(0)

            valid_predictions.append(y_pred.cpu().numpy())

        valid_rmse = np.sqrt(total_valid_loss / len(valid_data))

    predictions = np.concatenate(valid_predictions, axis=0)
    
    plot_predictions(y_true=valid_data.y[sequence_length:], y_pred=predictions, test_rmse=valid_rmse, start_date=valid_data.dates[sequence_length], end_date=valid_data.dates[-1], set_type='Validation')

    test_rmse = 0.0 # test rsme
    test_predictions = []

    # Test - Load the best model params to run test
    model.load_state_dict(torch.load("model_params/model.params", map_location=device))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for data, y_true in test_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            y_pred = model(data)
            test_loss = criterion(y_pred.squeeze(-1), y_true)

            total_test_loss += test_loss.item() * data.size(0)

            test_predictions.append(y_pred.cpu().numpy())

        test_rmse = np.sqrt(total_test_loss / len(test_data))

    predictions = np.concatenate(test_predictions, axis=0)

    plot_predictions(y_true=test_data.y[sequence_length:], y_pred=predictions, test_rmse=test_rmse, start_date=test_data.dates[sequence_length], end_date=test_data.dates[-1])

    # Print Train Loss
    print(f"Train Loss: {round(train_rmse, 2)}")
    # Print Validation Loss
    print(f"Validation Loss: {round(valid_rmse, 2)}")
    # Print Test Loss
    print(f"Test Loss: {round(test_rmse, 2)}")

# Plotting Functions
def plot_predictions(y_true, y_pred, test_rmse, start_date, end_date, set_type='Test'):
    plt.plot(y_true, label="True Closing Price")
    plt.plot(y_pred, label="Predicted Closing Price")
    plt.xlabel('day')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    title = set_type + f" Predictions vs. True Stock Closing Price | \n{start_date.astype('datetime64[D]')} to {end_date.astype('datetime64[D]')} (rmse={test_rmse})"
    plt.title(title)
    plt.savefig(set_type + f"_predictions.png")
    plt.show()


main()