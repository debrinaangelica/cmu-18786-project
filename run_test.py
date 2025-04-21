import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import data as dataset
from model import LSTM
from plotters import plot_predictions

def main():
    # === Hyperparameters ===
    input_dim = 3
    hidden_dim = 64
    output_dim = 1
    num_layers = 3

    batch_size = 16

    data_splits = (70,20,10) # train/validation/test :: 70/20/10
    sequence_length = 20
    # =======================

    # Architecture:
    #   - normalization: min max
    #   - loss function: l2, mae, 
    #   - optimizer: adam?

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()

    train_data, valid_data, test_data = dataset.create_dataset(tweet_data_src='data/sentiment/daily_sentiment_summary.csv', sequence_length=sequence_length, data_splits=data_splits)
    
    input_dim = test_data.get_input_dim()

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    test_rmse = 0.0 # test rsme
    test_predictions = []


    # Test - Load the best model params to run test
    model.load_state_dict(torch.load("model_params_vader/model.params", map_location=device))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for data, y_true in test_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            y_pred = model(data)
            test_loss = criterion(data, y_true)

            total_test_loss += test_loss.item() * data.size(0)

            test_predictions.append(y_pred.cpu().numpy())

        test_rmse = np.sqrt(total_test_loss / len(test_data))

    predictions = np.concatenate(test_predictions, axis=0)
    
    plot_predictions(y_true=test_data.y, y_pred=predictions, test_rmse=test_rmse, save_name='predictions_vader.png')

    # Print Test Loss
    print(f"Test Loss: {round(test_rmse, 2)}")


if __name__ == "__main__":
    main()