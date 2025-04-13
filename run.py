import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import data as dataset
from model import LSTM
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='LSTM_Information.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Things to log:
#   - train, validation rmse (every n epochs)
#   - test rmse
#   - train, valid, test predictions (once)
#   - save model parameters every {interval}

def main():
    # === Hyperparameters ===
    input_dim = 3
    hidden_dim = 50
    output_dim = 1
    num_layers = 1

    num_epochs = 2000
    batch_size = 16
    learning_rate = 0.001

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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_data, valid_data, test_data = dataset.create_dataset(tweet_data_src='data/sentiment/daily_sentiment_summary.csv', sequence_length=sequence_length, data_splits=data_splits)
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    train_rmse = [] # train rmse
    valid_rmse = [] # validation rmse
    test_rmse = 0.0 # test rsme
    
    best_valid_loss = None
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        total_loss = 0.0
        total_valid_loss = 0.0

        for data, y_true in train_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
        
        train_loss = total_loss / len(train_data)
        train_rmse.append(np.sqrt(train_loss))

        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                for data, y_true in val_loader:
                    data = data.to(device)
                    y_true = y_true.to(device)
                    y_pred = model(data)
                    valid_loss = criterion(y_pred, y_true)

                    total_valid_loss += valid_loss.item() * data.size(0)
                valid_loss = total_valid_loss / len(valid_data)
                valid_rmse.append(np.sqrt(valid_loss))
                if best_valid_loss == None or valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    # Save the model params
                    torch.save(model.state_dict(), "model_params/model.params")
                
        if epoch % 100 == 0:
            logger.info('Epoch %d: Train RMSE %.4f, Valid RMSE %.4f', epoch, train_rmse[-1], valid_rmse[-1])

            # Update tqdm's description with current metrics; using tqdm.write writes outside the progress bar.
            tqdm.write(f"Epoch {epoch}: Train RMSE {train_rmse[-1]:.4f}, Valid RMSE {valid_rmse[-1]:.4f}")

    # Test - Load the best model params to run test
    model.load_state_dict(torch.load("model_params/model.params", map_location=device))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for data, y_true in test_loader:
            data = data.to(device)
            y_true = y_true.to(device)
            y_pred = model(data)
            test_loss = criterion(y_pred, y_true)

            total_test_loss += test_loss.item() * data.size(0)
        test_rmse = np.sqrt(total_test_loss / len(test_data))

    # Plot Train Loss
    plot_loss(train_rmse, 'Training Loss', save_name='train_loss.png')
    # Plot Validation Loss
    plot_loss(valid_rmse, 'Validation Loss', save_name='validation_loss.png')
    # Print Test Loss
    print(f"Test Loss: {round(test_rmse, 2)}")
    logger.info(f"Test Loss: {round(test_rmse, 2)}")

# Plotting Functions
def plot_loss(losses, title='Training Loss', save_name='plot_loss.png'):
    plt.plot(losses, label=title)
    plt.xlabel('epoch')
    plt.ylabel('rmse loss')
    plt.legend()
    plt.grid(True)
    title = f"{title}\n(epoch: {len(losses)}, loss: {round(losses[-1], 2)})"
    plt.title(title)
    # plt.show()
    plt.savefig(save_name)

if __name__ == "__main__":
    main()
