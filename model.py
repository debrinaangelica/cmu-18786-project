import torch
import torch.nn as nn

# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2 
output_dim = 1


# Sample from: 
# https://www.kaggle.com/code/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch
class LSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1, output_dim=1):
        super(LSTM, self).__init__()
        """
        sequence length --> window size (number of days to consider)
        input features --> input_dim=4: 
            - open price (normalized with the function from paper#1)
            - closing price (normalized with the function from paper#1)
            - sentiment score
            - # of tweets for this day
        maybe also:
            - trade volume for the day
            - other prices (opening, %change, highest, adjusted close)

        output features --> output_dim=1:
            - %change of the next day
        """
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first: If True, then the input and output tensors are provided 
        # as (batch, seq, feature) instead of (seq, batch, feature). 
        # Note that this does not apply to hidden or cell states. 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)        

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        return out
    
