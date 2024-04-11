import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.optim as optim
import pickle as pk
from torch.utils.data.sampler import BatchSampler


with open('/scratch/dzinga/tensors_final.pkl', 'rb') as file:
    loaded_tensor_dict = pk.load(file)
print("Data Loaded.")

train_sessions = loaded_tensor_dict['train_sessions']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SessionBatchSampler(BatchSampler):
    def __init__(self, session_data, batch_size, sequence_len):
        self.session_data = session_data
        self.num_of_features = session_data[0].shape[1]
        self.batch_size = batch_size
        self.sequence_len = sequence_len

    def __iter__(self):
        # When the sampler is called
        # for each session
        for idx in range(len(self.session_data)):
            items = self.session_data[idx]
            # split the session in batches
            for i in range(0, len(items), self.batch_size):
                batch_items = items[i:i+self.batch_size]
                # if the size of the batch has enough elements to look back as set
                if len(batch_items) - self.sequence_len > 0:
                    # initialise the batch input x and output y
                    batch_x = torch.zeros((len(batch_items) - self.sequence_len, self.sequence_len, self.num_of_features))
                    batch_y = torch.zeros((len(batch_items) - self.sequence_len, self.num_of_features))
                    # generate the sequence of input x and then the ouput y
                    for j in range(len(batch_items) - self.sequence_len):
                        batch_x[j] = batch_items[j:j+self.sequence_len]
                        batch_y[j] = batch_items[j+self.sequence_len]
                    # return the example
                    yield batch_x, batch_y

    def __len__(self):
        count = 0
        for _, items in self.session_data.grouped_data.items():
            count += (len(items) + self.batch_size - 1) // self.batch_size
        return count

session_batch_sampler = SessionBatchSampler(train_sessions, batch_size=32, sequence_len=10)

class LstmModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_of_layers, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_of_layers,
            batch_first=True,
            dropout=0.2
        )
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        # This resets the hidden and cell states of the LSTM
        # Should be done for each batch
        # assuming that the batch is a separate sequence
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_of_layers, batch_size, self.hidden_channels).requires_grad_().to(device)
        c0 = torch.zeros(self.num_of_layers, batch_size, self.hidden_channels).requires_grad_().to(device)

        # LSTM
        x, _ = self.lstm(x, (h0, c0))

        # Extract last prediction and feed that through the MLP
        x = self.linear(x[:, -1, :])

        return x

lstm_hidden_channels = 128
lstm_layers = 3
lstm_trained_epoch = 0

model = LstmModel(
    in_channels=33,
    hidden_channels=lstm_hidden_channels,
    num_of_layers=lstm_layers,
    out_channels=33
    ).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

print('Training...')

n_epochs = 5000
for epoch in range(n_epochs):
    lstm_trained_epoch += 1
    model.train()
    epoch_loss = 0
    epoch_batch_count = 0
    for x_batch, y_batch in session_batch_sampler:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # Uncomment to print batch shapes
        # print(f'Predictor batch shape: {x_batch.shape}')
        # print(f'Outcome batch shape: {y_batch.shape}')
        y_pred = model(x_batch)
        # Uncomment to print predictions
        # print(y_pred)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.float()
        epoch_batch_count += 1
    print("Epoch %d: train RMSE %.4f" % (lstm_trained_epoch, epoch_loss / epoch_batch_count))
    
    # Save the model
model_save_path = '/data/dzinga/LSTM_final.pt'
torch.save(model.state_dict(), model_save_path)

print("Done")
