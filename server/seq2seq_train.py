import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_frames import FrameLoader

from params import NORMALIZED_PARAMS_NAMES

device = torch.device('cpu')

OUTPUT_SIZE = 30
WINDOW_SIZE = 7

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, expected_column, window_size=WINDOW_SIZE, output_size=OUTPUT_SIZE):
        self.x = df.values
        self.y = df[expected_column].values
        self.window_size = window_size
        self.output_size = output_size

    def __len__(self):
        return len(self.x) - self.window_size - self.output_size
    

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx:idx + self.window_size], dtype=torch.float32, device=device),
                torch.tensor(self.y[idx + self.window_size: idx + self.window_size + self.output_size], dtype=torch.float32, device=device))
    

class Seq2Seq(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = torch.nn.LSTM(input_size, hidden_size)
        self.decoder = torch.nn.LSTM(hidden_size, output_size)
        self.fc = torch.nn.Linear(output_size, output_size)

    def forward(self, input_seq):
        _, (hidden, cell) = self.encoder(input_seq)
        output, _ = self.decoder(input_seq[-1].view(1, 1, -1), (hidden, cell))
        output = self.fc(output.squeeze(0))
        return output
    

PREDICTED_COLUMN = 'pres'

dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
df = dataframe_loader.load()
dataset = Dataset(df, PREDICTED_COLUMN)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

input_size = 19
hidden_size = 16
output_size = 30
model = Seq2Seq(input_size, hidden_size, output_size)

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    for seq, labels in dataloader:
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

