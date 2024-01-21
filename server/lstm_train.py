import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_frames import FrameLoader

from params import NORMALIZED_PARAMS_NAMES

device = torch.device('cpu')

OUTPUT_SIZE = 30
WINDOW_SIZE = 7

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size=WINDOW_SIZE, output_size=OUTPUT_SIZE):
        self.x = df.values
        self.y = df.values
        self.window_size = window_size
        self.output_size = output_size

    def __len__(self):
        return len(self.x) - self.window_size - self.output_size
    

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx:idx + self.window_size], dtype=torch.float32, device=device),
                torch.tensor(self.y[idx + self.window_size: idx + self.window_size + self.output_size], dtype=torch.float32, device=device))


class RecurentNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size=19, hidden_size=32, num_layers=2, output_size=OUTPUT_SIZE):
        super(RecurentNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size * len(NORMALIZED_PARAMS_NAMES))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        out = out.view(-1, self.output_size, len(NORMALIZED_PARAMS_NAMES))
        return out
    

def train(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}')


def test(model, dataloader, criterion):
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            output = model(x)
            error = criterion(output, y)
            print(f'Error: {error.item():.4f}')


    
if __name__ == '__main__':
    dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
    df = dataframe_loader.load()
    
    model = RecurentNeuralNetwork().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = Dataset(df)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    train(model, dataloader, criterion, optimizer, epochs=5)

    torch.save(model.state_dict(), f'./models/lstm.pth')

    test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    test(model, test_dataloader, criterion)
