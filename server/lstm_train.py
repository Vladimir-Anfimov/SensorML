import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_frames import FrameLoader
from torch.utils.data.dataset import random_split

from params import NORMALIZED_PARAMS_NAMES, OUTPUT_SIZE, WINDOW_SIZE

device = torch.device('cpu')



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, window_size=WINDOW_SIZE, output_size=OUTPUT_SIZE):
        self.df = df
        self.window_size = window_size
        self.output_size = output_size

    def __len__(self):
        return len(self.df) - self.window_size - self.output_size + 1
    

    def __getitem__(self, idx):
        x = self.df[idx:idx+self.window_size].to_numpy()
        y = self.df[idx+self.window_size:idx+self.window_size+self.output_size].to_numpy()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


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
    

def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for seq, labels in dataloader:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def test(model, dataloader, criterion):
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            output = model(x)
            error = criterion(output, y)
            print(f'Error: {error.item():.4f}')


    
if __name__ == '__main__':
    dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
    df = dataframe_loader.load()
    
    dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
    df = dataframe_loader.load()
    dataset = Dataset(df)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    model = RecurentNeuralNetwork()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, dataloader_train, criterion, optimizer, num_epochs=5)

    test(model, dataloader_test, criterion)

    torch.save(model.state_dict(), './models/lstm/lstm_normalized.pth')
