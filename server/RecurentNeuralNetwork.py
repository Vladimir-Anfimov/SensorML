import pandas as pd
import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, expected_column, window_size=10):
        self.x = df.drop(expected_column, axis=1).values
        self.y = df[expected_column].values
        self.window_size = window_size

    def __len__(self):
        return len(self.x) - self.window_size
    

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx:idx + self.window_size], dtype=torch.float32),
                torch.tensor(self.y[idx + self.window_size], dtype=torch.float32))


class RecurentNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RecurentNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
    

def train(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            # print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')


def test(model, dataloader):
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            output = model(x)
            mse_error = torch.nn.functional.mse_loss(output, y.unsqueeze(1))
            print(f'MSE Error: {mse_error.item():.4f}')


def test_denormalize(model, dataloader, max_value, min_value):
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            output = model(x)
            denormalized_output = output * (max_value - min_value) + min_value
            denormalized_y = y * (max_value - min_value) + min_value

            mse_error = torch.nn.functional.mse_loss(denormalized_output, denormalized_y.unsqueeze(1))
            print(f'MSE Error (denormalized): {mse_error.item():.4f}')


if __name__ == '__main__':
    PREDICTED_COLUMN = 'pres'
    df_raw = pd.read_csv('./data/raw_data.csv')
    min_value = df_raw[PREDICTED_COLUMN].min()
    max_value = df_raw[PREDICTED_COLUMN].max()

    df = pd.read_csv('./data/normalized_data.csv')

    model = RecurentNeuralNetwork(input_size=18, hidden_size=32, num_layers=2, output_size=1)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = Dataset(df, PREDICTED_COLUMN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train(model, dataloader, criterion, optimizer, epochs=1)

    test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    test(model, test_dataloader)

    test_denormalize(model, test_dataloader, max_value, min_value)
    
