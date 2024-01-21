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


class RecurentNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size=19, hidden_size=32, num_layers=2, output_size=OUTPUT_SIZE):
        super(RecurentNeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
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


# def test_denormalize(model, dataloader, criterion, max_value, min_value):
#     with torch.no_grad():
#         for i, (x, y) in enumerate(dataloader):
#             output = model(x)
#             denormalized_output = output * (max_value - min_value) + min_value
#             denormalized_y = y * (max_value - min_value) + min_value

#             error = criterion(denormalized_output, denormalized_y)
#             print(f'Error (denormalized): {error.item():.4f}')




def start(PREDICTED_COLUMN):
    # df_raw = pd.read_csv('./data/raw_data.csv')
    # min_value = df_raw[PREDICTED_COLUMN].min()
    # max_value = df_raw[PREDICTED_COLUMN].max()

    dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
    df = dataframe_loader.load()
    
    model = RecurentNeuralNetwork().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = Dataset(df, PREDICTED_COLUMN)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    train(model, dataloader, criterion, optimizer, epochs=5)

    torch.save(model.state_dict(), f'./models/lstm/{PREDICTED_COLUMN}-OS-{OUTPUT_SIZE}-WS-{WINDOW_SIZE}.pth')

    test_dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    test(model, test_dataloader, criterion)

    # test_denormalize(model, test_dataloader, criterion, max_value, min_value)
    

    
if __name__ == '__main__':
    input_answer = input('Are you sure you want to train the RNN LSTM model? (y/n) ')

    if input_answer.lower() == 'y':
        print('Ok, let\'s go!')
        for column in NORMALIZED_PARAMS_NAMES:
            print(f'Training {column}...')
            start(column)
    else:
        print('Ok, bye!')