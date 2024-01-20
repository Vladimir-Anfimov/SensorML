import math
import pandas as pd
import torch
from data_frames import FrameLoader
from rnn_train import RecurentNeuralNetwork

model_path = "./models/lstm/pres_OS30_WS7.pth"

model = RecurentNeuralNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()

PREDICTED_COLUMN = 'pres'

dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
df = dataframe_loader.load()
df.drop(columns=[PREDICTED_COLUMN], inplace=True)
data = df.head(7).values

total_days = 40
predictions = torch.tensor(data, dtype=torch.float32)

for i in range((total_days//model.output_size)):
    new_prediction = model(predictions[-7:].unsqueeze(0))
    print(predictions.shape, new_prediction.shape)
    exit()
    predictions = torch.cat((predictions, new_prediction))

print(predictions, predictions.shape)