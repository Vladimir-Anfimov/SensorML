import math
import pandas as pd
import torch
from data_frames import FrameLoader
from multivar_rnn_interate import MultiVarRNNIterate


PREDICTED_COLUMN = 'pres'

dataframe_loader = FrameLoader(FrameLoader.NORMALIZED)
df = dataframe_loader.load()
data = df.head(7).values

# total_days = 40
predictions = torch.tensor(data, dtype=torch.float32)

multiVarRNN = MultiVarRNNIterate(MultiVarRNNIterate.ALL)
new_prediction = multiVarRNN.get_prediction_vector(predictions.unsqueeze(0))
print(new_prediction.shape)

print()

print(new_prediction)


# for i in range(math.ceil(total_days//30)):
#     print(predictions.shape, new_prediction.shape)
#     predictions = torch.cat((predictions, new_prediction))

# print(predictions, predictions.shape)