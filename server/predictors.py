import math
import torch
from lstm_train import RecurentNeuralNetwork
from seq2seq_train import Seq2SeqNeuralNetwork



class Predictor:
    def __init__(self, window_size, output_size):
        self.window_size = window_size
        self.output_size = output_size

    def predict(self, df, total_days=30):
        predicted = torch.tensor(df.values, dtype=torch.float32).view(1, -1, len(df.columns))

        for i in range(math.ceil(total_days / self.output_size)):
            predicted = torch.cat((predicted, self.model.forward(predicted[:, -self.window_size:, :])), dim=1)

        return predicted[:, -total_days:, :].squeeze()




class LstmPredictorNormalized(Predictor):
    LSTM_NORMALIZED_MODEL_PATH = './models/lstm/lstm_normalized.pth'

    def __init__(self, window_size=7, output_size=30):
        super().__init__(window_size, output_size)

        state_dict = torch.load(self.LSTM_NORMALIZED_MODEL_PATH)
        self.model = RecurentNeuralNetwork()
        self.model.load_state_dict(state_dict)
    


class Seq2SeqPredictorNormalized(Predictor):
    SEQ_2_SEQ_NORMALIZED_MODEL_PATH = './models/seq2seq/seq2seq_normalized.pth'

    def __init__(self, window_size=7, output_size=30):
        super().__init__(window_size, output_size)

        state_dict = torch.load(self.SEQ_2_SEQ_NORMALIZED_MODEL_PATH)
        self.model = Seq2SeqNeuralNetwork()
        self.model.load_state_dict(state_dict)



if __name__ == '__main__':
    from data_frames import FrameLoader

    df = FrameLoader(FrameLoader.NORMALIZED).load()
    df = df[:100]
    predictor = Seq2SeqPredictorNormalized()
    print(predictor.predict(df, 100).shape)