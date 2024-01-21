import math
import pandas as pd
import torch
from lstm_train import RecurentNeuralNetwork
from seq2seq_train import Seq2SeqNeuralNetwork
from params import OUTPUT_SIZE, RAW_PARAMS_NAMES, WINDOW_SIZE
from data_frames import FrameLoader
from preproccessing import normalize


class Predictor:
    def __init__(self, window_size, output_size,):
        self.window_size = window_size
        self.output_size = output_size

    def predict(self, df, total_records):
        start_timestamp = pd.to_datetime(df['Timestamp'][len(df) - 1]) + pd.Timedelta(hours=1)
        df = normalize(df)
        predicted = self.model(torch.tensor(df.values, dtype=torch.float32).view(1, -1, len(df.columns)))
        
        print(f'Total executions: {math.ceil(total_records / self.output_size)}')
        
        for i in range(math.ceil(total_records / self.output_size) - 1):
            predicted = torch.cat((predicted, self.model(predicted[:, -self.window_size:, :])), dim=1)

        predicted = predicted.squeeze(0)
        predicted = predicted.detach().numpy()
        predicted = pd.DataFrame(predicted, columns=df.columns)
        predicted = predicted[:total_records]

        return self.denormalize(predicted, start_timestamp)


    def denormalize(self, df, start_timestamp):
        raw_df = FrameLoader(FrameLoader.RAW).load()

        for column in RAW_PARAMS_NAMES: 
            if column == 'Timestamp':
                continue
            df[column] = df[column] * (raw_df[column].max() - raw_df[column].min()) + raw_df[column].min()

        # insereaza la inceputul dataframe-ului o coloana cu timestamp-uri
        df.insert(0, 'Timestamp', 0)
        df['Timestamp'] = pd.date_range(start=start_timestamp, periods=len(df), freq='H')
        
        df = df.drop(['sin_month', 'cos_month', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour'], axis=1)


        return df


class LstmPredictorNormalized(Predictor):
    LSTM_NORMALIZED_MODEL_PATH = './models/lstm/lstm_normalized.pth'

    def __init__(self, window_size=WINDOW_SIZE, output_size=OUTPUT_SIZE):
        super().__init__(window_size, output_size)

        state_dict = torch.load(self.LSTM_NORMALIZED_MODEL_PATH)
        self.model = RecurentNeuralNetwork()
        self.model.load_state_dict(state_dict)


class Seq2SeqPredictorNormalized(Predictor):
    SEQ_2_SEQ_NORMALIZED_MODEL_PATH = './models/seq2seq/seq2seq_normalized.pth'

    def __init__(self, window_size=WINDOW_SIZE, output_size=OUTPUT_SIZE):
        super().__init__(window_size, output_size)

        state_dict = torch.load(self.SEQ_2_SEQ_NORMALIZED_MODEL_PATH)
        self.model = Seq2SeqNeuralNetwork()
        self.model.load_state_dict(state_dict)



if __name__ == '__main__':
    dataframe_loader = FrameLoader(FrameLoader.RAW)
    df = dataframe_loader.load()
    df = df[:30]
    predictor = Seq2SeqPredictorNormalized()
    predicted = predictor.predict(df, 2)
    print(predicted)
