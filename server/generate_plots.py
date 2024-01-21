from predictors import *
import pandas as pd
from data_frames import FrameLoader
import matplotlib.pyplot as plt
from lstm_train import RecurentNeuralNetwork
from seq2seq_train import Seq2SeqNeuralNetwork
import os

class PlotsGenerator:
    def __init__(self, dataframe_loader) -> None:
        self.dataframe_loader = dataframe_loader
        self.df = dataframe_loader.load()[:336]

    def lstm_plots(self):
        figs: list[plt.Figure] = []
        predictor = LstmPredictorNormalized()
        predicted = predictor.predict(self.df, 128)

        save_dir = 'images/lstm/'
        os.makedirs(save_dir, exist_ok=True)

        for column in predicted.columns[1:]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.df['Timestamp'], self.df[column].astype('float32'), '#181818', label='Your values')
            plt.plot(predicted['Timestamp'], predicted[column].astype('float32'), '#0072B2', label='Predicted values')
            plt.title(f'Forecast for the parameter \'{column}\' using LSTM')
            plt.legend()
            figs.append(plt.gcf)
            # plt.savefig(f'images/lstm/lstm_{column}.png')
            plt.close()
        
        return figs

    def seq2seq_plots(self):
        figs: list[plt.Figure] = []
        predictor = Seq2SeqPredictorNormalized()
        predicted = predictor.predict(self.df, 128)

        save_dir = 'images/seq2seq/'
        os.makedirs(save_dir, exist_ok=True)

        for column in predicted.columns[1:]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.df['Timestamp'], self.df[column].astype('float32'), '#181818', label='Your values')
            plt.plot(predicted['Timestamp'], predicted[column].astype('float32'), '#0072B2', label='Predicted values')
            plt.title(f'Forecast for the parameter \'{column}\' using Seq2Seq')
            plt.legend()
            # plt.savefig(f'images/seq2seq/seq2seq_{column}.png')
            figs.append(plt.gcf)
            plt.close()
        
        return figs


if __name__ == '__main__':
    PlotsGenerator(FrameLoader(FrameLoader.RAW)).seq2seq_plots()
    PlotsGenerator(FrameLoader(FrameLoader.RAW)).lstm_plots()

