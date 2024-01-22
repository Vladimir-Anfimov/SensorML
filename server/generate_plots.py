from predictors import *
import pandas as pd
from data_frames import FrameLoader
import matplotlib.pyplot as plt
from lstm_train import RecurentNeuralNetwork
from seq2seq_train import Seq2SeqNeuralNetwork
import os


friendly_names = {
    "pres": "Pressure",
    "temp1": "Temperature in point A",
    "umid": "Humidity",
    "temp2": "Temperature in point B",
    "V450": "Voltage at 450nm",
    "B500": "Blue at 500nm",
    "G550": "Green at 550nm",
    "Y570": "Yellow at 570nm",
    "O600": "Orange at 600nm",
    "R650": "Red at 650nm",
    "temps1": "Soil Temperature in point A",
    "temps2": "Soil Temperature in point B",
    "lumina": "Light",
}


class PlotsGenerator:
    def __init__(self, user_df) -> None:
        self.df = user_df

    def lstm_plots(self):
        figs: list[plt.Figure] = []
        predictor = LstmPredictorNormalized()
        predicted = predictor.predict(self.df, 672)

        save_dir = "images/lstm/"
        os.makedirs(save_dir, exist_ok=True)

        for column in predicted.columns[1:]:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.df.iloc[8 * 7 * 24 :]["Timestamp"],
                self.df.iloc[8 * 7 * 24 :][column].astype("float32"),
                "#181818",
                label="Your values",
            )
            plt.plot(
                predicted["Timestamp"],
                predicted[column].astype("float32"),
                "#0072B2",
                label="Predicted values",
            )
            plt.title(friendly_names[column])
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel(friendly_names[column])
            figs.append(plt.gcf())
            plt.close()

        return figs, predicted

    def seq2seq_plots(self):
        figs: list[plt.Figure] = []
        predictor = Seq2SeqPredictorNormalized()
        predicted = predictor.predict(self.df, 672)

        save_dir = "images/seq2seq/"
        os.makedirs(save_dir, exist_ok=True)

        for column in predicted.columns[1:]:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.df.iloc[8 * 7 * 24 :]["Timestamp"],
                self.df.iloc[8 * 7 * 24 :][column].astype("float32"),
                "#181818",
                label="Your values",
            )
            plt.plot(
                predicted["Timestamp"],
                predicted[column].astype("float32"),
                "#0072B2",
                label="Predicted values",
            )
            plt.title(friendly_names[column])
            plt.legend()
            plt.xlabel("Date")
            plt.ylabel(friendly_names[column])
            figs.append(plt.gcf())
            plt.close()

        return figs, predicted


if __name__ == "__main__":
    dataframe_loader = FrameLoader(FrameLoader.RAW)
    df = dataframe_loader.load()[:336]

    PlotsGenerator(df).seq2seq_plots()
    PlotsGenerator(df).lstm_plots()
