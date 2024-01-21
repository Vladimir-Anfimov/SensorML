from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_frames import FrameLoader


class ProphetPredictor:
    def __init__(self, df) -> None:
        self.df = df

    def train(self):
        models = []
        for column in df.columns:
            if column == "Timestamp":
                continue
            models.append(self.train_column(column))
        return models

    def train_column(self, column):
        model = Prophet()
        model.fit(
            self.df[["Timestamp", column]]
            .rename(columns={"Timestamp": "ds", column: "y"})
            .dropna()
        )

        return model

    def warm_start_params(self, model: Prophet):
        # https://facebook.github.io/prophet/docs/additional_topics.html#updating-fitted-models
        res = {}
        for pname in ["k", "m", "sigma_obs"]:
            if model.mcmc_samples == 0:
                res[pname] = model.params[pname][0][0]
            else:
                res[pname] = np.mean(model.params[pname])
        for pname in ["delta", "beta"]:
            if model.mcmc_samples == 0:
                res[pname] = model.params[pname][0]
            else:
                res[pname] = np.mean(model.params[pname], axis=0)
        return res

    def predict(
        self,
        df,
        trained_models,
    ) -> list[tuple[Prophet, pd.DataFrame]]:
        models = []
        for i, model in enumerate(trained_models):
            column = df.columns[i + 1]
            models.append(
                self.predict_column(
                    df,
                    column,
                    model,
                )
            )
        return models

    def predict_column(self, new_df, column, trained_model, periods=4 * 7 * 24):
        model = Prophet()
        model.fit(
            new_df[["Timestamp", column]]
            .rename(columns={"Timestamp": "ds", column: "y"})
            .dropna(),
            init=self.warm_start_params(trained_model),
        )

        future = model.make_future_dataframe(
            periods=periods, freq="H", include_history=False
        )
        forecast = model.predict(future)

        return model, forecast

    def generate_plots(self, df, models):
        figs: list[plt.Figure] = []
        for i, (model, forecast) in enumerate(models):
            column = df.columns[i + 1]
            plt.figure(figsize=(10, 6))
            plt.plot(
                df.iloc[8 * 7 * 24 :]["Timestamp"],
                df.iloc[8 * 7 * 24 :][column],
                "#181818",
                label="Your values",
            )
            plt.plot(
                forecast["ds"],
                forecast["yhat"],
                "#0072B2",
                label="Predicted values",
            )
            plt.legend()
            plt.title(f"Prognoza pentru parametrul {column}")
            figs.append(plt.gcf())

        # save figures
        for i, fig in enumerate(figs):
            fig.savefig(f"./images/prophet_{i}.png", bbox_inches="tight")
        return figs

    def save(self, df, models):
        for i, (model) in enumerate(models):
            with open(f"./models/prophet/{df.columns[i + 1]}.json", "w") as f:
                f.write(model_to_json(model))

    def load(self, df):
        models = []
        for i, column in enumerate(df.columns):
            if column == "Timestamp":
                continue
            with open(f"./models/prophet/{column}.json", "r") as f:
                models.append(model_from_json(f.read()))
        return models

    @staticmethod
    def get_plots(uploaded_df):
        df = FrameLoader(FrameLoader.RAW).load()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        predictor = ProphetPredictor(df)
        models = predictor.load(train_df)
        user_models = predictor.predict(uploaded_df, models)
        return predictor.generate_plots(test_df, user_models)


if __name__ == "__main__":
    df = FrameLoader(FrameLoader.RAW).load()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    cuttoff = int(len(df) * 0.8)
    train_df = df.iloc[:cuttoff]
    test_df = df.iloc[cuttoff:]
    # predictor = ProphetPredictor(train_df)
    # # models = predictor.train()
    # models = predictor.load(train_df)
    # # predictor.save(train_df, models)
    # user_models = predictor.predict(test_df, models)
    # predictor.generate_plots(test_df, user_models)

    plots = ProphetPredictor.get_plots(test_df)
    