from data_frames import FrameLoader
from prophet import Prophet
import matplotlib.pyplot as plt

df = FrameLoader(FrameLoader.RAW).load()

def predict(df, column, starttime=0, trainperiod = 168, periods=48):
    traindata = df.iloc[starttime:starttime + trainperiod]
    testdata = df.iloc[starttime + trainperiod:starttime + trainperiod + periods]
    model = Prophet()
    model.fit(traindata[['Timestamp', column]].rename(columns={'Timestamp': 'ds', column: 'y'}).dropna())

    future = model.make_future_dataframe(periods=periods, freq='H', include_history=True)
    forecast = model.predict(future)

    return model, forecast, testdata

def all_predictions(df, starttime=0):
    models = []
    for column in df.columns:
        if column == 'Timestamp':
            continue
        models.append(predict(df, column, starttime))
    return models

def plot_all_predictions(df, models):
    for i, (model, forecast, testdata) in enumerate(models):
        column = df.columns[i + 1]
        plt.figure(figsize=(10, 6))
        model.plot(forecast, xlabel='Data', ylabel=column)
        plt.plot(testdata['Timestamp'], testdata[column], 'g.', label='Expected values')
        plt.legend()
        plt.title(f'Prognoza pentru parametrul {column}')
        plt.show()

plot_all_predictions(df, all_predictions(df, 100))
