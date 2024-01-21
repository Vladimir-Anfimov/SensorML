import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_frames import FrameLoader

def load():
    dataframe_loader = FrameLoader(FrameLoader.RAW)
    df = dataframe_loader.load()
    return df

def normalize(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour

    MAX_VALUES = {'month': 12, 'day': 31, 'hour': 23}
    
    for component in MAX_VALUES.keys():
        max_value = MAX_VALUES[component]
        df[f'sin_{component}'] = np.sin(2 * np.pi * df[component]/max_value)
        df[f'cos_{component}'] = np.cos(2 * np.pi * df[component]/max_value)

        df[f'sin_{component}'] = (df[f'sin_{component}'] + 1) / 2
        df[f'cos_{component}'] = (df[f'cos_{component}'] + 1) / 2

    df = df.drop(['Timestamp', 'month', 'day', 'hour'], axis=1)

    scaler = MinMaxScaler()

    columns_to_normalize=['pres','temp1','umid','temp2','V450','B500','G550','Y570','O600','R650','temps1','temps2','lumina']

    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # df.to_csv('./data/normalized_data.csv', index=False)

    return df



if __name__ == '__main__':
    input_verification = input('Are you sure you want to normalize the data? (y/n): ')
    if input_verification != 'y':
        exit(0)
    df = load()
    normalize(df)