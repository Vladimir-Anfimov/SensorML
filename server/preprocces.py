import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load():
    df = pd.read_csv('./data/raw_data.csv')
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

    print(df.head())

    df.to_csv('./data/normalized_data.csv', index=False)



if __name__ == '__main__':
    df = load()
    normalize(df)