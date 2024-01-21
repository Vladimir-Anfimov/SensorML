import pandas as pd


class FrameLoader:
    NORMALIZED = './data/normalized_data.csv'
    RAW = './data/raw_data.csv'
    FEATURED_SELECTED_NORMALIZED = './data/featured_selected_normalized_data.csv'  
    
    def __init__(self, type) -> None:
        if type != FrameLoader.NORMALIZED and type != FrameLoader.RAW and type != FrameLoader.FEATURED_SELECTED_NORMALIZED:
            raise Exception('Invalid type')

        self.type = type


    def load(self):
        return pd.read_csv(self.type) 