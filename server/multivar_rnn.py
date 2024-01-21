import torch
from rnn_train import RecurentNeuralNetwork
from params import NORMALIZED_PARAMS_NAMES


class MultiVarRNN:
    ALL = 'all'
    FEATURE_SELECTED = 'feature_selected'

    def __init__(self, mode=ALL):
        self.mode = mode
        self.load_atributes()

    
    def load_atributes(self):
        if self.mode == self.ALL:
            self.params_names = NORMALIZED_PARAMS_NAMES
        elif self.mode == self.FEATURE_SELECTED:
            raise Exception('Not implemented yet')
        else:
            raise Exception('Invalid mode')
        
    def predict_atribute(self, atribute, input_tensor):
        if atribute not in self.params_names:
            raise Exception('Invalid atribute')
        
        model = RecurentNeuralNetwork()
        model.load_state_dict(torch.load(f'./models/lstm/{atribute}-OS-30-WS-7.pth'))
        model.eval()

        return model(input_tensor)
    

    def get_prediction_vector(self, input_tensor):
        prediction_vector = torch.tensor([], dtype=torch.float32)
        for atribute in self.params_names:
            new_prediction = self.predict_atribute(atribute, input_tensor)
            prediction_vector = torch.cat((prediction_vector, new_prediction))
        return prediction_vector
