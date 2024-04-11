import numpy as np
import pickle


class Predictions:
    def __init__(self):
        self.dict = {}
        # self.path = path

    def add(self, distance, predictions):
        self.dict["{0}".format(distance)] = predictions

    def get(self, distance):
        return self.dict["{0}".format(distance)]

    def save(self):
        with open('files/prediction_depolarizing_0.pkl', 'wb') as fp:
            pickle.dump(self.dict, fp)

    def load(self):
        #with open('files/prediction_depolarizing_0.pkl', 'rb') as fp:
        with open('files/prediction_depolarizing_0.pkl', 'rb') as fp:
            self.dict = pickle.load(fp)
        return self

    def get_dict(self):
        return self.dict

