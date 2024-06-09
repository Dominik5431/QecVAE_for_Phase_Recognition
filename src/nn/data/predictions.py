import pickle
from pathlib import Path


class Predictions:
    def __init__(self, name):
        self.dict = {}
        self.name = name

    def add(self, distance, latent):
        self.dict[distance] = latent

    def get(self, distance):
        return self.dict[distance]

    def save(self):
        with open(str(Path().resolve().parent) + "/data/" + self.name + ".pkl", 'ab') as fp:
            pickle.dump(self.dict, fp)
        # with open("data/" + self.name + ".pkl", 'ab') as fp:
        #    pickle.dump(self.dict, fp)

    def load(self):
        try:
            with open(str(Path().resolve().parent) + "/data/" + self.name + ".pkl", 'rb+') as fp:
                try:
                    self.dict = pickle.load(fp)
                except EOFError:
                    self.dict = {}
            # with open("data/" + self.name + ".pkl", 'rb+') as fp:
            #    try:
            #        self.dict = pickle.load(fp)
            #    except EOFError:
            #        self.dict = {}
        except FileNotFoundError:
            self.dict = {}
        return self

    def get_dict(self):
        return self.dict
