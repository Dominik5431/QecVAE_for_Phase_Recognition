import pickle
from pathlib import Path


class ResultsWrapper:
    """
    Wrapper class that manages the results of the latent space evaluation or the reconstruction error evaluation.
    Stores the results in self.dict
    Implements method to add data, get data, and save and load the results.
    """
    def __init__(self, name: str, cluster: bool = False):
        self.dict = {}
        self.name = name
        self.cluster = cluster

    def add(self, distance, latent):
        self.dict[distance] = latent

    def get(self, distance):
        return self.dict[distance]

    def save(self):
        if not self.cluster:
            with open(str(Path().resolve().parent) + "/data/" + self.name + ".pkl", 'ab') as fp:
                pickle.dump(self.dict, fp)
        else:
            with open("data/" + self.name + ".pkl", 'ab') as fp:
                pickle.dump(self.dict, fp)

    def load(self):
        try:
            # Cluster requires main script at the root --> hence different folder architecture
            if not self.cluster:
                with open(str(Path().resolve().parent) + "/data/" + self.name + ".pkl", 'rb+') as fp:
                    try:
                        self.dict = pickle.load(fp)
                    except EOFError:
                        self.dict = {}
            else:
                with open("data/" + self.name + ".pkl", 'rb+') as fp:
                    try:
                        self.dict = pickle.load(fp)
                    except EOFError:
                        self.dict = {}
        except FileNotFoundError:
            self.dict = {}
        return self

    def get_dict(self):
        return self.dict
