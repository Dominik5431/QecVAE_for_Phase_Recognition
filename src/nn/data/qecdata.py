import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import logging
from pathlib import Path


class QECDataset(Dataset, ABC):
    def __init__(self, distance: int, noises, name: str, load: bool, random_flip: bool, supervised: bool = False, cluster: bool = False):
        super().__init__()
        self.train = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distance = distance
        self.noises = noises
        self.name = name
        self.random_flip = random_flip
        self.syndromes = None
        self.flips = None
        self.labels = None
        self.load_data = load
        self.cluster = cluster
        self.supervised = supervised
        if not (type(name) is str):
            raise ValueError

    def initialize(self, num: int):
        if self.load_data:
            try:
                self.load()
            except NameError:
                logging.error("No valid noise model specified.")
        else:
            if self.supervised:
                self.syndromes, self.labels = self.generate_data(num, 10)  # Attention: in supervised mode no flips necessary!
            elif self.random_flip:
                self.syndromes, self.flips = self.generate_data(num, 10)
            else:
                self.syndromes = self.generate_data(num, 10)
        return self

    def training(self):
        if self.supervised:
            # print("Training modus not available in supervised mode.")
            return self
        self.train = True
        return self

    def eval(self):
        if self.supervised:
            # print("Training modus not available in supervised mode.")
            return self
        self.train = False
        return self

    def save(self):
        if not self.cluster:
            torch.save(self.syndromes, str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name))
            if self.supervised:
                torch.save(self.labels, str(Path().resolve().parent) + "/data/labels_{0}.pt".format(self.name))
            elif not self.train and self.random_flip:
                torch.save(self.flips, str(Path().resolve().parent) + "/data/flips_{0}.pt".format(self.name))
        else:
            torch.save(self.syndromes, "data/syndromes_{0}.pt".format(self.name))
            if self.supervised:
                torch.save(self.labels, "data/labels_{0}.pt".format(self.name))
            elif not self.train and self.random_flip:
                torch.save(self.flips, "data/flips_{0}.pt".format(self.name))

    def load(self):
        if not self.cluster:
            self.syndromes = torch.load(str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name), mmap=True, map_location=torch.device('cpu'))  # TODO check if mmap reduces memory usage
            if self.supervised:
                self.labels = torch.load(str(Path().resolve().parent) + "/data/labels_{0}.pt".format(self.name), mmap=True, map_location=torch.device('cpu'))
            elif not self.train and self.random_flip:
                self.flips = torch.load(str(Path().resolve().parent) + "/data/flips_{0}.pt".format(self.name), mmap=True, map_location=torch.device('cpu'))
        else:
            self.syndromes = torch.load("data/syndromes_{0}.pt".format(self.name), mmap=True)
            if self.supervised:
                self.labels = torch.load("data/labels_{0}.pt".format(self.name), mmap=True)
            elif not self.train and self.random_flip:
                self.syndromes = torch.load("data/flips_{0}.pt".format(self.name), mmap=True)

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n, rounds):
        raise NotImplementedError
