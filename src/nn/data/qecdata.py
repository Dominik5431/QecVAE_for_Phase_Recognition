import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import logging
from pathlib import Path


class QECDataset(Dataset, ABC):
    def __init__(self, distance: int, noises, name: str, num: int, load: bool, random_flip: bool):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distance = distance
        self.noises = noises
        self.name = name
        self.random_flip = random_flip
        if load:
            self.syndromes = None
            try:
                self.load()
            except NameError:
                logging.error("No valid noise model specified.")
        else:
            self.syndromes = self.generate_data(num, 10)

        if not (type(name) is str):
            raise ValueError

    def save(self):
        torch.save(self.syndromes, str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name))
        # torch.save(self.syndromes, "data/syndromes_{0}.pt".format(self.name))

    def load(self):
        self.syndromes = torch.load(str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name), mmap=True)  # TODO check if mmap reduces memory usage
        # self.syndromes = torch.load("data/syndromes_{0}.pt".format(self.name), mmap=True)

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n, rounds):
        raise NotImplementedError
