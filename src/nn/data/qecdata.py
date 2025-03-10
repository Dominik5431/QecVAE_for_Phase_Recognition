import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import logging
from pathlib import Path


class QECDataset(Dataset, ABC):
    """
        Custom Dataset for QEC data.
        Upon initialization, the data samples are either generated or loaded.
    """
    def __init__(self, distance: int, noises, name: str, load: bool, device: torch.device, random_flip: bool = False, cluster: bool = False, only_syndromes: bool = False):
        super().__init__()
        self.train = True
        self.device = device
        self.distance = distance
        self.noises = noises
        self.name = name
        self.random_flip = random_flip
        self.syndromes = None
        self.logical = None
        self.flips = None
        self.labels = None
        self.load_data = load
        self.cluster = cluster
        self.only_syndromes = only_syndromes
        if not (type(name) is str):
            raise ValueError

    def initialize(self, num: int):
        if self.load_data:
            try:
                self.load()
            except NameError:
                logging.error("No valid noise model specified.")
        else:
            # Depending on which setting is chosen, syndromes, flips and/or logicals are needed.
            if self.random_flip and self.only_syndromes:
                self.syndromes, self.flips = self.generate_data(num)
            elif self.random_flip and not self.only_syndromes:
                self.syndromes, self.logical, self.flips = self.generate_data(num)
            elif not self.random_flip and self.only_syndromes:
                self.syndromes = self.generate_data(num)
            else:
                self.syndromes = self.generate_data(num)
        return self

    def training(self):
        """
        Sets training mode
        :return: self
        """
        self.train = True
        return self

    def eval(self):
        """
        Sets eval mode
        :return: self
        """
        self.train = False
        return self

    def save(self):
        pre = ""
        if not self.cluster:
            pre += str(Path().resolve().parent) + "/"

        torch.save(self.syndromes, pre + "data/syndromes_{0}.pt".format(self.name))
        if not self.train and self.random_flip:
            torch.save(self.flips, str(Path().resolve().parent) + "data/flips_{0}.pt".format(self.name))
        if not self.only_syndromes:
            torch.save(self.logical, pre + "data/logical_{0}.pt".format(self.name))

    def load(self):
        pre = ""
        if not self.cluster:
            pre += str(Path().resolve().parent) + "/"

        self.syndromes = torch.load(pre + "data/syndromes_{0}.pt".format(self.name), mmap=True, map_location=self.device)
        if not self.train and self.random_flip:
            self.flips = torch.load(pre + "data/flips_{0}.pt".format(self.name), mmap=True, map_location=self.device)
        if not self.only_syndromes:
            self.logical = torch.load(pre + "data/logical_{0}.pt".format(self.name), mmap=True, map_location=self.device)

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n):
        """
            Method that generates the syndromes. Needs to be implemented by each class that inherits from this class.
            Detailed implementation depends upon the specific qec code and noise model.
            :param n: Number of samples to generate.
            :return:
        """
        raise NotImplementedError
