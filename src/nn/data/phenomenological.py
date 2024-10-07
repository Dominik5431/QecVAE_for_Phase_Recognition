import torch
from torch.utils.data import Dataset
from src.error_code.error_code import ToricCodePheno
import numpy as np
from .qecdata import QECDataset


class PhenomenologicalSurfaceData(QECDataset):
    def __init__(self, distance, noises, name, num, load, random_flip):
        super().__init__(distance=distance, noises=noises, name=name, num=num, load=load, random_flip=random_flip)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        return self.syndromes[index]

    def generate_data(self, n, rounds):
        syndromes = []  # deleted rounds, TODO check if shuffling is really doing the proper thing here
        for noise in self.noises:
            code = SurfaceCode(self.distance, noise, self.random_flip)
            syndromes = syndromes + list(
                map(lambda x: x[:int(0.5 * len(x))], np.array(code.get_syndromes(n))))
        # Bring data into right shape
        # In torch, batch has indices (N,C,H,W)
        syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), 2, self.distance, self.distance))
        return torch.as_tensor(syndromes, device=self.device, dtype=torch.double)

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val

