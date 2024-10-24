import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.error_code.error_code import BitFlipToricCode
import numpy as np
from .qecdata import QECDataset


class BitFlipToricData(QECDataset):
    def __init__(self, distance: int, noises, name: str, load: bool, random_flip: bool, device, sequential: bool = False,
                 cluster: bool = False, only_syndromes : bool = False, supervised: bool = False,):
        super().__init__(distance=distance, noises=noises, name=name, load=load, device=device, random_flip=random_flip,
                         cluster=cluster, only_syndromes=only_syndromes, supervised=supervised)
        self.sequential = sequential

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        if self.supervised:
            return self.syndromes[index], self.labels[index]
        output = (self.syndromes[index],)
        if not self.only_syndromes:
            output = output + (self.logical[index],)
        if not self.train:
            output = output + (self.flips[index],)
        return output

    def generate_data(self, n):
        syndromes = []  # deleted rounds, TODO check if shuffling is really doing the proper thing here
        flips = []
        labels = []
        logical = []
        for noise in tqdm(self.noises):
            code = BitFlipToricCode(self.distance, noise, self.random_flip)
            if not self.train and self.random_flip and not self.supervised:
                syndromes_noise, flips_noise = code.get_syndromes(n, self.train, self.supervised)
                flips = flips + flips_noise
            else:
                syndromes_noise = code.get_syndromes(n, self.train, self.supervised)
            syndromes = syndromes + list(
                map(lambda x: x[:int(0.5 * len(x))], syndromes_noise))
            if self.supervised:
                labels = labels + [[1, 0] if noise < 0.109 else [0, 1]]*len(syndromes_noise)

        # Bring data into right shape
        # In torch, batch has indices (N,C,H,W)
        if self.sequential:
            syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), self.distance ** 2, 2))
        else:
            syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), 1, self.distance, self.distance))

        output = (torch.as_tensor(np.array(syndromes), device=self.device, dtype=torch.float32),)

        if not self.only_syndromes:
            output = output + (torch.as_tensor(np.array(logical), device=self.device, dtype=torch.float32),)
        if self.random_flip:
            output = output + (torch.as_tensor(np.array(flips), device=self.device, dtype=torch.float32),)

        return output

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val

