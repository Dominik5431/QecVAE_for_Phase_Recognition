import torch
from torch.utils.data import Dataset
from src.error_code.error_code import RotatedSurfaceCode, BitFlipSurfaceCode
import numpy as np
from .qecdata import QECDataset


class BitFlipSurfaceData(QECDataset):
    def __init__(self, distance: int, noises, name: str, num: int, load: bool, random_flip: bool):
        super().__init__(distance, noises, name, num, load, random_flip)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        return self.syndromes[index]

    def generate_data(self, n, rounds):
        syndromes = []  # deleted rounds, TODO check if shuffling is really doing the proper thing here
        for noise in self.noises:
            code = BitFlipSurfaceCode(self.distance, noise, self.random_flip)
            syndromes = syndromes + list(
                map(lambda x: x[:int(0.5 * len(x))], np.array(code.get_syndromes(n))))
        # Bring data into right shape
        # In torch, batch has indices (N,C,H,W)
        syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), 1, self.distance, self.distance))
        return torch.as_tensor(syndromes, device=self.device, dtype=torch.double)

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val


class BitFlipRotatedSurfaceData(QECDataset):
    def __init__(self, distance, noises, name, num, load: bool):
        super().__init__(distance, noises, name, num, load)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        return self.syndromes[index]

    def generate_data(self, num, rounds):
        def ising_shape(arr):
            result = np.zeros((1, self.distance - 1, self.distance))
            shift = -1
            for k in np.arange(self.distance + 1):
                if k % 2 == 0:
                    shift += 1
                for j in np.arange(int((self.distance - 1) / 2)):
                    result[0, j + shift, -int((self.distance - 1) / 2) + j - k + shift] = arr[
                        k * int((self.distance - 1) / 2) + j]
            return result

        # Important: generate data alternating between noise, otherwise data will be ordered as labels = [0, ..., 0, 1, ..., 1]
        # Shuffling with limited buffer_size then will not mix 0 and 1 labels, shuffling with buffer over whole data set negates the effect of memory saving since the memory occupancy will be too large for large datasets
        syndromes = []
        for _ in np.arange(rounds):
            for noise in self.noises:
                code = RotatedSurfaceCode(self.distance, noise)
                syndromes = syndromes + list(
                    map(lambda x: x[:int(0.5 * len(x))], np.array(code.get_syndromes(int(num / rounds)))))
        # Bring data into right shape
        # In torch, batch has indices (N,C,H,W)
        syndromes = np.array(list(map(ising_shape, np.array(syndromes))))
        return torch.as_tensor(syndromes, device=self.device)

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val
