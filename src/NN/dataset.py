import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from abc import ABC, abstractmethod
from src.ErrorCode.error_code import BitFlipSurface, DepolarizingSurface
import numpy as np
import logging
from pathlib import Path


class QECDataset(Dataset, ABC):
    def __init__(self, distance: int, noises, name: str, num: int, load: bool):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distance = distance
        self.noises = noises
        self.name = name
        if load:
            self.syndromes = None
            try:
                self.load()
            except NameError as N:
                logging.error("No valid noise model specified.")
        else:
            self.syndromes = self.generate_data(num, 10)
        if not (type(name) is str):
            raise ValueError

    def save(self):
        torch.save(self.syndromes, str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name))

    def load(self):
        self.syndromes = torch.load(str(Path().resolve().parent) + "/data/syndromes_{0}.pt".format(self.name))

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n, rounds):
        raise NotImplementedError


class BitFlipSurfaceData(QECDataset):
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
                code = BitFlipSurface(self.distance, noise)
                syndromes = syndromes + list(code.get_syndromes(int(num / rounds)))
        syndromes = list(map(lambda x: np.where(x, -1, 1), syndromes))
        # Bring data into right shape
        syndromes_new = np.zeros((len(syndromes), 1, self.distance - 1, self.distance))  # In torch, batch has indices (N,C,H,W)
        for i, elm in enumerate(syndromes):
            syndromes_new[i] = ising_shape(elm)
        return torch.as_tensor(syndromes_new, device=self.device)

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self, [ratio - 1/len(self), 1 - ratio + 1/len(self)])
        return dataset_train, dataset_val


class DepolarizingSurfaceData(QECDataset):
    def __init__(self, distance, noises, name, num):
        super().__init__(distance, noises, name, num)

    def generate_data(self, num, rounds):
        # Important: generate data alternating between noise, otherwise data will be ordered as labels = [0, ..., 0,
        # 1, ..., 1] Shuffling with limited buffer_size then will not mix 0 and 1 labels, shuffling with buffer over
        # whole data set negates the effect of memory saving since the memory occupancy will be too large for large
        # datasets
        def order_shape(arr):
            result = np.zeros((self.distance + 1, self.distance + 1, 3))
            z_stabs = np.zeros((self.distance + 1, self.distance + 1))
            x_stabs = np.zeros((self.distance + 1, self.distance + 1))
            all_stabs = np.zeros((self.distance + 1, self.distance + 1))
            shift = True
            for k in np.arange(self.distance + 1):
                for h in np.arange(1, self.distance, 2):
                    if shift:
                        z_stabs[k, h + 1] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]  # h starts at 1,
                        # with 2 increment!
                        all_stabs[k, h + 1] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                    elif not shift:
                        z_stabs[k, h] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                        all_stabs[k, h] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                shift = not shift
            shift = False
            for k in np.arange(1, self.distance):
                for h in np.arange(0, self.distance + 1, 2):
                    if shift:
                        x_stabs[k, h + 1] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                        all_stabs[k, h + 1] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                    elif not shift:
                        x_stabs[k, h] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                        all_stabs[k, h] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                shift = not shift
            result[:, :, 0] = all_stabs
            result[:, :, 1] = z_stabs
            result[:, :, 2] = x_stabs
            return result

        syndromes = []
        for _ in np.arange(rounds):
            for noise in self.noises:
                code = DepolarizingSurface(self.distance, noise)
                syndromes = syndromes + code.get_syndromes(int(num / rounds))
        syndromes = list(map(lambda x: torch.where(x, -1, 1), syndromes))
        # Bring data into right shape
        syndromes_new = np.zeros((len(syndromes), self.distance + 1, self.distance + 1, 3))
        # 1: all stabilizers, 2: only Z stabilizers, 3: only X stabilizers # TODO change this since we use VAEs now
        for i, elm in enumerate(self.syndromes):
            syndromes_new[i] = order_shape(elm)
        self.syndromes = torch.as_tensor(syndromes_new, device=self.device)
        return self

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self, [ratio, 1 - ratio])
        return torch.as_tensor(dataset_train), torch.as_tensor(dataset_val)
