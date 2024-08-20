import torch
from torch.utils.data import Dataset
import numpy as np
from .qecdata import QECDataset
from src.error_code import DepolarizingSurfaceCode


# TODO cross-check with code in bitflip.py upon changes I made when adapting the respective code

class DepolarizingSurfaceData(QECDataset):
    def __init__(self, distance, noises, name, load, random_flip, sequential: bool = False, cluster: bool = False):
        super().__init__(distance, noises, name, load, random_flip, cluster)
        self.sequential = sequential

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        if self.train:
            return self.syndromes[index]
        else:
            return self.syndromes[index], self.flips[index]

    def generate_data(self, n, rounds):
        syndromes = []  # deleted rounds, TODO check if shuffling is really doing the proper thing here
        flips = []
        for noise in self.noises:
            code = DepolarizingSurfaceCode(self.distance, noise, self.random_flip)
            if not self.train:
                syndromes_noise, flips_noise = code.get_syndromes(n, self.train)
                flips = flips + flips_noise
            else:
                syndromes_noise = code.get_syndromes(n, self.train)
            syndromes = syndromes + list(syndromes_noise)

        # Bring data into right shape
        # In torch, batch has indices (N,C,H,W)
        if self.sequential:
            syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), self.distance ** 2, 2))
        else:
            syndromes = np.reshape(np.array(syndromes), (n * len(self.noises), 2, self.distance, self.distance))
        if not self.random_flip:
            return torch.as_tensor(syndromes, device=self.device, dtype=torch.double)
        return torch.as_tensor(syndromes, device=self.device, dtype=torch.double), torch.as_tensor(flips, device=self.device, dtype=torch.double)

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val


class DepolarizingRotatedSurfaceData(QECDataset):
    def __init__(self, distance, noises, name, num, load, random_flip):
        super().__init__(distance, noises, name, num, load, random_flip)

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
