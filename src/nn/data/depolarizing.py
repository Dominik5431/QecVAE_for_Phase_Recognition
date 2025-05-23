import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from .qecdata import QECDataset
from src.error_code import DepolarizingToricCode, SurfaceCode


class DepolarizingToricData(QECDataset):
    """
        Implements a custom Dataset for syndromes of the Toric code under depolarizing noise.
    """
    def __init__(self, distance, noises, name, load, random_flip, device, sequential: bool = False,
                 cluster: bool = False, only_syndromes: bool = False):
        super().__init__(distance=distance, noises=noises, name=name, load=load, device=device, random_flip=random_flip,
                         cluster=cluster, only_syndromes=only_syndromes)
        self.sequential = sequential

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, index):
        output = (self.syndromes[index],)
        if not self.only_syndromes:
            output = output + (self.logical[index],)
        if not self.train:
            output = output + (self.flips[index],)
        return output

    def generate_data(self, n):
        syndromes_list = []  # measurement syndromes
        flips_list = []  # record if a syndrome has been flipped purposely
        logical_list = []  # measured logicals

        # Generate data for each noise value
        for noise in tqdm(self.noises):
            code = DepolarizingToricCode(self.distance, noise, self.random_flip)

            if self.train or not self.random_flip:
                syndromes_noise = code.get_syndromes(n, self.train)
                flips_noise = None
            else:
                syndromes_noise, flips_noise = code.get_syndromes(n, self.train)
                flips_list.extend(flips_noise)  # Extend instead of repeated list addition

            syndromes_list.append(syndromes_noise)

        # Stack syndromes for efficient reshaping
        syndromes = np.vstack(syndromes_list).reshape(-1, 2, self.distance, self.distance)
        syndromes = syndromes.astype(np.float32)
        output = (torch.as_tensor(syndromes, device=self.device),)

        if not self.only_syndromes:
            output = output + (torch.as_tensor(np.array(logical_list), device=self.device, dtype=torch.float32),)
        if self.random_flip:
            output = output + (torch.as_tensor(np.array(flips_list), device=self.device, dtype=torch.float32),)

        return output

    def get_train_test_data(self, ratio):  # think about if really necessary or if there is a nicer solution
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val


class DepolarizingSurfaceData(QECDataset):
    """
        Implements a custom Dataset for syndromes of the Surface code under depolarizing noise.
    """
    def __init__(self, distance: int, noises, name: str, load: bool, device: torch.device, cluster: bool = False,
                 only_syndromes: bool = False):
        super().__init__(distance, noises, name, load, device, cluster, only_syndromes)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, idx):
        return self.syndromes[idx]

    def generate_data(self, n, only_syndromes: bool = False):
        syndromes = []  # measurement syndromes
        for noise in self.noises:
            c = SurfaceCode(self.distance, noise, noise_model='depolarizing')
            syndromes_noise = c.get_syndromes(n, only_syndromes=only_syndromes)
            syndromes = syndromes + list(syndromes_noise)
        # data is already provided sequentially as [syndromes, noise]
        return torch.as_tensor(np.array(syndromes), device=self.device)

    def get_train_val_data(self, ratio=0.8):
        # Splits samples into train and validation data
        train_set, val_set = torch.utils.data.random_split(self, [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return train_set, val_set
