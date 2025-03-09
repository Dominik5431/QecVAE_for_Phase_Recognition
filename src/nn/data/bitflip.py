import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.error_code.error_code import BitFlipToricCode
import numpy as np
from .qecdata import QECDataset


class BitFlipToricData(QECDataset):
    """
    Implements a custom Dataset for syndromes of the Toric code under bit-flip noise.
    """
    def __init__(self, distance: int, noises, name: str, load: bool, random_flip: bool, device, sequential: bool = False,
                 cluster: bool = False, only_syndromes : bool = False):
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

        for noise in self.noises:
            code = BitFlipToricCode(self.distance, noise, self.random_flip)

            if self.train or not self.random_flip:
                syndromes_noise = code.get_syndromes(n, self.train)
                flips_noise = None
            else:
                syndromes_noise, flips_noise = code.get_syndromes(n, self.train)
                flips_list.extend(flips_noise)  # Extend instead of repeated list addition

            syndromes_list.append(syndromes_noise[:, :syndromes_noise.shape[1] // 2])  # NumPy slicing

        # Stack syndromes for efficient reshaping
        syndromes = np.vstack(syndromes_list).reshape(-1, 1, self.distance, self.distance)

        # Convert to torch tensors
        output = (torch.as_tensor(syndromes, device=self.device, dtype=torch.float32),)

        if not self.only_syndromes:
            logical_tensor = torch.as_tensor(np.array(logical_list), device=self.device, dtype=torch.float32)
            output += (logical_tensor,)

        if self.random_flip:
            flips_tensor = torch.as_tensor(np.array(flips_list), device=self.device, dtype=torch.float32)
            output += (flips_tensor,)

        return output

    def get_train_test_data(self, ratio):
        # Splits train and validation data.
        dataset_train, dataset_val = torch.utils.data.random_split(self,
                                                                   [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return dataset_train, dataset_val

