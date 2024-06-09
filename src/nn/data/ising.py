from pathlib import Path

import torch
from torch.utils.data import Dataset


class IsingData(Dataset):
    def __init__(self, name, L, T_max, delta, configs=None):
        super(IsingData, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.L = L
        if configs is None:
            try:
                self.load()
            except FileNotFoundError:
                self.configs = None
        else:
            self.configs = configs

    def __len__(self):
        return self.configs.size(dim=0)

    def __getitem__(self, index):
        return self.configs.data[index]

    def collate(self, batch):
        if self.configs is None:
            self.configs = batch
            self.configs = self.configs.to(self.device)
        else:
            self.configs = torch.cat((self.configs, batch.to(self.device)), dim=1)
        return self

    def save(self):
        torch.save(self.configs, str(Path().resolve().parent) + "/data/syndromes_{0}-{1}.pt".format(self.name, self.L))
        # torch.save(self.configs, "data/syndromes_{0}.pt".format(self.name))

    def load(self):
        self.configs = torch.load(str(Path().resolve().parent) + "/data/syndromes_{0}-{1}.pt".format(self.name, self.L),
                                  mmap=True)  # TODO check if mmap reduces memory usage
        # self.configs = torch.load("data/syndromes_{0}.pt".format(self.name), mmap=True)
        return self

    def get_configs(self):
        return self.configs

    def get_train_test_data(self, ratio=3 / 4):
        self.configs = torch.permute(self.configs, [1, 0, 2, 3])
        configs_train, configs_test = torch.split(self.configs, [15, 25])
        assert len(configs_train) + len(configs_test) == len(self.configs)
        dataset_train, dataset_val = torch.split(configs_train, [int(ratio * len(configs_train)), len(configs_train) - int(ratio * len(configs_train))])
        dataset_train = torch.reshape(dataset_train, (dataset_train.size(0) * dataset_train.size(1), 1, dataset_train.size(2), dataset_train.size(3)))
        dataset_val = torch.reshape(dataset_val, (dataset_val.size(0) * dataset_val.size(1), 1, dataset_val.size(2), dataset_val.size(3)))
        return dataset_train, dataset_val, torch.permute(configs_test, [1, 0, 2, 3])