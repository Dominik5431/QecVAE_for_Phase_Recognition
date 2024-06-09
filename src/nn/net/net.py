import torch
import torch.nn as nn
from pathlib import Path


class Net(nn.Module):
    def __init__(self, name):
        super(Net, self).__init__()
        self.name = name

    def save(self):
        torch.save(self.state_dict(), str(Path().resolve().parent) + "/data/{0}.pt".format(self.name))
        # torch.save(self.state_dict(), "data/net_{0}.pt".format(self.name))

    def load(self):
        self.load_state_dict(torch.load(str(Path().resolve().parent) + "/data/{0}.pt".format(self.name)))
        # self.load_state_dict(torch.load("data/net_{0}.pt".format(self.name)))

