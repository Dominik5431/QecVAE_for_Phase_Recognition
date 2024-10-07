import unittest

import numpy as np
from matplotlib import pyplot as plt

from src.nn import BitFlipToricData, DepolarizingToricData


class TestDataset(unittest.TestCase):
    BitFlipSurfaceData = BitFlipToricData(13, [0.05, 0.15], 'TestDataset', 100, False, False)

    def setUp(self):
        pass

    def parameters(self):
        print(BitFlipToricData)

    def test_BitFlipSurfaceData(self):
        d = 7
        noise = 0.01
        data_test = BitFlipToricData(distance=d, noises=[noise],
                                     name="BFS_Testing-{0}".format(d),
                                     num=10, load=False, random_flip=False)
        sample = data_test.get_syndromes().cpu().numpy()[0, 0]
        plt.imshow(sample, cmap='magma')
        plt.show()

    def test_DepolarizingSurfaceData(self):
        d = 3
        noise = 0.0
        data_test = (DepolarizingToricData(distance=d, noises=[noise],
                                           name="DS_Testing-{0}".format(d),
                                           load=False, random_flip=False, sequential=False)
                     .training()
                     .initialize(num=10))
        print(data_test.get_syndromes()[0].size())
        sample = data_test.get_syndromes()[0].cpu().numpy()[0]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(sample[0], cmap='magma')
        axs[1].imshow(sample[1], cmap='magma')
        print(data_test.get_syndromes()[1])
        plt.show()
        print(np.sum(sample[0]) + np.sum(sample[1]))


if __name__ == '__main__':
    unittest.main()
