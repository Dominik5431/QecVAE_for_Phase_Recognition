import unittest

import numpy as np
from matplotlib import pyplot as plt

from src.nn import BitFlipSurfaceData, DepolarizingSurfaceData


class TestDataset(unittest.TestCase):
    BitFlipSurfaceData = BitFlipSurfaceData(13, [0.05, 0.15], 'TestDataset', 100, False, False)

    def setUp(self):
        pass

    def parameters(self):
        print(BitFlipSurfaceData)

    def test_BitFlipSurfaceData(self):
        d = 7
        noise = 0.01
        data_test = BitFlipSurfaceData(distance=d, noises=[noise],
                                       name="BFS_Testing-{0}".format(d),
                                       num=10, load=False, random_flip=False)
        sample = data_test.get_syndromes().cpu().numpy()[0, 0]
        plt.imshow(sample, cmap='magma')
        plt.show()

    def test_DepolarizingSurfaceData(self):
        d = 33
        noise = 0.01
        data_test = DepolarizingSurfaceData(distance=d, noises=[noise],
                                            name="DS_Testing-{0}".format(d),
                                            num=10, load=False, random_flip=False)
        print(data_test.get_syndromes().size())
        sample = data_test.get_syndromes().cpu().numpy()[0]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(sample[0], cmap='magma')
        axs[1].imshow(sample[1], cmap='magma')
        plt.show()
        print(np.sum(sample[0]) + np.sum(sample[1]))


if __name__ == '__main__':
    unittest.main()
