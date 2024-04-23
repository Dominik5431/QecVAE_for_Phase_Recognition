import unittest
import numpy as np
from src.NN.dataset import BitFlipSurfaceData


class TestDataset(unittest.TestCase):
    BitFlipSurfaceData = BitFlipSurfaceData(13, [0.05, 0.15], 'TestDataset', 100, False)

    def setUp(self):
        pass

    def parameters(self):
        print(BitFlipSurfaceData)


if __name__ == '__main__':
    unittest.main()
