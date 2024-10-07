import unittest
import numpy as np
from src.error_code.error_code import BitFlipToricCode, DepolarizingToricCode
import matplotlib.pyplot as plt


class TestErrorCode(unittest.TestCase):

    def setUp(self):
        pass

    def test_output_type(self):
        self.assertEqual(type(list), BitFlipToricCode(3, 0, random_flip=True).get_syndromes(10))

    def test_bad_distance(self):
        with self.assertRaises(ValueError):
            BitFlipToricCode(6, 0, random_flip=True).get_syndromes(10)

    def test_initialization(self):
        d = 7
        syndromes = np.array(BitFlipToricCode(d, 0, False).get_syndromes(2))
        print(list(map(lambda x: x[:int(0.5 * len(x))], syndromes)))
        print(len(syndromes[0]))
        self.assertTrue((syndromes == 1).all())

    def test_bit_flip_surface_code(self):
        d = 7
        syndromes = np.array(BitFlipToricCode(d, 0.2, False).get_syndromes(10))
        x_faults = syndromes[:, int(0.5 * (np.shape(syndromes)[1])):]
        self.assertTrue((x_faults == 1).all())

    def test_depolarizing_surface_code(self):
        d = 7
        syndromes = np.array(DepolarizingToricCode(d, 0.01, True).get_syndromes(10))
        print(syndromes)


if __name__ == '__main__':
    unittest.main()
