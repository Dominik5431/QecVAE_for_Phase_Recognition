import unittest
import numpy as np
from src.ErrorCode.error_code import BitFlipSurface


class TestErrorCode(unittest.TestCase):

    def setUp(self):
        pass

    def test_output(self):
        self.assertEqual(type(BitFlipSurface(3, 0).get_syndromes(10)), type(np.zeros((3, 0))))

    def test_bad_distance(self):
        with self.assertRaises(ValueError):
            BitFlipSurface(6, 0).get_syndromes(10)


if __name__ == '__main__':
    unittest.main()
