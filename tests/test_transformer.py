import unittest
import torch
from src.nn.net import EncoderLayer


class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.encoder = EncoderLayer(d_model=1, n_head=1, d_ff=100, dropout=0)

    def test_encode(self):
        distance = 3
        sample = torch.randn(1, distance ** 2, 1)
        result = self.encoder(sample)
        print(result)


if __name__ == '__main__':
    unittest.main()
