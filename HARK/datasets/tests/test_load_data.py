import unittest
from HARK.datasets import load_SCF_wealth_weights


class test_load_SCF_wealth_weights(unittest.TestCase):
    def setUp(self):
        self.SCF_wealth, self.SCF_weights = load_SCF_wealth_weights()

    def test_shape(self):
        self.assertEqual(self.SCF_wealth.shape, (3553,))
        self.assertEqual(self.SCF_weights.shape, (3553,))
