import unittest

from HARK.distribution import Bernoulli
import HARK.model as model
from HARK.model import Control

# TODO: let the shock constructor reference this parameter.
LivPrb = 0.98

test_block_A_data = {
    "name": "test block A",
    "shocks": {
        "live": Bernoulli(p=LivPrb),
    },
    "dynamics": {
        "y": lambda p: p,
        "m": lambda Rfree, a, y: Rfree * a + y,
        "c": Control(["m"]),
        "p": lambda PermGroFac, p: PermGroFac * p,
        "a": lambda m, c: m - c,
    },
    "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
}

test_block_B_data = {"name": "test block B", "shocks": {"SB": Bernoulli(p=0.1)}}

test_block_C_data = {"name": "test block B", "shocks": {"SC": Bernoulli(p=0.2)}}

test_block_D_data = {"name": "test block D", "shocks": {"SD": Bernoulli(p=0.3)}}


class test_DBlock(unittest.TestCase):
    def setUp(self):
        self.test_block_A = model.DBlock(**test_block_A_data)

    def test_init(self):
        self.assertEqual(self.test_block_A.name, "test block A")


class test_RBlock(unittest.TestCase):
    def setUp(self):
        self.test_block_B = model.DBlock(**test_block_B_data)
        self.test_block_C = model.DBlock(**test_block_C_data)
        self.test_block_D = model.DBlock(**test_block_D_data)

    def test_init(self):
        r_block_tree = model.RBlock(
            blocks=[
                self.test_block_B,
                model.RBlock(blocks=[self.test_block_C, self.test_block_D]),
            ]
        )

        r_block_tree.get_shocks()
        self.assertEqual(len(r_block_tree.get_shocks()), 3)
