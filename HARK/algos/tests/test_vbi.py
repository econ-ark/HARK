import HARK.algos.vbi as vbi
from HARK.distribution import Bernoulli
from HARK.model import Control, DBlock
import numpy as np

import unittest


block_1 = DBlock(
    **{
        "name": "vbi_test_1",
        "shocks": {
            "coin": Bernoulli(p=0.5),
        },
        "dynamics": {
            "m": lambda y, coin: y + coin,
            "c": Control(["m"]),
            "a": lambda m, c: m - c,
        },
        "reward": {"u": lambda c: 1 - (c - 1) ** 2},
    }
)


class test_vbi(unittest.TestCase):
    # def setUp(self):
    #    pass

    def test_solve_block_1(self):
        state_grid = {"m": np.linspace(0, 2, 10)}

        dr, dec_vf, arr_vf = vbi.vbi_solve(block_1, lambda a: a, state_grid)

        self.assertAlmostEqual(dr["c"](**{"m": 1}), 0.5)
