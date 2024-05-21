import unittest

from HARK.distribution import Bernoulli
import HARK.model as model
from HARK.model import Control
import HARK.models.consumer as cons

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
        self.cblock = cons.consumption_block_normalized

        self.pre = {"k": 2, "R": 1.05, "PermGroFac": 1.1, "theta": 1, "CRRA": 2}

        self.dr = {"c": lambda m: m}

    def test_init(self):
        self.assertEqual(self.test_block_A.name, "test block A")

    def test_transition(self):
        post = self.cblock.transition(self.pre, self.dr)

        self.assertEqual(post["a"], 0)

    def test_calc_reward(self):
        self.assertEqual(self.cblock.calc_reward({"c": 1, "CRRA": 2})["u"], -1.0)

    def test_state_action_value_function(self):
        savf = self.cblock.state_action_value_function_from_continuation(lambda a: 0)

        av = savf(self.pre, self.dr)

        self.assertEqual(av, -0.34375)

        cv = 1
        dv = self.cblock.decision_value_function(self.dr, lambda a: cv)(self.pre)

        self.assertEqual(dv, av + cv)


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
