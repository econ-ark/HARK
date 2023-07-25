import unittest

import numpy as np
import yaml

import HARK.abstract.variables


class test_pyyaml(unittest.TestCase):
    def setUp(self):
        self.path = "HARK/abstract/tests/"

    def test_partial(self):
        with open(self.path + "consindshk.yml") as f:
            data = yaml.safe_load(f)

    def test_full(self):
        with open(self.path + "consindshk_full.yml") as f:
            data = yaml.safe_load(f)
