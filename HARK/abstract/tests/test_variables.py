import unittest

import numpy as np
import yaml

import HARK.abstract.variables


class test_pyyaml(unittest.TestCase):
    def setUp(self):
        pass

    def test_partial(self):
        with open("consindshk.yml") as f:
            data = yaml.safe_load(f)

    def test_full(self):
        with open("consindshk_full.yml") as f:
            data = yaml.safe_load(f)
