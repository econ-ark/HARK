import os
import unittest

import HARK.model as model
import HARK.parser as parser
import yaml


class test_consumption_parsing(unittest.TestCase):
    def setUp(self):
        this_file_path = os.path.dirname(__file__)
        consumer_yaml_path = os.path.join(
            this_file_path, "../HARK/models/consumer.yaml"
        )

        self.consumer_yaml_file = open(consumer_yaml_path, "r")

    def test_parse(self):
        self.consumer_yaml_file

        config = yaml.load(self.consumer_yaml_file, Loader=parser.harklang_loader())

        self.assertEqual(config["calibration"]["DiscFac"], 0.96)
        self.assertEqual(config["blocks"][0]["name"], "consumption normalized")

        ## construct and test the consumption block
        cons_norm_block = model.DBlock(**config["blocks"][0])
        cons_norm_block.construct_shocks(config["calibration"])
        cons_norm_block.discretize({"theta": {"N": 5}})
        self.assertEqual(cons_norm_block.calc_reward({"c": 1, "CRRA": 2})["u"], -1.0)

        ## construct and test the portfolio block
        portfolio_block = model.DBlock(**config["blocks"][1])
        portfolio_block.construct_shocks(config["calibration"])
        portfolio_block.discretize({"risky_return": {"N": 5}})
