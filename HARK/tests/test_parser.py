import os
import unittest


import HARK.parser as parser
import yaml


class test_consumption_parsing(unittest.TestCase):
    def setUp(self):
        this_file_path = os.path.dirname(__file__)
        consumer_yaml_path = os.path.join(this_file_path, "../models/consumer.yaml")

        self.consumer_yaml_file = open(consumer_yaml_path, "r")

    def test_parse(self):
        self.consumer_yaml_file

        config = yaml.load(self.consumer_yaml_file, Loader=parser.harklang_loader())
        pass

        """
        try:
            config = yaml.load(open('perfect_foresight_full_experimental.yaml', 'r'), Loader= get_loader())

            # data is copied
            assert config['model']['blocks']['consumption'] == config['model']['strategies'][1]['block']
            # data is maintained once in memory and referenced in both places
            assert config['model']['blocks']['consumption'] is config['model']['strategies'][1]['block']

            # object created by parser
            c1 = config['model']['blocks']['consumption']['dynamics']['c']
            c2 = config['model']['strategies'][1]['block']['dynamics']['c']

            # objects are equal
            assert c1 == c2
            # objects are identical in memory; the reference is shared.
                assert c1 is c2

            a_str = config['model']['blocks']['consumption']['dynamics']['a']
            a_expr = parse_expr(a_str)
            a_func = lambdify(list(a_expr.free_symbols), a_expr, "numpy")

            m = np.random.random(100)
            c = np.random.random(100)

            #import pdb; pdb.set_trace()

        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
        """
