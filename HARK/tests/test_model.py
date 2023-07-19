from HARK.model import Control, Model
import unittest


class test_Model(unittest.TestCase):
    def setUp(self):

        equations_a = {
            'm' : lambda a, r : a * r,
            'c' : Control(['m']),
            'a' : lambda m, c : m - c
        }

        parameters_a = {
            'r' : 1.02
        }

        parameters_b = {
            'r' : 1.03
        }

        equations_c = {
            'm' : lambda a, r : a * r,
            'c' : Control(['m']),
            'a' : lambda m, c : m - 2 * c
        }

        # similar test to distance_metric
        self.model_a = Model(
            equations = equations_a,
            parameters = parameters_a
        )

        self.model_b = Model(
            equations = equations_a,
            parameters = parameters_b
        )

        self.model_c = Model(
            equations = equations_c,
            parameters = parameters_a
        )

    def test_eq(self):
        self.assertEqual(self.model_a, self.model_a)
        self.assertNotEqual(self.model_a, self.model_b)
        self.assertNotEqual(self.model_a, self.model_c)