import unittest

import HARK.simulation as simulation

class SimulationTests(unittest.TestCase):
    '''
    Tests for simulation.py sampling distributions
    with default seed.
    '''

    def test_drawMeanOneLognormal(self):
        self.assertEqual(
            simulation.drawMeanOneLognormal(1)[0],
            3.5397367004222002)

    def test_drawLognormal(self):
        self.assertEqual(
            simulation.drawLognormal(1)[0],
            5.836039190663969)

    def test_drawNormal(self):
        self.assertEqual(
            simulation.drawNormal(1)[0],
            1.764052345967664)

    def test_drawWeibull(self):
        self.assertEqual(
            simulation.drawWeibull(1)[0],
            0.79587450816311)

    def test_drawUniform(self):
        self.assertEqual(
            simulation.drawUniform(1)[0],
            0.5488135039273248)

    def test_drawBernoulli(self):
        self.assertEqual(
            simulation.drawBernoulli(1)[0],
            False)
