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

    def test_Lognormal(self):
        self.assertEqual(
            simulation.Lognormal().draw(1)[0],
            5.836039190663969)

    def test_Normal(self):
        self.assertEqual(
            simulation.Normal().draw(1)[0],
            1.764052345967664)

    def test_Weibull(self):
        self.assertEqual(
            simulation.Weibull().draw(1)[0],
            0.79587450816311)

    def test_Uniform(self):
        self.assertEqual(
            simulation.Uniform().draw(1)[0],
            0.5488135039273248)

    def test_Bernoulli(self):
        self.assertEqual(
            simulation.Bernoulli().draw(1)[0],
            False)
