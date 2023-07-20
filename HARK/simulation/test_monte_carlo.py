"""
This file implements unit tests for the Monte Carlo simulation module
"""
import unittest

from HARK.distribution import MeanOneLogNormal
from HARK.simulation.monte_carlo import *

shocks = {
    'psi' : MeanOneLogNormal(1),
    'theta' : MeanOneLogNormal(1)

}

pre = {
    'R' : 1.05,
    'aNrm' : 1,
    'gamma' : 1.1,
    'psi' : 1.1, # TODO: draw this from a shock,
    'theta' : 1.1 # TODO: draw this from a shock
}

dynamics = {
    'G' : lambda gamma, psi : gamma * psi,
    'Rnrm' : lambda R, G : R / G,
    'bNrm' : lambda Rnrm, aNrm : Rnrm * aNrm,
    'mNrm' : lambda bNrm, theta : bNrm + theta,
    'cNrm' : Control(['mNrm']),
    'aNrm' : lambda mNrm, cNrm : mNrm - cNrm
}

dr = {
    'cNrm' : lambda mNrm : mNrm / 2
}

class test_draw_shocks(unittest.TestCase):
    def test_draw_shocks(self):

        drawn = draw_shocks(shocks, 2)

        self.assertEqual(len(drawn['psi']), 2)

class test_sim_one_period(unittest.TestCase):
    def test_sim_one_period(self):

        post = sim_one_period(dynamics, pre, dr)

        self.assertAlmostEqual(post['cNrm'], 0.98388429)










###############################################################3

'''
init_parameters = {}
init_parameters["PermGroFac"] = 1.05
init_parameters["PermShkStd"] = 1.5
init_parameters["PermShkCount"] = 5
init_parameters["TranShkStd"] = 3.0
init_parameters["TranShkCount"] = 5
init_parameters["RiskyAvg"] = 1.05
init_parameters["RiskyStd"] = 1.5
init_parameters["RiskyCount"] = 5
init_parameters["Rfree"] = 1.03

frames_A = [
    Frame(("bNrm",), ("aNrm",), transition=lambda Rfree, aNrm: Rfree * aNrm),
    Frame(("mNrm",), ("bNrm", "TranShk"), transition=lambda bNrm: mNrm),
    Frame(("cNrm"), ("mNrm",), control=True),
    Frame(
        ("U"),
        ("cNrm", "CRRA"),  # Note CRRA here is a parameter not a state var
        transition=lambda cNrm, CRRA: (CRRAutility(cNrm, CRRA),),
        reward=True,
        context={"CRRA": 2.0},
    ),
    Frame(("aNrm"), ("mNrm", "cNrm"), transition=lambda mNrm, cNrm: (mNrm - cNrm,)),
]


class test_FrameModel(unittest.TestCase):
    def setUp(self):
        self.model = FrameModel(frames_A, init_parameters)

    def test_init(self):
        self.model.frames.var("aNrm")

        self.assertTrue(
            isinstance(
                list(self.model.frames.var("bNrm").parents.values())[0],
                BackwardFrameReference,
            )
        )

        self.assertTrue(
            isinstance(
                list(self.model.frames.var("aNrm").children.values())[0],
                ForwardFrameReference,
            )
        )

    def test_make_terminal(self):
        terminal_model = self.model.make_terminal()

        self.assertEqual(len(self.model.make_terminal().frames.var("aNrm").children), 0)

    def test_prepend(self):
        double_model = self.model.prepend(self.model)

        self.assertEqual(len(double_model.frames), 10)

    def test_repeat(self):
        repeat_model = self.model.repeat({"bNrm": {"Rfree": [1.01, 1.03, 1.02]}})

        self.assertEqual(len(repeat_model.frames), 15)

        self.assertEqual(repeat_model.frames.var("bNrm_1").context["Rfree"], 1.03)
'''