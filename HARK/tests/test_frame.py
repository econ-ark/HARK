"""
This file implements unit tests for the frame.py module.
"""
import unittest

from HARK.frame import BackwardFrameReference, ForwardFrameReference, Frame, FrameModel
from HARK.rewards import CRRAutility

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
