'''
Tests that the gentle intro to HARK notebook runs correctly
'''
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import str
from builtins import zip
from builtins import range
from builtins import object

import os
import sys

import nbformat
import unittest
from nbconvert.preprocessors import ExecutePreprocessor

class TestGentleIntroToHark(unittest.TestCase):

    def test_notebook_runs(self):
        # we only test that the notebook works in python3
        if sys.version_info[0] < 3:
            return

        test_path = os.path.dirname(os.path.realpath(__file__))
        nb_path = os.path.join(test_path, '..', 'Gentle-Intro-To-HARK.ipynb')
        with open(nb_path) as nb_f:
            nb = nbformat.read(nb_f, as_version=nbformat.NO_CONVERT)

        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        ep.allow_errors = True
        # this actually runs the notebook
        ep.preprocess(nb, {})

        errors = []
        for cell in nb.cells:
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if output.output_type == 'error':
                        errors.append(output)

        self.assertFalse(errors)
