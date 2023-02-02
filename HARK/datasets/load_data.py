import csv
import numpy as np
from copy import deepcopy
import os

DATASETS = os.path.dirname(os.path.abspath(__file__)) + "/data/"


def load_SCF_wealth_weights():
    """U.S. Survey of Consumer Finances data

    Returns
    -------
    SCF_wealth, SCF_weights: np.ndarray, np.ndarray
    """
    with open(DATASETS + "SCFwealthDataReduced.txt") as f:
        SCF_reader = csv.reader(f, delimiter="\t")
        SCF_raw = list(SCF_reader)
    SCF_wealth = np.zeros(len(SCF_raw)) + np.nan
    SCF_weights = deepcopy(SCF_wealth)
    for j in range(len(SCF_raw)):
        SCF_wealth[j] = float(SCF_raw[j][0])
        SCF_weights[j] = float(SCF_raw[j][1])
    return SCF_wealth, SCF_weights
