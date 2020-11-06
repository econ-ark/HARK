"""
This file contains classes and functions for representing,
solving, and simulating agents who must allocate their resources
among consumption, saving in a risk-free asset (with a low return),
and saving in a risky asset (with higher average return).

This file also demonstrates a "frame" model architecture.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from copy import deepcopy
from HARK import HARKobject, NullFunc, FrameAgentType  # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    ValueFunc,  # For representing 1D value function
    MargValueFunc,  # For representing 1D marginal value function
    utility,  # CRRA utility function
    utility_inv,  # Inverse CRRA utility function
    utilityP,  # CRRA marginal utility function
    utility_invP,  # Derivative of inverse CRRA utility function
    utilityP_inv,  # Inverse CRRA marginal utility function
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    ValueFunc2D,  # For representing 2D value function
    MargValueFunc2D,  # For representing 2D marginal value function
)

from HARK.ConsumptionSaving.ConsPortfolioModel import (
    init_portfolio,
    solveConsPortfolio,
    PortfolioConsumerType,
    PortfolioSolution
)

from HARK.distribution import combineIndepDstns
from HARK.distribution import Lognormal, Bernoulli  # Random draws for simulating agents
from HARK.interpolation import (
    LinearInterp,  # Piecewise linear interpolation
    CubicInterp,  # Piecewise cubic interpolation
    LinearInterpOnInterp1D,  # Interpolator over 1D interpolations
    BilinearInterp,  # 2D interpolator
    ConstantFunction,  # Interpolator-like class that returns constant value
    IdentityFunction,  # Interpolator-like class that returns one of its arguments
)

class PortfolioConsumerFrameType(FrameAgentType, PortfolioConsumerType):
    """
    A consumer type with a portfolio choice, using Frame architecture.

    A subclass of PortfolioConsumerType for now.
    This is mainly to keep the _solver_ logic intact.
    """

    init = {
        'ShareNow' : lambda : 0,
        'AdjustNow' : lambda : False
    }

    frames = {
        ('RiskyNow','AdjustNow') : PortfolioConsumerType.getShocks,
        ('pLvlNow', 'PlvlAggNow', 'bNrmNow', 'mNrmNow') : PortfolioConsumerType.getStates,
        ('cNrmNow', 'ShareNow') : PortfolioConsumerType.getControls,
        ('aNrmNow', 'aNrmNow') : PortfolioConsumerType.getPostStates
    }
