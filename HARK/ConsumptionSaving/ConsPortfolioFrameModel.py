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
from HARK.distribution import Lognormal, MeanOneLogNormal, Bernoulli  # Random draws for simulating agents
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

    # values for aggregate variables
    # to be set when simulation initializes.
    # currently not doing anything because still using old
    # initializeSim()
    aggregate_init_values = {
        'PlvlAggNow' : 1.0
    }

    def birth_aNrmNow(self, N):
        return Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).draw(N)

    def birth_pLvlNow(self, N):
        pLvlInitMeanNow = self.pLvlInitMean + np.log(
            self.state_now["PlvlAggNow"]
        )  # Account for newer cohorts having higher permanent income

        return Lognormal(
            pLvlInitMeanNow,
            self.pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)


    # values to assign to agents at birth
    birth_values = {
        'ShareNow' : 0,
        'AdjustNow' : False,
        'aNrmNow' : birth_aNrmNow,
        'pLvlNow' : birth_pLvlNow
    }

    def transition_ShareNow(self, **context):
        ShareNow = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, self.shocks['AdjustNow'])

            ShareNow[those] = self.solution[t].ShareFuncAdj(self.state_now['mNrmNow'][those])

            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(
                these,
                np.logical_not(self.shocks['AdjustNow']))
            ShareNow[those] = self.solution[t].ShareFuncFxd(
                context['mNrmNow'][those], ShareNow[those]
            )

        # redundant for now
        self.controls["ShareNow"] = ShareNow

        return ShareNow

    def transition_cNrmNow(self, **context):
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        ShareNow = self.controls["ShareNow"]

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, context['AdjustNow'])
            cNrmNow[those] = self.solution[t].cFuncAdj(context['mNrmNow'][those])

            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(
                these,
                np.logical_not(context['AdjustNow']))
            cNrmNow[those] = self.solution[t].cFuncFxd(
                context['mNrmNow'][those], ShareNow[those]
            )

        # Store controls as attributes of self
        # redundant for now
        self.controls["cNrmNow"] = cNrmNow

        return cNrmNow

    frames = {
        ('RiskyNow','AdjustNow') : PortfolioConsumerType.getShocks,
        ('pLvlNow', 'PlvlAggNow', 'bNrmNow', 'mNrmNow') : PortfolioConsumerType.getStates,
        ('ShareNow') : transition_ShareNow,
        ('cNrmNow') : transition_cNrmNow,
        ('aNrmNow', 'aNrmNow') : PortfolioConsumerType.getPostStates
    }
