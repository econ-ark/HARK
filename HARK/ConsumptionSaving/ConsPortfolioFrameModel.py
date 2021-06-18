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
from HARK import NullFunc, Frame, FrameAgentType  # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    utility,  # CRRA utility function
    utility_inv,  # Inverse CRRA utility function
    utilityP,  # CRRA marginal utility function
    utility_invP,  # Derivative of inverse CRRA utility function
    utilityP_inv,  # Inverse CRRA marginal utility function
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    init_portfolio,
    solveConsPortfolio,
    PortfolioConsumerType,
    PortfolioSolution
)

from HARK.distribution import combine_indep_dstns
from HARK.distribution import (
    IndexDistribution,
    Lognormal,
    MeanOneLogNormal,
    Bernoulli  # Random draws for simulating agents
)
from HARK.interpolation import (
    
    LinearInterp,  # Piecewise linear interpolation
    CubicInterp,  # Piecewise cubic interpolation
    LinearInterpOnInterp1D,  # Interpolator over 1D interpolations
    BilinearInterp,  # 2D interpolator
    ConstantFunction,  # Interpolator-like class that returns constant value
    IdentityFunction,  # Interpolator-like class that returns one of its arguments
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
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
        'PlvlAgg' : 1.0
    }

    # TODO: streamline this so it can draw the parameters from context
    def birth_aNrmNow(self, N):
        """
        Birth value for aNrmNow
        """
        return Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).draw(N)

    # TODO: streamline this so it can draw the parameters from context
    def birth_pLvlNow(self, N):
        """
        Birth value for pLvlNow
        """
        pLvlInitMeanNow = self.pLvlInitMean + np.log(
            self.state_now["PlvlAgg"]
        )  # Account for newer cohorts having higher permanent income

        return Lognormal(
            pLvlInitMeanNow,
            self.pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)


    def draw_Risky(self):
        """
        Gets the attribute Risky as a single draw from a lognormal distribution.
        Uses the attributes RiskyAvgTrue and RiskyStdTrue if RiskyAvg is time-varying,
        else just uses the single values from RiskyAvg and RiskyStd.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "RiskyDstn" in self.time_vary:
            RiskyAvg = self.RiskyAvgTrue
            RiskyStd = self.RiskyStdTrue
        else:
            RiskyAvg = self.RiskyAvg
            RiskyStd = self.RiskyStd
        RiskyAvgSqrd = RiskyAvg ** 2
        RiskyVar = RiskyStd ** 2

        mu = np.log(RiskyAvg / (np.sqrt(1.0 + RiskyVar / RiskyAvgSqrd)))
        sigma = np.sqrt(np.log(1.0 + RiskyVar / RiskyAvgSqrd))
        self.shocks["Risky"] = Lognormal(
            mu, sigma, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(1)

        return self.shocks["Risky"]

    def transition_ShareNow(self, **context):
        """
        Transition method for ShareNow.
        """
        ShareNow = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, self.shocks['Adjust'])

            ShareNow[those] = self.solution[t].ShareFuncAdj(self.state_now['mNrm'][those])

            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(
                these,
                np.logical_not(self.shocks['Adjust']))
            ShareNow[those] = self.solution[t].ShareFuncFxd(
                context['mNrm'][those], ShareNow[those]
            )

        self.controls["Share"] = ShareNow

        return ShareNow

    def transition_cNrmNow(self, **context):
        """
        Transition method for cNrmNow.
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        ShareNow = self.controls["Share"]

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, context['Adjust'])
            cNrmNow[those] = self.solution[t].cFuncAdj(context['mNrm'][those])

            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(
                these,
                np.logical_not(context['Adjust']))
            cNrmNow[those] = self.solution[t].cFuncFxd(
                context['mNrm'][those], ShareNow[those]
            )

        # Store controls as attributes of self
	    # redundant for now
        self.controls["cNrm"] = cNrmNow
        
        return cNrmNow

    frames = [
        Frame(
            ('Risky'),None, 
            transition = draw_Risky
        ),
        Frame(
            ('Adjust'),None, 
            default = {'Adjust' : False},
            transition = IndexDistribution(
                Bernoulli,
                {'p' : init_portfolio['AdjustPrb']},
                # seed=self.RNG.randint(0, 2 ** 31 - 1) : TODO: Seed logic
            ) # self.t_cycle input implied
        ),
        Frame(
            ('pLvl', 'PlvlAgg', 'bNrm', 'mNrm'), None,
            default = {'pLvl' : birth_pLvlNow},
            transition = PortfolioConsumerType.get_states
        ),
        Frame(
            ('Share'), None,
            default = {'Share' : 0}, 
            transition = transition_ShareNow
        ),
        Frame(
            ('cNrm'), None, 
            transition = transition_cNrmNow
        ),
        Frame(
            ('aNrm', 'aLvl'), None,
            default = {'aNrm' : birth_aNrmNow}, 
            transition = PortfolioConsumerType.get_poststates
        )
    ]
