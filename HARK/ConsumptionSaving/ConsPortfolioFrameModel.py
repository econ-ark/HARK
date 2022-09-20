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
from HARK.frame import Frame, FrameAgentType, FrameModel

from HARK.ConsumptionSaving.ConsIndShockModel import LognormPermIncShk
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    init_portfolio,
    PortfolioConsumerType,
)

from HARK.distribution import combine_indep_dstns, add_discrete_outcome_constant_mean
from HARK.distribution import (
    IndexDistribution,
    Lognormal,
    MeanOneLogNormal,
    Bernoulli  # Random draws for simulating agents
)
from HARK.utilities import (
    CRRAutility,
)

class PortfolioConsumerFrameType(FrameAgentType, PortfolioConsumerType):
    """
    A consumer type with a portfolio choice, using Frame architecture.

    A subclass of PortfolioConsumerType for now.
    This is mainly to keep the _solver_ logic intact.
    """

    def __init__(self, **kwds):
        params = init_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(
            self, **kwds
        )
        # Initialize a basic consumer type
        FrameAgentType.__init__(
            self, self.model, **kwds
        )

        self.shocks = {}
        self.controls = {}
        self.state_now = {}

    def solve(self):
        # Some contortions are needed here to make decision rule shaped objects
        # out of the HARK solution objects

        super().solve(self)

        ## TODO: make this a property of FrameAgentTypes or FrameModels?
        self.decision_rules = {}

        def decision_rule_Share_from_solution(solution_t):
            def decision_rule_Share(Adjust, mNrm, Share):
                Share = np.zeros(len(Adjust)) + np.nan

                Share[Adjust] = solution_t.ShareFuncAdj(mNrm[Adjust])

                Share[~Adjust] = solution_t.ShareFuncFxd(mNrm[~Adjust], Share[~Adjust])

                return Share

            return decision_rule_Share

        def decision_rule_cNrm_from_solution(solution_t):
            def decision_rule_cNrm(Adjust, mNrm, Share):
                cNrm = np.zeros(len(Adjust)) + np.nan

                cNrm[Adjust] = solution_t.cFuncAdj(mNrm[Adjust])

                cNrm[~Adjust] = solution_t.cFuncFxd(
                    mNrm[~Adjust], Share[~Adjust]
                )

                return cNrm

            return decision_rule_cNrm

        self.decision_rules[('Share',)] = [decision_rule_Share_from_solution(sol) for sol in self.solution]
        self.decision_rules[('cNrm',)] = [decision_rule_cNrm_from_solution(sol) for sol in self.solution]

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

    # maybe replace reference to init_portfolio to self.parameters?
    model = FrameModel([
        # todo : make an aggegrate value
        Frame(('PermShkAgg',), ('PermGroFacAgg',),
            transition = lambda PermGroFacAgg : (PermGroFacAgg,),
            aggregate = True
        ),
        Frame(
            ('PermShk'), None,
            default = {'PermShk' : 1.0}, # maybe this is unnecessary because the shock gets sampled at t = 0
            # this is discretized before it's sampled
            transition = IndexDistribution(
                    Lognormal.from_mean_std,
                    {
                        'mean' : init_portfolio['PermGroFac'],
                        'std' : init_portfolio['PermShkStd']
                    }
                ).approx(
                    init_portfolio['PermShkCount'], tail_N=0
                ),
        ),
        Frame(
            ('TranShk'), None,
            default = {'TranShk' : 1.0}, # maybe this is unnecessary because the shock gets sampled at t = 0
            transition = add_discrete_outcome_constant_mean(
                IndexDistribution(
                    MeanOneLogNormal,
                    {
                        'sigma' : init_portfolio['TranShkStd']
                    }).approx(
                        init_portfolio['TranShkCount'], tail_N=0
                    ),
                    p = init_portfolio['UnempPrb'], x = init_portfolio['IncUnemp'] 
            )
        ),
        Frame( ## TODO: Handle Risky as an Aggregate value
            ('Risky'), None, 
            transition = IndexDistribution(
                Lognormal.from_mean_std,
                {
                    'mean' : init_portfolio['RiskyAvg'],
                    'std' : init_portfolio['RiskyStd']
                }
                # seed=self.RNG.randint(0, 2 ** 31 - 1) : TODO: Seed logic
            ).approx(
                init_portfolio['RiskyCount']
            ),
            aggregate = True
        ),
        Frame(
            ('Adjust'), None, 
            default = {'Adjust' : False},
            transition = IndexDistribution(
                Bernoulli,
                {'p' : init_portfolio['AdjustPrb']},
                # seed=self.RNG.randint(0, 2 ** 31 - 1) : TODO: Seed logic
            ) # self.t_cycle input implied
        ),
        Frame(
            ('Rport'), ('Share', 'Risky', 'Rfree'), 
            transition = lambda Share, Risky, Rfree : (Share * Risky + (1.0 - Share) * Rfree,)
        ),
        Frame(
            ('PlvlAgg'), ('PlvlAgg', 'PermShkAgg'), 
            default = {'PlvlAgg' : 1.0},
            transition = lambda PlvlAgg, PermShkAgg : PlvlAgg * PermShkAgg,
            aggregate = True
        ),
        Frame(
            ('pLvl',),
            ('pLvl', 'PermShk'),
            default = {'pLvl' : birth_pLvlNow},
            transition = lambda pLvl, PermShk : (pLvl * PermShk,)
        ),
        Frame(
            ('bNrm',),
            ('aNrm', 'Rport', 'PermShk'), 
            transition = lambda aNrm, Rport, PermShk: (Rport / PermShk) * aNrm
        ),
        Frame(
            ('mNrm',),
            ('bNrm', 'TranShk'),
            transition = lambda bNrm, TranShk : (bNrm + TranShk,)
        ),
        Frame(
            ('Share'), ('Adjust', 'mNrm', 'Share'),
            default = {'Share' : 0}, 
            control = True
        ),
        Frame(
            ('cNrm'), ('Adjust','mNrm','Share'), 
            control = True
        ),
        Frame(
            ('U'), ('cNrm','CRRA'), ## Note CRRA here is a parameter not a state var
            transition = lambda cNrm, CRRA : (CRRAutility(cNrm, CRRA),),
            reward = True
        ),
        Frame(
            ('aNrm'), ('mNrm', 'cNrm'),
            default = {'aNrm' : birth_aNrmNow},
            transition = lambda mNrm, cNrm : (mNrm - cNrm,)
        ),
        Frame(
            ('aLvl'), ('aNrm', 'pLvl'),
            transition = lambda aNrm, pLvl : (aNrm * pLvl,)
        )
    ],
    init_portfolio)
