'''
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
'''
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import product
from copy import deepcopy
from HARK.dcegm import calcMultilineEnvelope
from HARK import HARKobject, NullFunc, AgentType # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import(
    IndShockConsumerType,       # PortfolioConsumerType inherits from it
    ValueFunc,                  # For representing 1D value function
    MargValueFunc,              # For representing 1D marginal value function
    utility,                    # CRRA utility function
    utility_inv,                # Inverse CRRA utility function
    utilityP,                   # CRRA marginal utility function
    utility_invP,               # Derivative of inverse CRRA utility function
    utilityP_inv,               # Inverse CRRA marginal utility function
    init_idiosyncratic_shocks   # Baseline dictionary to build on
)
from HARK.ConsumptionSaving.ConsGenIncProcessModel import(
    ValueFunc2D,                # For representing 2D value function
    MargValueFunc2D             # For representing 2D marginal value function
)
from HARK.distribution import combineIndepDstns 
from HARK.distribution import Lognormal, Bernoulli # Random draws for simulating agents
from HARK.interpolation import(
        LinearInterp,           # Piecewise linear interpolation
        CubicInterp,            # Piecewise cubic interpolation
        LinearInterpOnInterp1D, # Interpolator over 1D interpolations
        BilinearInterpOnInterp1D,
        BilinearInterp,         # 2D interpolator
        TrilinearInterp,        # 3D interpolator
        ConstantFunction,       # Interpolator-like class that returns constant value
        IdentityFunction        # Interpolator-like class that returns one of its arguments
)

from HARK.utilities import makeGridExpMult

class ValueFunc3D(HARKobject):
    '''
    A class for representing a value function in a model with three state
    satirbles. The underlying interpolation is
    in the space of (m,n,s) --> u_inv(v); this class "re-curves" to the value function.
    '''
    distance_criteria = ['func', 'CRRA']

    def __init__(self, vFuncNvrs, CRRA):
        '''
        Constructor for a new value function object.
        Parameters
        ----------
        vFuncNvrs : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources and persistent
            income: u_inv(vFunc(m,p))
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        None
        '''
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA

    def __call__(self, m, n, s):
        '''
        Evaluate the value function at given levels of market resources m and
        persistent income p.
        Parameters
        ----------
        m : float or np.array
            Market resources whose value is to be calcuated.
        n : float or np.array
            Iliquid resources whose value is to be calculated.
        s : float or np.array
            Income contribution shares whose value is to be calculated.
        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with market resources m and
            persistent income p; has same size as inputs m and p.
        '''
        return utility(self.func(m, n, s), gam=self.CRRA)



class MargValueFunc3D(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m,p) = u'(c(m,p)) holds (with CRRA utility).
    '''
    distance_criteria = ['dvdxNvrs', 'CRRA']

    def __init__(self, dvdxNvrsFunc, CRRA):
        '''
        Constructor for a new marginal value function object.
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources and the level of persistent income: uP_inv(vPfunc(m,p)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,p)) = cFunc(m,p).
        CRRA : float
            Coefficient of relative risk aversion.
        Returns
        -------
        None
        '''
        self.dvdxNvrsFunc = deepcopy(dvdxNvrsFunc)
        self.CRRA = CRRA

    def __call__(self, m, n, s):
        '''
        Evaluate the marginal value function at given levels of market resources
        m and persistent income p.
        Parameters
        ----------
        m : float or np.array
            Market resources whose marginal value is to be calcuated.
        n : float or np.array
            Iliquid resources whose marginal value is to be calculated.
        s : float or np.array
            Income contribution shares whose marginal value is to be calculated.
        Returns
        -------
        vP : float or np.array
            Marginal value of the given assets when beginning this period with
            market resources m, iliquid assets n, and contribution share s;
            has same size as inputs m, n, and s.
        '''
        return utilityP(self.dvdxNvrsFunc(m, n, s), gam=self.CRRA)


class RiskyAssetConsumerType(IndShockConsumerType):
    """
    A consumer type that has access to a risky asset with lognormal returns
    that are possibly correlated with his income shocks.
    Investment into the risky asset happens through a "share" that represents
    either
    - The share of the agent's total resources allocated to the risky asset.
    - The share of income that the agent diverts to the risky asset
    depending on the model.
    There is a friction that prevents the agent from adjusting this share with
    an exogenously given probability.
    """
    poststate_vars_ = ['aNrmNow', 'pLvlNow', 'ShareNow', 'AdjustNow']
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + ['DiscreteShareBool']

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):
        params = init_risky.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        IndShockConsumerType.__init__(
            self,
            cycles=cycles,
            verbose=verbose,
            quiet=quiet,
            **kwds
        )

        self.update()


    def preSolve(self):
        AgentType.preSolve(self)
        self.updateSolutionTerminal()


    def update(self):
        self.updateShareGrid()
        self.updateDGrid()
        IndShockConsumerType.update(self)
        self.updateAdjustPrb()
        self.updateRiskyDstn()
        self.updateShockDstn()

    def updateRiskyDstn(self):
        '''
        Creates the attributes RiskyDstn from the primitive attributes RiskyAvg,
        RiskyStd, and RiskyCount, approximating the (perceived) distribution of
        returns in each period of the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Determine whether this instance has time-varying risk perceptions
        if (type(self.RiskyAvg) is list) and (type(self.RiskyStd) is list) and (len(self.RiskyAvg) == len(self.RiskyStd)) and (len(self.RiskyAvg) == self.T_cycle):
            self.addToTimeVary('RiskyAvg','RiskyStd')
        elif (type(self.RiskyStd) is list) or (type(self.RiskyAvg) is list):
            raise AttributeError('If RiskyAvg is time-varying, then RiskyStd must be as well, and they must both have length of T_cycle!')
        else:
            self.addToTimeInv('RiskyAvg','RiskyStd')

        # Generate a discrete approximation to the risky return distribution if the
        # agent has age-varying beliefs about the risky asset
        if 'RiskyAvg' in self.time_vary:
            RiskyDstn = []
            for t in range(self.T_cycle):
                RiskyAvgSqrd = self.RiskyAvg[t] ** 2
                RiskyVar = self.RiskyStd[t] ** 2
                mu = np.log(self.RiskyAvg[t] / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
                sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
                RiskyDstn.append(Lognormal(mu=mu, sigma=sigma).approx(self.RiskyCount))
            self.RiskyDstn = RiskyDstn
            self.addToTimeVary('RiskyDstn')

        # Generate a discrete approximation to the risky return distribution if the
        # agent does *not* have age-varying beliefs about the risky asset (base case)
        else:
            RiskyAvgSqrd = self.RiskyAvg ** 2
            RiskyVar = self.RiskyStd ** 2
            mu = np.log(self.RiskyAvg / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
            sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
            self.RiskyDstn = Lognormal(mu=mu, sigma=sigma).approx(self.RiskyCount)
            self.addToTimeInv('RiskyDstn')


    def updateShockDstn(self):
        '''
        Combine the income shock distribution (over PermShk and TranShk) with the
        risky return distribution (RiskyDstn) to make a new attribute called ShockDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if 'RiskyDstn' in self.time_vary:
            self.ShockDstn = [combineIndepDstns(self.IncomeDstn[t], self.RiskyDstn[t]) for t in range(self.T_cycle)]
        else:
            self.ShockDstn = [combineIndepDstns(self.IncomeDstn[t], self.RiskyDstn) for t in range(self.T_cycle)]
        self.addToTimeVary('ShockDstn')

        # Mark whether the risky returns and income shocks are independent (they are)
        self.IndepDstnBool = True
        self.addToTimeInv('IndepDstnBool')
    
    def updateAdjustPrb(self):
        
        if (type(self.AdjustPrb) is list and (len(self.AdjustPrb) == self.T_cycle)):
            self.addToTimeVary('AdjustPrb')
        elif type(self.AdjustPrb) is list:
            raise AttributeError('If AdjustPrb is time-varying, it must have length of T_cycle!')
        else:
            self.addToTimeInv('AdjustPrb')
            

    def updateShareGrid(self):
        '''
        Creates the attribute ShareGrid as an evenly spaced grid on [0.,1.], using
        the primitive parameter ShareCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.ShareGrid = np.linspace(0.,1.,self.ShareCount)
        self.addToTimeInv('ShareGrid')

    def getRisky(self):
        '''
        Sets the attribute RiskyNow as a single draw from a lognormal distribution.
        Uses the attributes RiskyAvgTrue and RiskyStdTrue if RiskyAvg is time-varying,
        else just uses the single values from RiskyAvg and RiskyStd.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if 'RiskyDstn' in self.time_vary:
            RiskyAvg = self.RiskyAvgTrue
            RiskyStd = self.RiskyStdTrue
        else:
            RiskyAvg = self.RiskyAvg
            RiskyStd = self.RiskyStd
        RiskyAvgSqrd = RiskyAvg**2
        RiskyVar = RiskyStd**2

        mu = np.log(RiskyAvg / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
        sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
        self.RiskyNow = Lognormal(mu, sigma, seed=self.RNG.randint(0, 2**31-1)).draw(1)


    def getAdjust(self):
        '''
        Sets the attribute AdjustNow as a boolean array of size AgentCount, indicating
        whether each agent is able to adjust their risky portfolio share this period.
        Uses the attribute AdjustPrb to draw from a Bernoulli distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.AdjustNow = Bernoulli(self.AdjustPrb, seed=self.RNG.randint(0, 2**31-1)).draw(self.AgentCount)


    def initializeSim(self):
        '''
        Initialize the state of simulation attributes.  Simply calls the same method
        for IndShockConsumerType, then sets the type of AdjustNow to bool.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.initializeSim(self)
        self.AdjustNow = self.AdjustNow.astype(bool)


    def simBirth(self,which_agents):
        '''
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel, then sets Share and Adjust
        to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        IndShockConsumerType.simBirth(self,which_agents)
        self.ShareNow[which_agents] = 0.
        self.AdjustNow[which_agents] = False


    def getShocks(self):
        '''
        Draw idiosyncratic income shocks, just as for IndShockConsumerType, then draw
        a single common value for the risky asset return.  Also draws whether each
        agent is able to update their risky asset share this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.getShocks(self)
        self.getRisky()
        self.getAdjust()


# Define a class to represent the single period solution of the portfolio choice problem
class RiskyContribSolution(HARKobject):
    '''
    A class for representing the single period solution of the portfolio choice model.
    
    Parameters
    ----------
    cFuncAdj : Interp2D
        Consumption function over normalized market resources and iliquid assets when
        the agent is able to adjust their contribution share.
    ShareFuncAdj : Interp2D
        Income share function over normalized market resources and iliquid assets when
        the agent is able to adjust their contribution share.
    DFuncAdj: Interp2D
        Policy function for the flow from the liquid to the iliquid stock of
        assets.
    
    vFuncAdj : ValueFunc2D
        Value function over normalized market resources and iliquid assets when
        the agent is able to adjust their contribution share.
    dvdmFuncAdj : MargValueFunc2D
        Marginal value of mNrm function over normalized market resources and iliquid assets
        when the agent is able to adjust their contribution share.
    dvdnFuncAdj : MargValueFunc2D
        Marginal value of nNrm function over normalized market resources and iliquid assets
        when the agent is able to adjust their contribution share.
    
    vFuncAdj2 : ValueFunc2D
        Stage value function for the sub-period that starts after the agent
        decides his consumption. Over normalized post-consumption liquid
        resources and iliquid assets.
    dvdaFuncAdj2 : MargValueFunc2D
        Derivative of vFuncAdj2 with respect to liquid resources.
    dvdnFuncAdj2 : MargValueFunc2D
        Derivative of vFuncAdj2 with respect to iliquid assets.
    
    vFuncAdj3 : ValueFunc2D
        Stage value function for the sub-period that starts after the agent
        rebalances his assets' allocation. Over normalized end-of-period liquid
        resources and iliquid assets.
    dvdaFuncAdj3 : MargValueFunc2D
        Derivative of vFuncAdj3 with respect to liquid resources.
    dvdnFuncAdj3 : MargValueFunc2D
        Derivative of vFuncAdj3 with respect to iliquid assets.
    
    cFuncFxd : Interp3D
        Consumption function over normalized market resources, iliquid assets and income
        contribution share when the agent is NOT able to adjust their contribution share.
    ShareFuncFxd : Interp3D
        Income share function over normalized market resources, iliquid assets and
        contribution share when the agent is NOT able to adjust their contribution share.
        This should always be an IdentityFunc, by definition.
    dFuncFxd: Interp2D
        Policy function for the flow from the liquid to the iliquid stock of
        assets when the agent is NOT able to adjust their contribution share.
        Should always be 0 by definition.
        
    vFuncFxd : ValueFunc3D
        Value function over normalized market resources, iliquid assets and income contribution
        share when the agent is NOT able to adjust their contribution share.
    dvdmFuncFxd : MargValueFunc3D
        Marginal value of mNrm function over normalized market resources, iliquid assets
        and income contribution share share when the agent is NOT able to adjust 
        their contribution share.
    dvdnFuncFxd : MargValueFunc3D
        Marginal value of nNrm function over normalized market resources, iliquid assets
        and income contribution share share when the agent is NOT able to adjust 
        their contribution share.
    dvdsFuncFxd : MargValueFunc3D
        Marginal value of contribution share function over normalized market resources,
        iliquid assets and income contribution share when the agent is NOT able to adjust
        their contribution share.
    '''
    
    # TODO: what does this do?
    distance_criteria = ['vPfuncAdj']

    def __init__(self,
        
        # Contribution stage
        vFuncSha = None,
        ShareFuncSha = None,
        dvdmFuncSha = None,
        dvdnFuncSha = None,
        
        # Adjusting stage
        vFuncAdj = None,
        DFuncAdj = None,
        dvdmFuncAdj = None,
        dvdnFuncAdj = None,
        
        # Consumption stage
        vFuncFxd = None,
        cFuncFxd = None,
        dvdmFuncFxd = None,
        dvdnFuncFxd = None,
        dvdsFuncFxd = None
        
    ):
        
        # Contribution stage
        if vFuncSha is None:
            vFuncSha = NullFunc()
        if ShareFuncSha is None:
            ShareFuncSha = NullFunc()
        if dvdmFuncSha is None:
            dvdmFuncSha = NullFunc()
        if dvdnFuncSha is None:
            dvdnFuncSha = NullFunc()
        
        # Adjusting stage
        if vFuncAdj is None:
            vFuncAdj = NullFunc()
        if DFuncAdj is None:
            DFuncAdj = NullFunc()
        if dvdmFuncAdj is None:
            dvdmFuncAdj = NullFunc()
        if dvdnFuncAdj is None:
            dvdnFuncAdj = NullFunc()
        
        # Consumption stage
        if vFuncFxd is None:
            vFuncFxd = NullFunc()
        if cFuncFxd is None:
            cFuncFxd = NullFunc()
        if dvdmFuncFxd is None:
            dvdmFuncFxd = NullFunc()
        if dvdnFuncFxd is None:
            dvdmFuncFxd = NullFunc()
        if dvdsFuncFxd is None:
            dvdsFuncFxd = NullFunc()
        
        # Set attributes of self
        self.vFuncSha = vFuncSha
        self.ShareFuncSha = ShareFuncSha
        self.dvdmFuncSha = dvdmFuncSha
        self.dvdnFuncSha = dvdnFuncSha
        
        # Adjusting stage
        self.vFuncAdj = vFuncAdj
        self.DFuncAdj = DFuncAdj
        self.dvdmFuncAdj = dvdmFuncAdj
        self.dvdnFuncAdj = dvdnFuncAdj
        
        # Consumption stage
        self.vFuncFxd = vFuncFxd
        self.cFuncFxd = cFuncFxd
        self.dvdmFuncFxd = dvdmFuncFxd
        self.dvdnFuncFxd = dvdnFuncFxd
        self.dvdsFuncFxd = dvdsFuncFxd
        
# Class for the contribution share stage solution
class RiskyContribShaSolution(HARKobject):
    
    # TODO: what does this do?
    distance_criteria = ['dvdmFuncSha']

    def __init__(self,
        
        # Contribution stage, adjust
        vFuncShaAdj = None,
        ShareFuncAdj = None,
        dvdmFuncShaAdj = None,
        dvdnFuncShaAdj = None,
        
        # Contribution stage, fixed
        vFuncShaFxd = None,
        ShareFuncFxd = None,
        dvdmFuncShaFxd = None,
        dvdnFuncShaFxd = None,
        dvdsFuncShaFxd = None
        
    ):
        
        # Contribution stage, adjust
        if vFuncShaAdj is None:
            vFuncShaAdj = NullFunc()
        if ShareFuncAdj is None:
            ShareFuncAdj = NullFunc()
        if dvdmFuncShaAdj is None:
            dvdmFuncShaAdj = NullFunc()
        if dvdnFuncShaAdj is None:
            dvdnFuncShaAdj = NullFunc()
        
        # Contribution stage, fixed
        if vFuncShaFxd is None:
            vFuncShaFxd = NullFunc()
        if ShareFuncFxd is None:
            ShareFuncFxd = NullFunc()
        if dvdmFuncShaFxd is None:
            dvdmFuncShaFxd = NullFunc()
        if dvdnFuncShaFxd is None:
            dvdnFuncShaFxd = NullFunc()
        if dvdsFuncShaFxd is None:
            dvdsFuncShaFxd = NullFunc()
        
        # Set attributes of self
        self.vFuncShaAdj = vFuncShaAdj
        self.ShareFuncAdj = ShareFuncAdj
        self.dvdmFuncShaAdj = dvdmFuncShaAdj
        self.dvdnFuncShaAdj = dvdnFuncShaAdj
        
        self.vFuncShaFxd = vFuncShaFxd
        self.ShareFuncFxd = ShareFuncFxd
        self.dvdmFuncShaFxd = dvdmFuncShaFxd
        self.dvdnFuncShaFxd = dvdnFuncShaFxd
        self.dvdsFuncShaFxd = dvdsFuncShaFxd
        
# Class for asset adjustment stage solution
class RiskyContribRebSolution(HARKobject):
    
    # TODO: what does this do?
    distance_criteria = ['dvdmFuncReb']

    def __init__(self,
        
        # Rebalancing stage, adjusting
        vFuncRebAdj = None,
        DFuncAdj = None,
        dvdmFuncRebAdj = None,
        dvdnFuncRebAdj = None,
        
        # Rebalancing stage, fixed
        vFuncRebFxd = None,
        DFuncFxd = None,
        dvdmFuncRebFxd = None,
        dvdnFuncRebFxd = None,
        dvdsFuncRebFxd = None
        
    ):
        
        # Rebalancing stage
        if vFuncRebAdj is None:
            vFuncRebAdj = NullFunc()
        if DFuncAdj is None:
            DFuncAdj = NullFunc()
        if dvdmFuncRebAdj is None:
            dvdmFuncRebAdj = NullFunc()
        if dvdnFuncRebAdj is None:
            dvdnFuncRebAdj = NullFunc()
        
        if vFuncRebFxd is None:
            vFuncRebFxd = NullFunc()
        if DFuncFxd is None:
            DFuncFxd = NullFunc()
        if dvdmFuncRebFxd is None:
            dvdmFuncRebFxd = NullFunc()
        if dvdnFuncRebFxd is None:
            dvdnFuncRebFxd = NullFunc()
        if dvdsFuncRebFxd is None:
            dvdsFuncRebFxd = NullFunc()
        
        # Rebalancing stage
        self.vFuncRebAdj = vFuncRebAdj
        self.DFuncAdj = DFuncAdj
        self.dvdmFuncRebAdj = dvdmFuncRebAdj
        self.dvdnFuncRebAdj = dvdnFuncRebAdj
        
        self.vFuncRebFxd = vFuncRebFxd
        self.DFuncFxd = DFuncFxd
        self.dvdmFuncRebFxd = dvdmFuncRebFxd
        self.dvdnFuncRebFxd = dvdnFuncRebFxd
        self.dvdsFuncRebFxd = dvdsFuncRebFxd
        
# Define a class to represent the single period solution of the portfolio choice problem
class RiskyContribConsSolution(HARKobject):
    
    # TODO: what does this do?
    distance_criteria = ['vPfuncAdj']

    def __init__(self,
                
        # Consumption stage
        vFuncCon = None,
        cFunc = None,
        dvdmFuncCon = None,
        dvdnFuncCon = None,
        dvdsFuncCon = None
        
    ):
                
        # Consumption stage
        if vFuncCon is None:
            vFuncCon = NullFunc()
        if cFunc is None:
            cFunc = NullFunc()
        if dvdmFuncCon is None:
            dvdmFuncCon = NullFunc()
        if dvdnFuncCon is None:
            dvdmFuncCon = NullFunc()
        if dvdsFuncCon is None:
            dvdsFuncCon = NullFunc()
        
        # Consumption stage
        self.vFuncCon = vFuncCon
        self.cFunc = cFunc
        self.dvdmFuncCon = dvdmFuncCon
        self.dvdnFuncCon = dvdnFuncCon
        self.dvdsFuncCon = dvdsFuncCon

        
class RiskyContribConsumerType(RiskyAssetConsumerType):
    """
    TODO: model description
    """
    poststate_vars_ = ['aNrmNow', 'nNrmNow', 'pLvlNow', 'ShareNow', 'AdjustNow']
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + ['DiscreteShareBool']

    def __init__(self, cycles=1, verbose=False, quiet=False, **kwds):
    
        params = init_riskyContrib.copy()
        params.update(kwds)
        kwds = params
        
        nPeriods = kwds['T_cycle']
        
        # Update parameters dealing with time to accommodate stages
        stage_time_pars = {'T_cycle': 3*nPeriods}
        pars = ['LivPrb','PermGroFac','PermShkStd','TranShkStd','AdjustPrb','tau']
        for p in pars:
            if type(kwds[p]) is list and len(kwds[p]):
                stage_time_pars[p] = [x for x in kwds[p] for _ in range(3)]
                
        kwds.update(stage_time_pars)
        
        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(
            self,
            cycles=cycles,
            verbose=verbose,
            quiet=quiet,
            **kwds
        )
        
        # Set the solver for the portfolio model, and update various constructed attributes
        self.solveOnePeriod = [solveRiskyContribRebStage,
                               solveRiskyContribShaStage,
                               solveRiskyContribConsStage]*nPeriods
        self.addToTimeVary('solveOnePeriod')
        self.update()
        
        
    def preSolve(self):
        AgentType.preSolve(self)
        self.updateSolutionTerminal()


    def update(self):
        RiskyAssetConsumerType.update(self)
        self.updateNGrid()
        self.updateMGrid()
        self.updateTau()
        
    def updateSolutionTerminal(self):
        '''
        Solves the terminal period of the portfolio choice problem.  The solution is
        trivial, as usual: consume all market resources, and put nothing in the risky
        asset (because you have nothing anyway).
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        
        # We assume that in the last stage, the agent just tries to consume
        # everythin he can. If he is "fixed", that's only liquid resources.
        # If he can adjust, share is irrelevant, but he withdraws and consumes
        # all iliquid resources.
        
        # Consumption and value of the fixed agent
        
        # Can't rebalance
        RFuncFxd_term = ConstantFunction(0.0)
        
        # Consume all thats available (liquid resources)
        cFuncFxd_term = IdentityFunction(i_dim = 0, n_dims = 3)
        vFuncFxd_term = ValueFunc3D(cFuncFxd_term, CRRA = self.CRRA)
        
        # Marginal values of the fixed agent
        dvdmFuncFxd_term = MargValueFunc3D(cFuncFxd_term, CRRA = self.CRRA)
        dvdnFuncFxd_term = ConstantFunction(0.0)
        dvdsFuncFxd_term = ConstantFunction(0.0)
        
        # Rebalancing of the adjusting agent: 
        # Withdraw everything from the pension fund and consume everything
        DFuncAdj_term = ConstantFunction(-1.0)
        
        # Find the withdrawal penalty
        if type(self.tau) is list:
            tau = self.tau[-1]
        else:
            tau = self.tau
        
        # Value and marginal value function of the adjusting agent
        vFuncAdj_term = ValueFunc2D(lambda m,n: m + n/(1+tau), self.CRRA)
        dvdmFuncAdj_term = MargValueFunc2D(lambda m,n: m + n/(1+tau), self.CRRA)
        # A marginal unit of n will be withdrawn and put into m. Then consumed.
        dvdnFuncAdj_term = lambda m,n: dvdmFuncAdj_term(m,n)/(1+tau)
        
        # Construct the terminal period solution
        self.solution_terminal = RiskyContribRebSolution(
            # Rebalancing stage
            vFuncRebAdj = vFuncAdj_term,
            DFuncAdj = DFuncAdj_term,
            dvdmFuncRebAdj = dvdmFuncAdj_term,
            dvdnFuncRebAdj = dvdnFuncAdj_term,
            
            # Adjusting stage
            vFuncRebFxd = vFuncFxd_term,
            DFuncFxd = RFuncFxd_term,
            dvdmFuncRebFxd = dvdmFuncFxd_term,
            dvdnFuncRebFxd = dvdnFuncFxd_term,
            dvdsFuncRebFxd = dvdsFuncFxd_term
        )
        
    
    def updateTau(self):
        
        if (type(self.tau) is list and (len(self.tau) == self.T_cycle)):
            self.addToTimeVary('tau')
        elif type(self.tau) is list:
            raise AttributeError('If tau is time-varying, it must have length of T_cycle!')
        else:
            self.addToTimeInv('tau')
        
    def updateRiskyDstn(self):
        '''
        Creates the attributes RiskyDstn from the primitive attributes RiskyAvg,
        RiskyStd, and RiskyCount, approximating the (perceived) distribution of
        returns in each period of the cycle.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Determine whether this instance has time-varying risk perceptions
        if (type(self.RiskyAvg) is list) and (type(self.RiskyStd) is list) and (len(self.RiskyAvg) == len(self.RiskyStd)) and (len(self.RiskyAvg) == self.T_cycle):
            self.addToTimeVary('RiskyAvg','RiskyStd')
        elif (type(self.RiskyStd) is list) or (type(self.RiskyAvg) is list):
            raise AttributeError('If RiskyAvg is time-varying, then RiskyStd must be as well, and they must both have length of T_cycle!')
        else:
            self.addToTimeInv('RiskyAvg','RiskyStd')
        
        # Generate a discrete approximation to the risky return distribution if the
        # agent has age-varying beliefs about the risky asset
        if 'RiskyAvg' in self.time_vary:
            RiskyDstn = []
            for t in range(self.T_cycle):
                RiskyAvgSqrd = self.RiskyAvg[t] ** 2
                RiskyVar = self.RiskyStd[t] ** 2
                mu = np.log(self.RiskyAvg[t] / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
                sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
                RiskyDstn.append(Lognormal(mu=mu, sigma=sigma).approx(self.RiskyCount))
            self.RiskyDstn = RiskyDstn
            self.addToTimeVary('RiskyDstn')
                
        # Generate a discrete approximation to the risky return distribution if the
        # agent does *not* have age-varying beliefs about the risky asset (base case)
        else:
            RiskyAvgSqrd = self.RiskyAvg ** 2
            RiskyVar = self.RiskyStd ** 2
            mu = np.log(self.RiskyAvg / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
            sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
            self.RiskyDstn = Lognormal(mu=mu, sigma=sigma).approx(self.RiskyCount)
            self.addToTimeInv('RiskyDstn')
            
            
    def updateShockDstn(self):
        '''
        Combine the income shock distribution (over PermShk and TranShk) with the
        risky return distribution (RiskyDstn) to make a new attribute called ShockDstn.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if 'RiskyDstn' in self.time_vary:
            self.ShockDstn = [combineIndepDstns(self.IncomeDstn[t], self.RiskyDstn[t]) for t in range(self.T_cycle)]
        else:
            self.ShockDstn = [combineIndepDstns(self.IncomeDstn[t], self.RiskyDstn) for t in range(self.T_cycle)]
        self.addToTimeVary('ShockDstn')
        
        # Mark whether the risky returns and income shocks are independent (they are)
        self.IndepDstnBool = True
        self.addToTimeInv('IndepDstnBool')
        
        
    def updateShareGrid(self):
        '''
        Creates the attribute ShareGrid as an evenly spaced grid on [0.,1.], using
        the primitive parameter ShareCount.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.ShareGrid = np.linspace(0.,self.ShareMax,self.ShareCount)
        self.addToTimeInv('ShareGrid')
            
    def updateDGrid(self):
        '''
        '''
        aux = np.linspace(0,1,self.dCount)
        self.dGrid = np.concatenate((-1*np.flip(aux[1:]),aux))
        self.addToTimeInv('dGrid')
        
    def updateNGrid(self):
        '''
        Updates the agent's iliquid assets grid by constructing a
        multi-exponentially spaced grid of nNrm values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None.
        '''
        # Extract parameters
        nNrmMin = self.nNrmMin
        nNrmMax = self.nNrmMax
        nNrmCount = self.nNrmCount
        exp_nest = self.nNrmNestFac
        # Create grid
        nNrmGrid = makeGridExpMult(ming = nNrmMin, maxg = nNrmMax, 
                                   ng = nNrmCount, timestonest = exp_nest)
        # Assign and set it as time invariant
        self.nNrmGrid = nNrmGrid
        self.addToTimeInv('nNrmGrid')
    
    def updateMGrid(self):
        '''
        Updates the agent's liquid assets exogenous grid by constructing a
        multi-exponentially spaced grid of mNrm values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None.
        '''
        # Extract parameters
        mNrmMin = self.mNrmMin
        mNrmMax = self.mNrmMax
        mNrmCount = self.mNrmCount
        exp_nest = self.mNrmNestFac
        # Create grid
        mNrmGrid = makeGridExpMult(ming = mNrmMin, maxg = mNrmMax, 
                                   ng = mNrmCount, timestonest = exp_nest)
        # Assign and set it as time invariant
        self.mNrmGrid = mNrmGrid
        self.addToTimeInv('mNrmGrid')
                
    def getRisky(self):
        '''
        Sets the attribute RiskyNow as a single draw from a lognormal distribution.
        Uses the attributes RiskyAvgTrue and RiskyStdTrue if RiskyAvg is time-varying,
        else just uses the single values from RiskyAvg and RiskyStd.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if 'RiskyDstn' in self.time_vary:
            RiskyAvg = self.RiskyAvgTrue
            RiskyStd = self.RiskyStdTrue  
        else:
            RiskyAvg = self.RiskyAvg
            RiskyStd = self.RiskyStd
        RiskyAvgSqrd = RiskyAvg**2
        RiskyVar = RiskyStd**2

        mu = np.log(RiskyAvg / (np.sqrt(1. + RiskyVar / RiskyAvgSqrd)))
        sigma = np.sqrt(np.log(1. + RiskyVar / RiskyAvgSqrd))
        self.RiskyNow = Lognormal(mu, sigma, seed=self.RNG.randint(0, 2**31-1)).draw(1)
        
        
    def getAdjust(self):
        '''
        Sets the attribute AdjustNow as a boolean array of size AgentCount, indicating
        whether each agent is able to adjust their risky portfolio share this period.
        Uses the attribute AdjustPrb to draw from a Bernoulli distribution.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if not ('AdjustPrb' in self.time_vary):
            
            self.AdjustNow = Bernoulli(self.AdjustPrb, seed=self.RNG.randint(0, 2**31-1)).draw(self.AgentCount)
       
        else: 
            
            AdjustNow = np.zeros(self.AgentCount)  # Initialize shock array
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                N = np.sum(these)
                if N > 0:
                    AdjustPrb = self.AdjustPrb[t - 1]
                    AdjustNow[these] = Bernoulli(AdjustPrb,
                                                 seed=self.RNG.randint(0, 2**31-1)).draw(N)
                    
            self.AdjustNow = AdjustNow
       
    def initializeSim(self):
        '''
        Initialize the state of simulation attributes.  Simply calls the same method
        for IndShockConsumerType, then sets the type of AdjustNow to bool.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.initializeSim(self)
        self.AdjustNow = self.AdjustNow.astype(bool)
        self.Stage = 0
    
    
    def simBirth(self,which_agents):
        '''
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel, then sets Share and Adjust
        to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        IndShockConsumerType.simBirth(self,which_agents)
        self.ShareNow[which_agents] = 0.
        self.AdjustNow[which_agents] = False
        
            
    def getShocks(self):
        '''
        Draw idiosyncratic income shocks, just as for IndShockConsumerType, then draw
        a single common value for the risky asset return.  Also draws whether each
        agent is able to update their risky asset share this period.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.getShocks(self)
        self.getRisky()
        self.getAdjust()
        
        
    def getControls(self):
        '''
        Calculates consumption cNrmNow and risky portfolio share ShareNow using
        the policy functions in the attribute solution.  These are stored as attributes.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cNrmNow  = np.zeros(self.AgentCount) + np.nan
        ShareNow = np.zeros(self.AgentCount) + np.nan
        
        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            
            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, self.AdjustNow)
            cNrmNow[those]  = self.solution[t].cFuncAdj(self.mNrmNow[those])
            ShareNow[those] = self.solution[t].ShareFuncAdj(self.mNrmNow[those])
            
            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(these, np.logical_not(self.AdjustNow))
            cNrmNow[those]  = self.solution[t].cFuncFxd(self.mNrmNow[those], self.ShareNow[those])
            ShareNow[those] = self.solution[t].ShareFuncFxd(self.mNrmNow[those], self.ShareNow[those])
        
        # Store controls as attributes of self
        self.cNrmNow = cNrmNow
        self.ShareNow = ShareNow
    
    def simOnePeriod(self):
        """
        Simulates one period for this type.  Calls the methods getMortality(), getShocks() or
        readShocks, getStates(), getControls(), and getPostStates().  These should be defined for
        AgentType subclasses, except getMortality (define its components simDeath and simBirth
        instead) and readShocks.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )
        
        # Simulation steps depend on the stage
        
        # Rebalancing stage (the first one)
        if self.Stage == 0:
            
            # Mortality and birth happens only in the first stage
            self.getMortality()
            
            # Shocks are drawn in the first stage
            if self.read_shocks:  # If shock histories have been pre-specified, use those
                self.readShocks()
            else:  # Otherwise, draw shocks as usual according to subclass-specific method
                self.getShocks()
            
            # Update states and controls
            self.getStatesReb()
            self.getControlsReb()    
            self.getPostStatesReb()
            
        # Contribution stage
        elif self.Stage == 1:
            pass
        
        # Consumption stage (the last one)
        elif self.Stage == 2:
            pass
        
        # Determine each agent's state at decision time
        #self.getControls()  # Determine each agent's choice or control variables based on states
        #self.getPostStates()  # Determine each agent's post-decision / end-of-period states using states and controls

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end
        
        # Update stage
        self.Stage = (self.Stage + 1)%3
        
    def getStatesReb(self):
        """
        Get states for the first stage: rebalancing.
        """
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow
        nNrmPrev = self.nNrmNow
        RfreeNow = self.Rfree
        RriskNow = self.RiskyNow
        
        # Calculate new states:
        
        # Permanent income
        self.pLvlNow = (
            pLvlPrev * self.shocks["PermShkNow"]
        )
        
        # Assets: mNrm and nNrm
        
        # Compute the effective growth factor of each asset
        RfEffNow = (
            RfreeNow / self.shocks["PermShkNow"]
        )
        RrEffNow = (
            RriskNow / self.shocks["PermShkNow"]
        )
        
        self.bNrmNow = RfEffNow * aNrmPrev  # Liquid balances before labor income
        self.gNrmNow = RrEffNow * nNrmPrev  # Iliquid balances before labor income
        
        # Liquid balances after labor income
        self.mNrmNow = (
            self.bNrmNow + self.shocks["TranShkNow"] * (1 - self.ShareNow)
        )
        # Iliquid balances after labor income
        self.nNrmNow = (
            self.gNrmNow + self.shocks["TranShkNow"] * self.ShareNow
        )
        
        return None

    
    def getControlsReb(self):
        """
        """
        DNrmNow = np.zeros(self.AgentCount) + np.nan
        
        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(0,self.T_cycle,3):
            
            # Find agents in this period-stage
            these = t == self.t_cycle
                           
            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.AdjustNow)
            DNrmNow[those] = self.solution[t].DFuncAdj(self.mNrmNow[those], self.nNrmNow[those])
                
            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.AdjustNow))
            DNrmNow[those] = self.solution[t].DFuncFxd(
                self.mNrmNow[those], self.nNrmNow[those], self.ShareNow[those]
            )

        # Store controls as attributes of self
        self.DNrmNow = DNrmNow
        
    def getPostStatesReb(self):
        """
        """
        
        # Post-states are assets after rebalancing

        if not 'tau' in self.time_vary:
        
            mNrmTildeNow, nNrmTildeNow = rebalanceAssets(self.DNrmNow, self.mNrmNow, self.nNrmNow, self.tau)
        
        else:
            
            # Initialize
            mNrmTildeNow = np.zeros_like(self.mNrmNow) + np.nan
            nNrmTildeNow = np.zeros_like(self.mNrmNow) + np.nan
            
            # Loop over each period of the cycle, getting controls separately depending on "age"
            for t in range(0,self.T_cycle,3):
            
                # Find agents in this period-stage
                these = t == self.t_cycle
                
                if np.sum(these) > 0:
                    tau = self.tau[t]
                    
                    mNrmTildeNow[these], nNrmTildeNow[these] = rebalanceAssets(self.DNrmNow[these], self.mNrmNow[these], self.nNrmNow[these], tau)
        
        self.mNrmTildeNow = mNrmTildeNow
        self.nNrmTildeNow = nNrmTildeNow
                    
                       
    
def rebalanceAssets(d,m,n,tau):
    
    # Initialize
    mTil = np.zeros_like(m) + np.nan
    nTil = np.zeros_like(m) + np.nan
    
    # Contributions
    inds = d >= 0
    mTil[inds] = m[inds]*(1-d[inds])
    nTil[inds] = n[inds] + m[inds]*d[inds]
    
    # Withdrawals
    inds = d < 0
    mTil[inds] = m[inds] - d[inds]*n[inds]/(1 + tau)
    nTil[inds] = n[inds]*(1+d[inds])
    
    return (mTil, nTil)

    
def findOptimalRebalance(m,n,vNvrs,tau):
    
    if (m == 0 and n == 0):
        dopt = 0
        fopt = 0
    else:
        fobj = lambda d: -1.*rebalanceFobj(d,m,n,vNvrs,tau)
        # For each case, we optimize numerically and compare with the extremes.
        if m > 0 and n > 0:
            # Optimize contributing and withdrawing separately
            opt_c = minimize_scalar(fobj, bounds=(0, 1), method='bounded')
            opt_w = minimize_scalar(fobj, bounds=(-1, 0), method='bounded')
            
            ds = np.array([opt_c.x,opt_w.x,-1,0,1])
            fs = np.array([opt_c.fun,opt_w.fun,fobj(-1),fobj(0),fobj(1)])
        elif m > 0:
            opt = minimize_scalar(fobj, bounds=(0, 1), method='bounded')
            ds = np.array([opt.x,0,1])
            fs = np.array([opt.fun,fobj(0),fobj(1)])
        else:
            opt = minimize_scalar(fobj, bounds=(-1, 0), method='bounded')
            ds = np.array([opt.x,-1,0])
            fs = np.array([opt.fun,fobj(-1),fobj(0)])
        
        # Pick the best candidate
        ind  = np.argmin(fs)
        dopt = ds[ind]
        fopt = -1.0*fs[ind]
        
                
    m_til, n_til = rebalanceAssets(dopt,m,n,tau)
    return dopt, m_til, n_til, fopt
        
# Consumption stage solver
def solveRiskyContribConsStage(solution_next,ShockDstn,IncomeDstn,RiskyDstn,
                               LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                               BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                               ShareGrid,dGrid,vFuncBool,AdjustPrb,
                               DiscreteShareBool,IndepDstnBool):
    
    # Make sure the individual is liquidity constrained.  Allowing a consumer to
    # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
    if BoroCnstArt != 0.0:
        raise ValueError('PortfolioConsumerType must have BoroCnstArt=0.0!')
        
    # Make sure that if risky portfolio share is optimized only discretely, then
    # the value function is also constructed (else this task would be impossible).
    if (DiscreteShareBool and (not vFuncBool)):
        raise ValueError('PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True!')
          
    # Define temporary functions for utility and its derivative and inverse
    u = lambda x : utility(x, CRRA)
    uP = lambda x : utilityP(x, CRRA)
    uPinv = lambda x : utilityP_inv(x, CRRA)
    uInv = lambda x : utility_inv(x, CRRA)
        
    # Unpack next period's solution
    vFuncRebAdj_next    = solution_next.vFuncRebAdj
    dvdmFuncRebAdj_next = solution_next.dvdmFuncRebAdj
    dvdnFuncRebAdj_next = solution_next.dvdnFuncRebAdj
    
    vFuncRebFxd_next    = solution_next.vFuncRebFxd
    dvdmFuncRebFxd_next = solution_next.dvdmFuncRebFxd
    dvdnFuncRebFxd_next = solution_next.dvdnFuncRebFxd
    dvdsFuncRebFxd_next = solution_next.dvdsFuncRebFxd
    
    # TODO: I am currently contructing the joint distribution of returns and
    # income, even if they are independent. Is there a way to speed things
    # up if they are independent?
    if IndepDstnBool:
        
        ShockDstn = combineIndepDstns(IncomeDstn, RiskyDstn)
    
    # Unpack the shock distribution
    ShockPrbs_next = ShockDstn.pmf
    PermShks_next  = ShockDstn.X[0]
    TranShks_next  = ShockDstn.X[1]
    Risky_next     = ShockDstn.X[2]
    
    # STEP ONE
    # Find end-of-period (continuation) value function and its derivatives.
    
    # It's possible for the agent to end with 0 iliquid assets regardless of
    # future income and probability of adjustment.
    nNrmGrid = np.insert(nNrmGrid, 0, 0.0)  
    
    # Now, under which parameters do we need to consider the possibility
    # of the agent ending with 0 liquid assets?:
    # -If he has guaranteed positive income next period.
    # -If he is sure he can draw on iliquid assets even if income and liquid
    #  assets are 0.
    zero_bound = (np.min(TranShks_next) == 0.)
    if (not zero_bound) or (zero_bound and AdjustPrb == 1.):
        aNrmGrid = np.insert(aXtraGrid, 0, 0.)
    else:
        #aNrmGrid = aXtraGrid
        aNrmGrid = np.insert(aXtraGrid, 0, 0.)
        
    # Create tiled arrays with conforming dimensions. These are used
    # to compute expectations.
    aNrm_N = aNrmGrid.size
    nNrm_N = nNrmGrid.size
    Share_N = ShareGrid.size
    Shock_N = ShockPrbs_next.size
    # Convention will be (a,n,s,Shocks)
    aNrm_tiled      = np.tile(np.reshape(aNrmGrid, (aNrm_N,1,1,1)), (1,nNrm_N,Share_N,Shock_N))
    nNrm_tiled      = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1,1)), (aNrm_N,1,Share_N,Shock_N))
    Share_tiled     = np.tile(np.reshape(ShareGrid, (1,1,Share_N,1)), (aNrm_N,nNrm_N,1,Shock_N))
    ShockPrbs_tiled = np.tile(np.reshape(ShockPrbs_next, (1,1,1,Shock_N)), (aNrm_N,nNrm_N,Share_N,1))
    PermShks_tiled  = np.tile(np.reshape(PermShks_next, (1,1,1,Shock_N)), (aNrm_N,nNrm_N,Share_N,1))
    TranShks_tiled  = np.tile(np.reshape(TranShks_next, (1,1,1,Shock_N)), (aNrm_N,nNrm_N,Share_N,1))
    Risky_tiled     = np.tile(np.reshape(Risky_next, (1,1,1,Shock_N)), (aNrm_N,nNrm_N,Share_N,1))
        
    # Calculate future states
    mNrm_next = Rfree*aNrm_tiled/(PermShks_tiled*PermGroFac) + (1. - Share_tiled)*TranShks_tiled
    nNrm_next = Risky_tiled*nNrm_tiled/(PermShks_tiled*PermGroFac) + Share_tiled*TranShks_tiled
    Share_next = Share_tiled
    
    # Evaluate realizations of the derivatives and levels of next period's
    # value function
    
    # The agent who can adjust starts at the "contrib" stage, the one who can't
    # starts at the Fxd stage.
    
    # Always compute the adjusting version
    vAdj_next    = vFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdmAdj_next = dvdmFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdnAdj_next = dvdnFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdsAdj_next = np.zeros_like(mNrm_next)# No marginal value of Share if it's a free choice!
    
    # We are interested in marginal values before the realization of the
    # adjustment random variable. Compute those objects
    if AdjustPrb < 1.:
        
        # "Fixed" counterparts
        vFxd_next    = vFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdmFxd_next = dvdmFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdnFxd_next = dvdnFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdsFxd_next = dvdsFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        
        # Expected values with respect to adjustment r.v.
        v_next    = AdjustPrb*vAdj_next    + (1.-AdjustPrb)*vFxd_next
        dvdm_next = AdjustPrb*dvdmAdj_next + (1.-AdjustPrb)*dvdmFxd_next
        dvdn_next = AdjustPrb*dvdnAdj_next + (1.-AdjustPrb)*dvdnFxd_next
        dvds_next = AdjustPrb*dvdsAdj_next + (1.-AdjustPrb)*dvdsFxd_next
        
    else: # Don't bother evaluating if there's no chance that contribution share is fixed
        v_next    = vAdj_next
        dvdm_next = dvdmAdj_next
        dvdn_next = dvdnAdj_next
        dvds_next = dvdsAdj_next
        
    # Calculate end-of-period marginal value of both assets by taking expectations
    temp_fac_A = uP(PermShks_tiled*PermGroFac) # Will use this in a couple places
    EndOfPrddvda = DiscFac*Rfree*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*dvdm_next, axis=3)
    EndOfPrddvdn = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*Risky_tiled*dvdn_next, axis=3)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvda)
    EndOfPrddvdnNvrs = uPinv(EndOfPrddvdn)
        
    # Calculate end-of-period value by taking expectations
    temp_fac_B = (PermShks_tiled*PermGroFac)**(1.-CRRA) # Will use this below
    EndOfPrdv = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_B*v_next, axis=3)
    EndOfPrdvNvrs = uInv(EndOfPrdv)
    
    # Construct an interpolator for EndOfPrdV. It will be used later.
    EndOfPrdvFunc = ValueFunc3D(TrilinearInterp(EndOfPrdvNvrs, aNrmGrid,
                                                nNrmGrid, ShareGrid),
                                CRRA)
    
    # Compute post-consumption marginal value of contribution share,
    # conditional on shocks
    EndOfPrddvds_cond_undisc = temp_fac_B*( TranShks_tiled*(dvdn_next - dvdm_next) + dvds_next)
    # Discount and integrate over shocks
    EndOfPrddvds = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*EndOfPrddvds_cond_undisc, axis=3)
    
    # STEP TWO:
    # Solve the consumption problem and create interpolators for c, vCon,
    # and its derivatives.
    
    # Recast a, n, and s now that the shock dimension has been integrated over
    aNrm_tiled  = aNrm_tiled[:,:,:,0]
    nNrm_tiled  = nNrm_tiled[:,:,:,0]
    Share_tiled = Share_tiled[:,:,:,0]
    
    # Apply EGM over liquid resources at every (n,s) to find consumption.
    c = EndOfPrddvdaNvrs
    mNrm_endog = aNrm_tiled + c
    
    # Now construct interpolators for cFxd and the derivatives of vFxd.
    # Since the m grid is different for every (n,s), we use bilinear
    # interpolators of 1-d linear interpolators over m.
    
    # The 1-d interpolators need to take into account whether there is a
    # natural borrowing constraint, which can depend on the value of n. Thus
    # we have to check if mGrid[0] == 0 and construct the interpolators
    # depending on that.
    cInterps = [[] for i in range(nNrm_N)]
    dvdnNvrsInterps = [[] for i in range(nNrm_N)]
    dvdsInterps = [[] for i in range(nNrm_N)]
    
    for nInd in range(nNrm_N):
        for sInd in range(Share_N):
            
            # Extract the endogenous m grid.
            m_end = mNrm_endog[:,nInd,sInd]
            
            # Check if there is a natural constraint
            
            if m_end[0] == 0.0:
                
                # There's no need to insert points since we have m==0.0
                
                # Create consumption interpolator
                cInterps[nInd].append(LinearInterp(m_end,c[:,nInd,sInd]))
                
                # Create dvdnFxd Interpolator
                dvdnNvrsInterps[nInd].append(LinearInterp(m_end,
                                                          EndOfPrddvdnNvrs[:,nInd,sInd]))
                
                # Create dvdsFxd interpolator
                # TODO: this returns NaN when m=n=0. This might propagate.
                # But dvds is not being used at the moment.
                dvdsInterps[nInd].append(LinearInterp(m_end,
                                                      EndOfPrddvds[:,nInd,sInd]))
                
            else:
                
                # We know that:
                # -The lowest gridpoints of both a and n are 0.
                # -Consumption at m < m0 is m.
                # -dvdnFxd at (m,n) for m < m0(n) is dvdnFxd(m0,n)
                # -Same is true for dvdsFxd
                
                # Create consumption interpolator
                cInterps[nInd].append(LinearInterp(np.insert(m_end,0,0),
                                                   np.insert(c[:,nInd,sInd],0,0)
                                                   )
                                      )
                
                # Create dvdnCon Interpolator
                dvdnNvrsInterps[nInd].append(LinearInterp(np.insert(m_end,0,0),
                                                          np.insert(EndOfPrddvdnNvrs[:,nInd,sInd],0,EndOfPrddvdnNvrs[0,nInd,sInd])
                                                          )
                                             )
                
                # Create dvdsCon interpolator
                dvdsInterps[nInd].append(LinearInterp(np.insert(m_end,0,0),
                                                      np.insert(EndOfPrddvds[:,nInd,sInd],0,EndOfPrddvds[0,nInd,sInd])
                                                      )
                                         )
                
    # 3D interpolators
    
    # Consumption interpolator
    cFunc = BilinearInterpOnInterp1D(cInterps, nNrmGrid, ShareGrid)
    # dvdmFxd interpolator
    dvdmFuncCon = MargValueFunc3D(cFunc, CRRA)
    # dvdnFxd interpolator
    dvdnNvrsFunc = BilinearInterpOnInterp1D(dvdnNvrsInterps, nNrmGrid, ShareGrid)
    dvdnFuncCon = MargValueFunc3D(dvdnNvrsFunc, CRRA)
    # dvds interpolator
    # TODO: dvds can be NaN. This is because a way to compute
    # EndOfPrddvds(0,0) has not been implemented yet.
    dvdsFuncCon = BilinearInterpOnInterp1D(dvdsInterps, nNrmGrid, ShareGrid)
    
    # It's useful to the value function on a regular grid
    # interpolator because:
    # - Endogenous grids don't cover the m < n region well.
    # - Faster evaluation when this is called by previous periods.
    
    # Add 0 to the m grid
    mNrmGrid = np.insert(mNrmGrid,0,0)
    mNrm_N = len(mNrmGrid)
    
    # Dimensions might change, so re-create tiled arrays
    mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1,1)), (1,nNrm_N,Share_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (mNrm_N,1,Share_N))
    Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (mNrm_N,nNrm_N,1))
    
    # Consumption, value function and inverse on regular grid
    cNrm_reg = cFunc(mNrm_tiled, nNrm_tiled, Share_tiled)
    aNrm_reg = mNrm_tiled - cNrm_reg
    vCon = u(cNrm_reg) + EndOfPrdvFunc(aNrm_reg, nNrm_tiled, Share_tiled) 
    # TODO: uInv(-Inf) seems to appropriately be yielding 0. Is it
    # necessary to hardcode it?
    vNvrsCon = uInv(vCon)
        
    # vNvrs interpolator. Useful to keep it since its faster to optimize
    # on it in the next step
    vNvrsFuncCon = TrilinearInterp(vNvrsCon, mNrmGrid, nNrmGrid, ShareGrid)
    vFuncCon     = ValueFunc3D(vNvrsFuncCon, CRRA)
    
    solution = RiskyContribConsSolution(
        vFuncCon = vFuncCon,
        cFunc = cFunc,
        dvdmFuncCon = dvdmFuncCon,
        dvdnFuncCon = dvdnFuncCon,
        dvdsFuncCon = dvdsFuncCon
    )
    
    return solution


# Solver for the contribution stage
def solveRiskyContribShaStage(solution_next,ShockDstn,IncomeDstn,RiskyDstn,
                              LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                              BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                              ShareGrid,dGrid,vFuncBool,AdjustPrb,
                              DiscreteShareBool,IndepDstnBool):
    
    # Unpack solution from the next sub-stage
    vFuncCon_next    = solution_next.vFuncCon
    cFunc_next       = solution_next.cFunc
    dvdmFuncCon_next = solution_next.dvdmFuncCon
    dvdnFuncCon_next = solution_next.dvdnFuncCon
    dvdsFuncCon_next = solution_next.dvdsFuncCon
    
    # Define temporary functions for utility and its derivative and inverse
    uPinv = lambda x : utilityP_inv(x, CRRA)
    
    # TODO: implement continuous share case.
    if not DiscreteShareBool:
        
        raise Exception('The case of continuous shares has not been implemented yet')
    
        
    # Create tiled grids    

    # Add 0 to the m and n grids
    nNrmGrid = np.insert(nNrmGrid, 0, 0.0)  
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.insert(mNrmGrid,0,0)
    mNrm_N = len(mNrmGrid)
    Share_N = len(ShareGrid)
    
    mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1,1)), (1,nNrm_N,Share_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (mNrm_N,1,Share_N))
    Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (mNrm_N,nNrm_N,1))
    
    m_idx_tiled = np.tile(np.reshape(np.arange(mNrm_N), (mNrm_N,1)), (1,nNrm_N))
    n_idx_tiled = np.tile(np.reshape(np.arange(nNrm_N), (1,nNrm_N)), (mNrm_N,1))
        
    
    # Evaluate value function to optimize over shares.
    # Do it in inverse space
    vNvrs = vFuncCon_next.func(mNrm_tiled, nNrm_tiled, Share_tiled)
    
    # Find the optimal share at each (m,n).
    optIdx = np.argmax(vNvrs, axis = 2)
    
    # Project grids
    mNrm_tiled = mNrm_tiled[:,:,0]
    nNrm_tiled = nNrm_tiled[:,:,0]
    
    # Compute objects needed for the value function and its derivatives
    vNvrsSha     = vNvrs[m_idx_tiled, n_idx_tiled, optIdx]
    optShare     = ShareGrid[optIdx]
    dvdmNvrsSha  = cFunc_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnSha      = dvdnFuncCon_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnNvrsSha  = uPinv(dvdnSha)
    # Interpolators
    vNvrsFuncSha    = BilinearInterp(vNvrsSha, mNrmGrid, nNrmGrid)
    vFuncSha        = ValueFunc2D(vNvrsFuncSha, CRRA)
    
    # TODO: do share interpolation more smartly taking into account that
    # it's discrete. (current bilinear can and will result in shares
    # outside the discrete grid).
    ShareFunc       = BilinearInterp(optShare, mNrmGrid, nNrmGrid)
    dvdmNvrsFuncSha = BilinearInterp(dvdmNvrsSha, mNrmGrid, nNrmGrid)
    dvdmFuncSha     = MargValueFunc2D(dvdmNvrsFuncSha, CRRA)
    dvdnNvrsFuncSha = BilinearInterp(dvdnNvrsSha, mNrmGrid, nNrmGrid)
    dvdnFuncSha     = MargValueFunc2D(dvdnNvrsFuncSha, CRRA)
    
    solution = RiskyContribShaSolution(
        vFuncShaAdj = vFuncSha,
        ShareFuncAdj = ShareFunc,
        dvdmFuncShaAdj = dvdmFuncSha,
        dvdnFuncShaAdj = dvdnFuncSha,
        
        # The fixed agent does nothing at this stage,
        # so his value functions are the next problem's
        vFuncShaFxd = vFuncCon_next,
        ShareFuncFxd = IdentityFunction(i_dim = 2, n_dims = 3),
        dvdmFuncShaFxd = dvdmFuncCon_next,
        dvdnFuncShaFxd = dvdnFuncCon_next,
        dvdsFuncShaFxd = dvdsFuncCon_next
    )
    
    return solution
    
# Solver for the asset rebalancing stage
def solveRiskyContribRebStage(solution_next,ShockDstn,IncomeDstn,RiskyDstn,
                              LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                              BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                              ShareGrid,dGrid,vFuncBool,AdjustPrb,
                              DiscreteShareBool,IndepDstnBool):
    
    # Extract next stage's solution
    vFuncAdj_next = solution_next.vFuncShaAdj
    dvdmFuncAdj_next = solution_next.dvdmFuncShaAdj
    dvdnFuncAdj_next = solution_next.dvdnFuncShaAdj
    
    vFuncFxd_next = solution_next.vFuncShaFxd
    dvdmFuncFxd_next = solution_next.dvdmFuncShaFxd
    dvdnFuncFxd_next = solution_next.dvdnFuncShaFxd
    dvdsFuncFxd_next = solution_next.dvdsFuncShaFxd
    
    # Define temporary functions for utility and its derivative and inverse
    uPinv = lambda x : utilityP_inv(x, CRRA)
    
    # Create tiled grids    

    # Add 0 to the m and n grids
    nNrmGrid = np.insert(nNrmGrid, 0, 0.0)  
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.insert(mNrmGrid,0,0)
    mNrm_N = len(mNrmGrid)
    d_N = len(dGrid)
    
    # It will be useful to pre-evaluate marginals at every (m,n,d) combination
    
    # Create tiled arrays for every d,m,n option
    d_N = len(dGrid)
    d_tiled    = np.tile(np.reshape(dGrid,    (d_N,1,1)), (1, mNrm_N,nNrm_N))
    mNrm_tiled = np.tile(np.reshape(mNrmGrid, (1,mNrm_N,1)), (d_N,1,nNrm_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,1,nNrm_N)), (d_N,mNrm_N,1))
    
    # Get post-rebalancing assets the m_tilde, n_tilde.
    m_tilde, n_tilde = rebalanceAssets(d_tiled, mNrm_tiled, nNrm_tiled, tau)
    
    # Now the marginals
    dvdm = dvdmFuncAdj_next(m_tilde, n_tilde)
    dvdn = dvdnFuncAdj_next(m_tilde, n_tilde)
    
    # Initialize
    dOpt = np.ones_like(mNrm_tiled[0])*np.nan
    
    # Find the optimal d's
    zeroind = np.where(dGrid == 0.0)[0][0]
    
    for mInd in range(mNrm_N):
        for nInd in range(nNrm_N):
            
            # First check if dvdm(d=-1) > dvdn(d=-1). If so, withdrawing
            # everything is optimal
            if dvdm[0,mInd,nInd] > (1+tau)*dvdn[0,mInd,nInd]:
                dOpt[mInd,nInd] = -1.
            else:
                # Next, check what happens at d == 0. This will determine which
                # of d<0, d==0, d>0 is optimal.
                if (1+tau)*dvdn[zeroind,mInd,nInd] >= dvdm[zeroind,mInd,nInd] and  dvdm[zeroind,mInd,nInd] >= dvdn[zeroind,mInd,nInd]:
                    
                    dOpt[mInd,nInd] = 0.
                    
                else:
                    if dvdm[zeroind,mInd,nInd] > (1+tau)*dvdn[zeroind,mInd,nInd]:
                        # The optimal d is negative
                        dinds = np.arange(0,zeroind+1)
                        FOC = (1+tau)*dvdn[dinds, mInd, nInd] - dvdm[dinds, mInd, nInd]
                        
                    else:
                        # The optimal d is positive
                        dinds = np.arange(zeroind, d_N)
                        FOC = dvdn[dinds, mInd, nInd] - dvdm[dinds, mInd, nInd]
                        
                    # Find first index at which FOC turns negative
                    pos1 = np.argmax(FOC<0)
                    ind1 = dinds[pos1]
                
                    # Linearly approximate where the FOC crosses 0
                    ind0 = ind1 - 1
                    pos0 = pos1 - 1
                    m = (FOC[pos1] - FOC[pos0])/(dGrid[ind1] - dGrid[ind0])
                    dOpt[mInd,nInd] = dGrid[ind0] - FOC[pos0]/m
         
    # Find m_tilde and n_tilde
    mtil_opt, ntil_opt = rebalanceAssets(dOpt, mNrm_tiled[0], nNrm_tiled[0], tau)
            
    # Evaluate inverse end-of-period value function
    vNvrsAdj = vFuncAdj_next.func(mtil_opt, ntil_opt)
    
    # Now the derivatives. These are not straight forward because of corner
    # solutions with partial derivatives that change the limits. The idea then
    # is to evaluate the possible uses of the marginal unit of resources and
    # take the maximum.
    
    # An additional unit of m
    marg_m      = dvdmFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n kept in n
    marg_n      = dvdnFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n withdrawn to m
    marg_n_to_m = marg_m/(1+tau)
    
    # Marginal value is the maximum of the marginals in their possible uses 
    dvdmAdj     = np.maximum(marg_m, marg_n)
    dvdmNvrsAdj = uPinv(dvdmAdj)
    dvdnAdj     = np.maximum(marg_n, marg_n_to_m)
    dvdnNvrsAdj = uPinv(dvdnAdj)
    
    # Interpolators
    # Value
    vNvrsFuncAdj = BilinearInterp(vNvrsAdj, mNrmGrid, nNrmGrid)
    vFuncAdj     = ValueFunc2D(vNvrsFuncAdj, CRRA)
    # Marginals
    dvdmFuncAdj = MargValueFunc2D(BilinearInterp(dvdmNvrsAdj, mNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj = MargValueFunc2D(BilinearInterp(dvdnNvrsAdj, mNrmGrid, nNrmGrid), CRRA)
    # Decison
    DFuncAdj = BilinearInterp(dOpt, mNrmGrid, nNrmGrid)
    
    solution = RiskyContribRebSolution(
            # Rebalancing stage adjusting
            vFuncRebAdj = vFuncAdj,
            DFuncAdj = DFuncAdj,
            dvdmFuncRebAdj = dvdmFuncAdj,
            dvdnFuncRebAdj = dvdnFuncAdj,
            
            # Rebalancing stage fixed (nothing happens, so value functions are
            # the ones from the next stage)
            vFuncRebFxd = vFuncFxd_next,
            DFuncFxd = ConstantFunction(0.0),
            dvdmFuncRebFxd = dvdmFuncFxd_next,
            dvdnFuncRebFxd = dvdnFuncFxd_next,
            dvdsFuncRebFxd = dvdsFuncFxd_next
        )
    
    return solution

    
    
# Make a dictionary to specify a risky asset consumer type
init_risky = init_idiosyncratic_shocks.copy()
init_risky['RiskyAvg']        = 1.08 # Average return of the risky asset
init_risky['RiskyStd']        = 0.20 # Standard deviation of (log) risky returns
init_risky['RiskyCount']      = 5    # Number of integration nodes to use in approximation of risky returns
init_risky['ShareCount']      = 25   # Number of discrete points in the risky share approximation
init_risky['AdjustPrb']       = 1.0  # Probability that the agent can adjust their risky portfolio share each period
init_risky['DiscreteShareBool'] = False # Flag for whether to optimize risky share on a discrete grid only

# Adjust some of the existing parameters in the dictionary
init_risky['aXtraMax']        = 100  # Make the grid of assets go much higher...
init_risky['aXtraCount']      = 200  # ...and include many more gridpoints...
init_risky['aXtraNestFac']    = 1    # ...which aren't so clustered at the bottom
init_risky['BoroCnstArt']     = 0.0  # Artificial borrowing constraint must be turned on
init_risky['CRRA']            = 5.0  # Results are more interesting with higher risk aversion
init_risky['DiscFac']         = 0.90 # And also lower patience

# Make a dictionary for a risky-contribution consumer type

# TODO: these parameters are preliminary and arbitrary!

init_riskyContrib = init_risky.copy()
init_riskyContrib['ShareMax']        = 0.9  # You don't want to put 100% of your wage into pensions.

# Regular grids in m and n
init_riskyContrib['mNrmMin']         = 1e-6
init_riskyContrib['mNrmMax']         = 50
init_riskyContrib['mNrmCount']       = 100
init_riskyContrib['mNrmNestFac']     = 1

init_riskyContrib['nNrmMin']         = 1e-6
init_riskyContrib['nNrmMax']         = 50
init_riskyContrib['nNrmCount']       = 100  
init_riskyContrib['nNrmNestFac']     = 1    

# Number of grid-points for finding the optimal asset rebalance
init_riskyContrib['dCount'] = 20

# Params from the life-cycle agent
init_riskyContrib['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_riskyContrib['PermShkStd'] = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]
init_riskyContrib['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,  0,  0,  0]
init_riskyContrib['AdjustPrb']  = [1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0]
init_riskyContrib['tau']        = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]  # Tax rate on risky asset withdrawals
init_riskyContrib['LivPrb']     = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
init_riskyContrib['T_cycle']    = 10
init_riskyContrib['T_retire']   = 7
init_riskyContrib['T_age']      = 11 # Make sure that old people die at terminal age and don't turn into newborns!


# Reduce dimensions while conding the model up
init_riskyContrib['ShareCount']      = 10
init_riskyContrib['aXtraCount']      = 40
init_riskyContrib['nNrmCount']       = 40  #
init_riskyContrib['mNrmCount']       = 45  #
init_riskyContrib['PermShkCount']    = 3  #
init_riskyContrib['TranShkCount']    = 3 

