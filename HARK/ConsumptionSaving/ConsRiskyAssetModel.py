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
        vFuncCon = None,
        ShareFuncCon = None,
        dvdmFuncCon = None,
        dvdnFuncCon = None,
        
        # Adjusting stage
        vFuncAdj = None,
        DFuncAdj = None,
        dvdmFuncAdj = None,
        dvdnFuncAdj = None,
        dvdsFuncAdj = None,
        
        # Consumption stage
        vFuncFxd = None,
        cFuncFxd = None,
        dvdmFuncFxd = None,
        dvdnFuncFxd = None,
        dvdsFuncFxd = None
        
    ):
        
        # Contribution stage
        if vFuncCon is None:
            vFuncCon = NullFunc()
        if ShareFuncCon is None:
            ShareFuncCon = NullFunc()
        if dvdmFuncCon is None:
            dvdmFuncCon = NullFunc()
        if dvdnFuncCon is None:
            dvdnFuncCon = NullFunc()
        
        # Adjusting stage
        if vFuncAdj is None:
            vFuncAdj = NullFunc()
        if DFuncAdj is None:
            DFuncAdj = NullFunc()
        if dvdmFuncAdj is None:
            dvdmFuncAdj = NullFunc()
        if dvdnFuncAdj is None:
            dvdnFuncAdj = NullFunc()
        if dvdsFuncAdj is None:
            dvdsFuncAdj = NullFunc()
        
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
        self.vFuncCon = vFuncCon
        self.ShareFuncCon = ShareFuncCon
        self.dvdmFuncCon = dvdmFuncCon
        self.dvdnFuncCon = dvdnFuncCon
        
        # Adjusting stage
        self.vFuncAdj = vFuncAdj
        self.DFuncAdj = DFuncAdj
        self.dvdmFuncAdj = dvdmFuncAdj
        self.dvdnFuncAdj = dvdnFuncAdj
        self.dvdsFuncAdj = dvdsFuncAdj
        
        # Consumption stage
        self.vFuncFxd = vFuncFxd
        self.cFuncFxd = cFuncFxd
        self.dvdmFuncFxd = dvdmFuncFxd
        self.dvdnFuncFxd = dvdnFuncFxd
        self.dvdsFuncFxd = dvdsFuncFxd
        
        
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

        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(
            self,
            cycles=cycles,
            verbose=verbose,
            quiet=quiet,
            **kwds
        )
        
        # Set the solver for the portfolio model, and update various constructed attributes
        self.solveOnePeriod = solveConsRiskyContrib
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
        
        # Consumption stage
        
        # Consume all thats available (liquid resources)
        cFuncFxd_term = IdentityFunction(i_dim = 0, n_dims = 3)
        vFuncFxd_term = ValueFunc3D(cFuncFxd_term, CRRA = self.CRRA)
        
        dvdmFuncFxd_term = MargValueFunc3D(cFuncFxd_term, CRRA = self.CRRA)
        dvdnFuncFxd_term = ConstantFunction(0.0)
        dvdsFuncFxd_term = ConstantFunction(0.0)
        
        # Adjusting stage
        
        # Find the withdrawal penalty
        if type(self.tau) is list:
            tau = self.tau[-1]
        else:
            tau = self.tau
            
        # Withdraw everything from the pension fund
        DFuncAdj_term = ConstantFunction(-1.0)
        vFuncAdj_term = ValueFunc3D(lambda m,n,s: m + n/(1+tau), self.CRRA)
        dvdmFuncAdj_term = MargValueFunc3D(lambda m,n,s: m + n/(1+tau), self.CRRA)
        dvdnFuncAdj_term = lambda m,n,s: dvdmFuncAdj_term(m,n,s)/(1+tau)
        dvdsFuncAdj_term = ConstantFunction(0.0)
        
        # Contribution stage: share is irrelevant, so functions are those from
        # the rebalancing stage.
        
        # Take the lowest share, arbitrarily
        aux_s = self.ShareGrid[0]
        
        vFuncCon_term = lambda m,n: vFuncAdj_term(m,n,aux_s*np.ones_like(m))
        ShareFuncCon_term = ConstantFunction(aux_s)
        dvdmFuncCon_term = lambda m,n: dvdmFuncAdj_term(m,n,aux_s*np.ones_like(m))
        dvdnFuncCon_term = lambda m,n: dvdnFuncAdj_term(m,n,aux_s*np.ones_like(m))
        
        # Construct the terminal period solution
        self.solution_terminal = RiskyContribSolution(
            # Contribution stage
            vFuncCon = vFuncCon_term,
            ShareFuncCon = ShareFuncCon_term,
            dvdmFuncCon = dvdmFuncCon_term,
            dvdnFuncCon = dvdnFuncCon_term,
            
            # Adjusting stage
            vFuncAdj = vFuncAdj_term,
            DFuncAdj = DFuncAdj_term,
            dvdmFuncAdj = dvdmFuncAdj_term,
            dvdnFuncAdj = dvdnFuncAdj_term,
            dvdsFuncAdj = dvdsFuncAdj_term,
            
            # Consumption stage
            vFuncFxd = vFuncFxd_term,
            cFuncFxd = cFuncFxd_term,
            dvdmFuncFxd = dvdmFuncFxd_term,
            dvdnFuncFxd = dvdnFuncFxd_term,
            dvdsFuncFxd = dvdsFuncFxd_term
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
    
def rebalanceAssets(d,a,n,tau):
        
    if d >= 0:
        a_til = a*(1-d)
        n_til = n + a*d
    else:
        a_til = a - d*n/(1 + tau)
        n_til = n*(1+d)
    
    return (a_til, n_til)

def rebalanceFobj(d,a,n,v3,tau):
        
    a_til, n_til = rebalanceAssets(d,a,n,tau)
    return v3(a_til, n_til)
    
def findOptimalRebalance(a,n,v3,tau):
    
    if (a == 0 and n == 0):
        dopt = 0
    else:
        fobj = lambda d: -1.*rebalanceFobj(d,a,n,v3,tau)
        # For each case, we optimize numerically and compare with the extremes.
        if a > 0 and n > 0:
            # Optimize contributing and withdrawing separately
            opt_c = minimize_scalar(fobj, bounds=(0, 1), method='bounded')
            opt_w = minimize_scalar(fobj, bounds=(-1, 0), method='bounded')
            
            ds = np.array([opt_c.x,opt_w.x,-1,0,1])
            fs = np.array([opt_c.fun,opt_w.fun,fobj(-1),fobj(0),fobj(1)])
        elif a > 0:
            opt = minimize_scalar(fobj, bounds=(0, 1), method='bounded')
            ds = np.array([opt.x,0,1])
            fs = np.array([opt.fun,fobj(0),fobj(1)])
        else:
            opt = minimize_scalar(fobj, bounds=(-1, 0), method='bounded')
            ds = np.array([opt.x,-1,0])
            fs = np.array([opt.fun,fobj(-1),fobj(0)])
        
        # Pick the best candidate
        dopt = ds[np.argmin(fs)]
                
    a_til, n_til = rebalanceAssets(dopt,a,n,tau)
    return dopt, a_til, n_til
        
    
                
# Define a non-object-oriented one period solver
def solveConsRiskyContrib(solution_next,ShockDstn,IncomeDstn,RiskyDstn,
                          LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                          BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                          ShareGrid,vFuncBool,AdjustPrb,
                          DiscreteShareBool,IndepDstnBool):
    '''
    Solve the one period problem for a portfolio-choice consumer.
    
    Parameters
    ----------
    solution_next : RiskyContribSolution
        Solution to next period's problem.
    ShockDstn : [np.array]
        List with four arrays: discrete probabilities, permanent income shocks,
        transitory income shocks, and risky returns.  This is only used if the
        input IndepDstnBool is False, indicating that income and return distributions
        can't be assumed to be independent.
    IncomeDstn : [np.array]
        List with three arrays: discrete probabilities, permanent income shocks,
        and transitory income shocks.  This is only used if the input IndepDsntBool
        is True, indicating that income and return distributions are independent.
    RiskyDstn : [np.array]
        List with two arrays: discrete probabilities and risky asset returns. This
        is only used if the input IndepDstnBool is True, indicating that income
        and return distributions are independent.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  In this model, it is *required* to be zero.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    nNrmGrid: np.array
        Array of iliquid risky asset balances.
    mNrmGrid: np.array
        Exogenous start-of-period liquid assets grid
    ShareGrid : np.array
        Array of risky portfolio shares on which to define the interpolation
        of the consumption function when Share is fixed.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    AdjustPrb : float
        Probability that the agent will be able to update his contribution share.
    DiscreteShareBool : bool
        Indicator for whether income contribution share should be optimized on the
        continuous [0,1] interval using the FOC (False), or instead only selected
        from the discrete set of values in ShareGrid (True).  If True, then
        vFuncBool must also be True.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.

    Returns
    -------
    solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
    '''
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
    vFuncCon_next     = solution_next.vFuncCon
    ShareFuncCon_next = solution_next.ShareFuncCon
    dvdmFuncCon_next  = solution_next.dvdmFuncCon
    dvdnFuncCon_next  = solution_next.dvdnFuncCon
    
    vFuncAdj_next     = solution_next.vFuncAdj
    DFuncAdj_next     = solution_next.DFuncAdj
    dvdmFuncAdj_next  = solution_next.dvdmFuncAdj
    dvdnFuncAdj_next  = solution_next.dvdnFuncAdj
    dvdsFuncAdj_next  = solution_next.dvdsFuncAdj
    
    vFuncFxd_next     = solution_next.vFuncFxd
    cFuncFxd_next     = solution_next.cFuncFxd
    dvdmFuncFxd_next  = solution_next.dvdmFuncFxd
    dvdnFuncFxd_next  = solution_next.dvdnFuncFxd
    dvdsFuncFxd_next  = solution_next.dvdsFuncFxd
    
    # Major method fork: (in)dependent risky asset return and income distributions
    if IndepDstnBool: # If the distributions ARE independent...
        
        # I deal with the non-independent case for now. So simply construct a
        # joint distribution.
        # TODO: speedup tricks
        ShockDstn = combineIndepDstns(IncomeDstn, RiskyDstn)
    
    # Unpack the shock distribution
    ShockPrbs_next = ShockDstn.pmf
    PermShks_next  = ShockDstn.X[0]
    TranShks_next  = ShockDstn.X[1]
    Risky_next     = ShockDstn.X[2]
    
    # STEP ONE
    # Find end-of-period (continuation) value function and its derivatives.
    
    # TODO: deal with the possibly-0-income case. Characterize situations in
    # which the agent will stay at positive savings.
    # He might still have a=0 if n>0 and the probability of adjusting is 1.
    zero_bound = (np.min(TranShks_next) == 0.) # Flag for whether the natural borrowing constraint is zero
    if zero_bound:
        aNrmGrid = aXtraGrid
    else:
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0) # Add an asset point at exactly zero
        nNrmGrid = np.insert(nNrmGrid, 0, 0.0)     
       
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
    
    # TODO (remove): a sanity check
    if not vFuncCon_next(0.02,0.6) >= vFuncCon_next(0.02,0.05):
        print('VFUNC NOT INCREASING!!!')
    
    # Always compute the adjusting version
    vCon_next    = vFuncCon_next(mNrm_next, nNrm_next)
    dvdmCon_next = dvdmFuncCon_next(mNrm_next,nNrm_next)
    dvdnCon_next = dvdnFuncCon_next(mNrm_next,nNrm_next)
    dvdsCon_next = np.zeros_like(mNrm_next) # No marginal value of Share if it's a free choice!
    
    # We are interested in marginal values before the realization of the
    # adjustment random variable. Compute those objects
    if AdjustPrb < 1.:
        
        # "Fixed" counterparts
        vFxd_next    = vFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdmFxd_next = dvdmFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdnFxd_next = dvdnFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdsFxd_next = dvdsFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        
        # Expected values with respect to adjustment r.v.
        v_next    = AdjustPrb*vCon_next    + (1.-AdjustPrb)*vFxd_next
        dvdm_next = AdjustPrb*dvdmCon_next + (1.-AdjustPrb)*dvdmFxd_next
        dvdn_next = AdjustPrb*dvdnCon_next + (1.-AdjustPrb)*dvdnFxd_next
        dvds_next = AdjustPrb*dvdsCon_next + (1.-AdjustPrb)*dvdsFxd_next
        
    else: # Don't bother evaluating if there's no chance that contribution share is fixed
        v_next    = vCon_next
        dvdm_next = dvdmCon_next
        dvdn_next = dvdnCon_next
        dvds_next = dvdsCon_next
        
    # Calculate end-of-period marginal value of both assets by taking expectations
    temp_fac_A = uP(PermShks_tiled*PermGroFac) # Will use this in a couple places
    EndOfPrddvda = DiscFac*Rfree*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*dvdm_next, axis=3)
    EndOfPrddvdn = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*Risky_tiled*dvdn_next, axis=3)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvda)
        
    # Calculate end-of-period value by taking expectations
    temp_fac_B = (PermShks_tiled*PermGroFac)**(1.-CRRA) # Will use this below
    EndOfPrdv = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_B*v_next, axis=3)
    EndOfPrdvNvrs = uInv(EndOfPrdv)
    
    # Construct an interpolator for EndOfPrdV. It will be useful later.
    EndOfPrdvFunc = ValueFunc3D(TrilinearInterp(EndOfPrdvNvrs, aNrmGrid,
                                                nNrmGrid, ShareGrid),
                                CRRA)
    
    # Compute post-consumption marginal value of contribution share,
    # conditional on shocks
    EndOfPrddvds_cond_undisc = temp_fac_B*( TranShks_tiled*(dvdn_next - dvdm_next) + dvds_next)
    # Discount and integrate over shocks
    EndOfPrddvds = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*EndOfPrddvds_cond_undisc, axis=3)
    
    # STEP TWO:
    # Solve the consumption problem and create interpolators for cFxd, vFxd,
    # and its derivatives.
    
    # Recast a, n, and s now that the shock dimension has been integrated over
    aNrm_tiled  = aNrm_tiled[:,:,:,0]
    nNrm_tiled  = nNrm_tiled[:,:,:,0]
    Share_tiled = Share_tiled[:,:,:,0]
    
    # Apply EGM over liquid resources at every (n,s) to find consumption.
    cFxd = EndOfPrddvdaNvrs
    mNrm_endog = aNrm_tiled + cFxd
    
    # Behaviour of cFxd, vFxd, and the derivatives at low values of (m,n)
    # depends on what combinations of (a,n) can force zero consumption next
    # period. For now I deal with the no-unemployment case, where this is
    # impossible.
    if zero_bound:
        
       raise Exception('The case where unemployment is possible has not been implemented yet')
       
    else:
       
        # We know that:
        # -The lowest gridpoints of both a and n are 0.
        # -Consumption at m < m0 is m.
        # -dvdnFxd at (m,n) for m < m0(n) is dvdnFxd(m0,n)
    
        # Create consumption interpolator
        cInterps = [[LinearInterp(np.insert(mNrm_endog[:,nInd,sInd],0,0),
                                  np.insert(cFxd[:,nInd,sInd],0,0))
                     for sInd in range(Share_N)]
                    for nInd in range(nNrm_N)]
        cFuncFxd = BilinearInterpOnInterp1D(cInterps, nNrmGrid, ShareGrid)
        
        # Create dvdmFxd interpolator
        dvdmFuncFxd = MargValueFunc3D(cFuncFxd, CRRA)
        
        # Create dvdnFxdInterpolator
        dvdnFxdInterps = [[LinearInterp(np.insert(mNrm_endog[:,nInd,sInd],0,0),
                                        np.insert(EndOfPrddvdn[:,nInd,sInd],0,EndOfPrddvdn[0,nInd,sInd]))
                           for sInd in range(Share_N)]
                          for nInd in range(nNrm_N)]
        dvdnFuncFxd = BilinearInterpOnInterp1D(dvdnFxdInterps, nNrmGrid, ShareGrid)
        
        # It's useful to have the value function with a regular-grid
        # interpolator because:
        # - Endogenous grids don't cover the m < n well.
        # - This value function will be optimized over, so evaluated many times
        #   and regular grid interpolators are faster.
        
        # Add 0 to the m grid
        mNrmGrid = np.insert(mNrmGrid,0,0)
        mNrm_N = len(mNrmGrid)
        
        # Dimensions might change, so re-create tiled arrays
        mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1,1)), (1,nNrm_N,Share_N))
        nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (mNrm_N,1,Share_N))
        Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (mNrm_N,nNrm_N,1))
        
        # Consumption, value function and inverse on regular grid
        cNrm_reg = cFuncFxd(mNrm_tiled, nNrm_tiled, Share_tiled)
        aNrm_reg = mNrm_tiled - cNrm_reg
        vFxd = u(cNrm_reg) + EndOfPrdvFunc(aNrm_reg, nNrm_tiled, Share_tiled) 
        # TODO: uInv(-Inf) seems to appropriately be yielding 0. Is it
        # necessary to hardcode it?
        vFxdNvrs = uInv(vFxd)
        
        # vNvrs interpolator. Useful to keep it since its faster to optimize
        # on it in the next step
        vFxdNvrsFunc = TrilinearInterp(vFxdNvrs, mNrmGrid, nNrmGrid, ShareGrid)
        vFxdFunc     = ValueFunc3D(vFxdNvrsFunc, CRRA)
    
    # STEP THREE:
    # Adjusting stage.
    d = 'Here.' # TODO
    
    # Construct an interpolator for the end-of-period marginal value of
    # iliquid assets and contribution shares, conditional on the contribution
    # share. This will be used by the FXD adent
    EndOfPrddvdnCondShareFunc = TrilinearInterp(EndOfPrddvdn,
                                                aNrmGrid, nNrmGrid, ShareGrid)
    EndOfPrddvdsCondShareFunc = TrilinearInterp(EndOfPrddvds,
                                                aNrmGrid, nNrmGrid, ShareGrid)
    
    # SECOND STEP: find the value function for the third (contribution) stage
    # and its derivatives for the agent who can adjust.
    
    if DiscreteShareBool: # Optimization of Share on the discrete set ShareGrid
        opt_idx = np.argmax(EndOfPrdv, axis=2)
        Share_opt = ShareGrid[opt_idx] # Best portfolio share is one with highest value
        
        # Create indices to extract end of period value at (n,m) optimized
        # through the share, which is vAdj3
        a_idx_tiled = np.tile(np.reshape(np.arange(aNrm_N), (aNrm_N,1)), (1,nNrm_N))
        n_idx_tiled = np.tile(np.reshape(np.arange(nNrm_N), (1,nNrm_N)), (aNrm_N,1))
        
        # Extract vAdj3 and its derivatives in their inverse form
        vAdj3Nvrs = EndOfPrdvNvrs[a_idx_tiled,n_idx_tiled,opt_idx]
        dvdaAdj3Nvrs = EndOfPrddvdaNvrs[a_idx_tiled,n_idx_tiled,opt_idx]
        dvdnAdj3 = EndOfPrddvdn[a_idx_tiled,n_idx_tiled,opt_idx]
        
    else: # Optimization of Share on continuous interval [0,1]
        # TODO?    
        pass
    
    # Construct interpolator for the optimal share
    ShareFuncAdj = BilinearInterp(Share_opt, aNrmGrid, nNrmGrid)
    
    # Construct interpolators for v3Adj and its derivatives
    vFuncAdj3    = ValueFunc2D(BilinearInterp(vAdj3Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdaFuncAdj3 = MargValueFunc2D(BilinearInterp(dvdaAdj3Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj3 = BilinearInterp(dvdnAdj3, aNrmGrid, nNrmGrid)
    
    # THIRD STEP: decision, value function, and derivatives for the rebalancing
    # stage.
    
    # Generate (a,N) combinations
    aNrm_tiled = np.tile(np.reshape(aNrmGrid, (aNrm_N,1)), (1,nNrm_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N)), (aNrm_N,1))
    
    # Find optimal d for every combination
    optRebalance = list(map(lambda x: findOptimalRebalance(x[0],x[1],vFuncAdj3,tau),
                            zip(aNrm_tiled.flatten(),nNrm_tiled.flatten())
                            )
                        )
    optRebalance = np.array(optRebalance)
    
    # Format rebalancing share and post-rebalancing assets as tiled arrays
    D_tiled      = np.reshape(optRebalance[:,0], (aNrm_N, nNrm_N))
    aTilde_tiled = np.reshape(optRebalance[:,1], (aNrm_N, nNrm_N))
    nTilde_tiled = np.reshape(optRebalance[:,2], (aNrm_N, nNrm_N))
    
    # Construct the value function
    vAdj2Nvrs = uInv(vFuncAdj3(aTilde_tiled, nTilde_tiled))
    vFuncAdj2    = ValueFunc2D(BilinearInterp(vAdj2Nvrs, aNrmGrid, nNrmGrid), CRRA)
    
    # Now the derivatives. These are not straight forward because of corner
    # solutions with partial derivatives that change the limits. The idea then
    # is to evaluate the possible uses of the marginal unit of resources and
    # take the maximum.
    
    # An additional unit of a
    marg_a   = dvdaFuncAdj3(aTilde_tiled, nTilde_tiled)
    # An additional unit of n kept in n
    marg_n    = dvdnFuncAdj3(aTilde_tiled, nTilde_tiled)
    # An additional unit of n withdrawn to a
    marg_n_to_a = marg_a/(1+tau)
    
    dvdaAdj2 = np.maximum(marg_a, marg_n)
    dvdaAdj2Nvrs = uPinv(dvdaAdj2)
    dvdnAdj2 = np.maximum(marg_n, marg_n_to_a)
    
    # Interpolators
    dvdaFuncAdj2 = MargValueFunc2D(BilinearInterp(dvdaAdj2Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj2 = BilinearInterp(dvdnAdj2, aNrmGrid, nNrmGrid)
    
    # Construct the rebalancing policy function
    DFuncAdj = BilinearInterp(D_tiled, aNrmGrid, nNrmGrid)
    
    # STEP FOUR: EGM inversion to get consumption at endogenous grid, solving
    # the first stage.
    
    # Invert consumption candidates and market resources from the marginal 
    # post-consumption value of liquid assets on the exogenous grid
    cAdjEndog = uPinv(dvdaFuncAdj2(aNrm_tiled, nNrm_tiled))
    mNrmEndog_tiled = aNrm_tiled + cAdjEndog
    
    # Evaluate value function at candidate points (needed for envelope)
    vAdjEndog = u(cAdjEndog) + vFuncAdj2(aNrm_tiled, nNrm_tiled)  
    # Transform
    vTAdjEndog = uInv(vAdjEndog)
    
    # Construct 2D interpolators for v, c, and marginals form
    # 1D interpolators at every value of n
    
    # Consumption
    cAdjInterps = [LinearInterp(np.insert(mNrmEndog_tiled[:,j],0,0),
                                np.insert(cAdjEndog[:,j],0,0))
                   for j in range(nNrm_N)]
    cFuncAdj = LinearInterpOnInterp1D(cAdjInterps, nNrmGrid)
    
    # Value
    vTAdjInterps = [LinearInterp(np.insert(mNrmEndog_tiled[:,j],0,0),
                                 np.insert(vTAdjEndog[:,j],0,0))
                   for j in range(nNrm_N)]
    vFuncAdj = ValueFunc2D(LinearInterpOnInterp1D(vTAdjInterps, nNrmGrid),
                           CRRA)
    
    # Marginal re: liquid assets
    dvdmFuncAdj = MargValueFunc2D(cFuncAdj, CRRA)
    
    # Marginal re: iliquid assets
    dvdnAdj = dvdnFuncAdj2(aNrm_tiled, nNrm_tiled)
    if zero_bound:
        # TODO: find a better solution than lower extrap!
        dvdnAdjInterps = [LinearInterp(mNrmEndog_tiled[:,j], dvdnAdj[:,j],
                                       lower_extrap=True,
                                       intercept_limit=0,slope_limit=0)
                          for j in range(nNrm_N)]
    else:
        # We know that dvdn is constant below  a=0.
        dvdnAdjInterps = [LinearInterp(np.insert(mNrmEndog_tiled[:,j],0,0),
                                       np.insert(dvdnAdj[:,j],0,dvdnAdj[0,j]),
                                       intercept_limit=0,slope_limit=0)
                          for j in range(nNrm_N)]
    dvdnFuncAdj = LinearInterpOnInterp1D(dvdnAdjInterps, nNrmGrid)
    dvdnFuncAdj(3,5)
    
    # Finally, create the (trivial) rebalancing and share functions for the
    # nonadjusting agent
    DFuncFxd = ConstantFunction(0)
    ShareFuncFxd = IdentityFunction(i_dim = 2, n_dims = 3)
    
    # STEP FIVE: solve the fixed-portfolio agent.
    aNrm_tiled  = np.tile(np.reshape(aNrmGrid, (aNrm_N,1,1)), (1,nNrm_N,Share_N))
    nNrm_tiled  = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (aNrm_N,1,Share_N))
    Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (aNrm_N,nNrm_N,1))
    
    # EGM inversion
    cFxdEndog = EndOfPrddvdaNvrs
    mNrmEndog_tiled = cFxdEndog + aNrm_tiled
    # Candidate inverse value, needed for envelopes
    vFxdEndog = u(cFxdEndog) + EndOfPrdv
    vTFxdEndog = uInv(vFxdEndog)
    
    # Create tridimensional interpolators from 2D lists of 1d interpolators
    # over m
    
    # Consumption
    cFxdInterps = [[LinearInterp(np.insert(mNrmEndog_tiled[:,nInd,sInd],0,0),
                              np.insert(cFxdEndog[:,nInd,sInd],0,0))
                    for sInd in range(Share_N)]
                   for nInd in range(nNrm_N)]
    cFuncFxd = BilinearInterpOnInterp1D(cFxdInterps, nNrmGrid, ShareGrid)
    
    # Value
    vTFxdInterps = [[LinearInterp(np.insert(mNrmEndog_tiled[:,nInd,sInd],0,0),
                                  np.insert(vTFxdEndog[:,nInd,sInd],0,0))
                     for sInd in range(Share_N)]
                    for nInd in range(nNrm_N)]
    
    vFuncFxd = ValueFunc3D(BilinearInterpOnInterp1D(vTFxdInterps,
                                                    nNrmGrid, ShareGrid),
                           CRRA)
        
    # Derivatives:
    
    # Liquid
    dvdmFuncFxd = MargValueFunc3D(cFuncFxd, CRRA)
    # Iliquid
    dvdnFxd  = EndOfPrddvdnCondShareFunc(aNrm_tiled, nNrm_tiled, Share_tiled)
    if zero_bound:
        # TODO: find a better option than lower extrapolation
        dvdnFxdInterps = [[LinearInterp(mNrmEndog_tiled[:,nInd,sInd],
                                        dvdnFxd[:,nInd,sInd], lower_extrap=True,
                                        intercept_limit=0,slope_limit=0)
                           for sInd in range(Share_N)]
                          for nInd in range(nNrm_N)]
    else:
        # dvdn is constant below a=0
        dvdnFxdInterps = [[LinearInterp(np.insert(mNrmEndog_tiled[:,nInd,sInd],0,0),
                                        np.insert(dvdnFxd[:,nInd,sInd],0,dvdnFxd[0,nInd,sInd]),
                                        intercept_limit=0,slope_limit=0)
                           for sInd in range(Share_N)]
                          for nInd in range(nNrm_N)]
    
    dvdnFuncFxd = BilinearInterpOnInterp1D(dvdnFxdInterps, nNrmGrid, ShareGrid)
    
    # Share
    dvdsFxd = EndOfPrddvdsCondShareFunc(aNrm_tiled, nNrm_tiled, Share_tiled)
    
    dvdsFxdInterps = [[LinearInterp(mNrmEndog_tiled[:,nInd,sInd],
                                    dvdsFxd[:,nInd,sInd])
                       for sInd in range(Share_N)]
                      for nInd in range(nNrm_N)]
    
    dvdsFuncFxd = BilinearInterpOnInterp1D(dvdsFxdInterps, nNrmGrid, ShareGrid)
    
    # Construct solution
    sol = RiskyContribSolution(
        cFuncAdj = cFuncAdj,
        ShareFuncAdj = ShareFuncAdj,
        DFuncAdj = DFuncAdj,
        vFuncAdj = vFuncAdj,
        dvdmFuncAdj = dvdmFuncAdj,
        dvdnFuncAdj = dvdnFuncAdj,
        vFuncAdj2 = vFuncAdj2,
        dvdaFuncAdj2 = dvdaFuncAdj2,
        dvdnFuncAdj2 = dvdnFuncAdj2,
        vFuncAdj3 = vFuncAdj3,
        dvdaFuncAdj3 = dvdaFuncAdj3,
        dvdnFuncAdj3 = dvdnFuncAdj3,
        cFuncFxd = cFuncFxd,
        ShareFuncFxd = ShareFuncFxd,
        DFuncFxd = DFuncFxd,
        vFuncFxd = vFuncFxd,
        dvdmFuncFxd = dvdmFuncFxd,
        dvdnFuncFxd = dvdnFuncFxd,
        dvdsFuncFxd = dvdsFuncFxd
    )
    
    return sol
    
    
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
init_riskyContrib['mNrmMax']         = 10
init_riskyContrib['mNrmCount']       = 100
init_riskyContrib['mNrmNestFac']     = 1

init_riskyContrib['nNrmMin']         = 1e-6
init_riskyContrib['nNrmMax']         = 10
init_riskyContrib['nNrmCount']       = 100  
init_riskyContrib['nNrmNestFac']     = 1    

# Params from the life-cycle agent
init_riskyContrib['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_riskyContrib['PermShkStd'] = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]
init_riskyContrib['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,  0,  0,  0]
init_riskyContrib['AdjustPrb']  = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
init_riskyContrib['tau']        = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]  # Tax rate on risky asset withdrawals
init_riskyContrib['LivPrb']     = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
init_riskyContrib['T_cycle']    = 10
init_riskyContrib['T_retire']   = 7
init_riskyContrib['T_age']      = 11 # Make sure that old people die at terminal age and don't turn into newborns!


# Reduce dimensions while conding the model up
init_riskyContrib['ShareCount']      = 10
init_riskyContrib['aXtraCount']      = 20
init_riskyContrib['nNrmCount']       = 20  #
init_riskyContrib['mNrmCount']       = 25  #
init_riskyContrib['PermShkCount']    = 3  #
init_riskyContrib['TranShkCount']    = 3 

