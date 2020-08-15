'''
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
'''
import numpy as np
from scipy.optimize import minimize_scalar
from itertools import product
from copy import deepcopy
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
        BilinearInterp,         # 2D interpolator
        ConstantFunction,       # Interpolator-like class that returns constant value
        IdentityFunction        # Interpolator-like class that returns one of its arguments
)

from HARK.utilities import makeGridExpMult

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
    time_inv_ = time_inv_ + ['AdjustPrb', 'DiscreteShareBool']

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
        IndShockConsumerType.update(self)
        self.updateRiskyDstn()
        self.updateShockDstn()
        self.updateShareGrid()

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
        cFuncAdj = None,
        ShareFuncAdj = None,
        DFuncAdj = None,
        vFuncAdj = None,
        dvdmFuncAdj = None,
        dvdnFuncAdj = None,
        vFuncAdj2 = None,
        dvdaFuncAdj2 = None,
        dvdnFuncAdj2 = None,
        vFuncAdj3 = None,
        dvdaFuncAdj3 = None,
        dvdnFuncAdj3 = None,
        cFuncFxd = None,
        ShareFuncFxd = None,
        DFuncFxd = None,
        vFuncFxd = None,
        dvdmFuncFxd = None,
        dvdnFuncFxd = None,
        dvdsFuncFxd = None
    ):

        # Change any missing function inputs to NullFunc
        if cFuncAdj is None:
            cFuncAdj = NullFunc()
        if ShareFuncAdj is None:
            ShareFuncAdj = NullFunc()
        if DFuncAdj is None:
            DFuncAdj = NullFunc()
        if vFuncAdj is None:
            vFuncAdj = NullFunc()
        if dvdmFuncAdj is None:
            dvdmFuncAdj = NullFunc()
        if dvdnFuncAdj is None:
            dvdnFuncAdj = NullFunc()
        if vFuncAdj2 is None:
            vFuncAdj2 = NullFunc()
        if dvdaFuncAdj2 is None:
            dvdaFuncAdj2 = NullFunc()
        if dvdnFuncAdj2 is None:
            dvdnFuncAdj2 = NullFunc()
        if vFuncAdj3 is None:
            vFuncAdj3 = NullFunc()
        if dvdaFuncAdj3 is None:
            dvdaFuncAdj3 = NullFunc()
        if dvdnFuncAdj3 is None:
            dvdnFuncAdj3 = NullFunc()
        if cFuncFxd is None:
            cFuncFxd = NullFunc()
        if ShareFuncFxd is None:
            ShareFuncFxd = NullFunc()
        if DFuncFxd is None:
            DFuncFxd = NullFunc()
        if vFuncFxd is None:
            vFuncFxd = NullFunc()
        if dvdmFuncFxd is None:
            dvdmFuncFxd = NullFunc()
        if dvdnFuncFxd is None:
            dvdnFuncFxd = NullFunc()
        if dvdsFuncFxd is None:
            dvdsFuncFxd = NullFunc()
        
        # Set attributes of self
        self.cFuncAdj = cFuncAdj
        self.ShareFuncAdj = ShareFuncAdj
        self.DFuncAdj = DFuncAdj
        self.vFuncAdj = vFuncAdj
        self.dvdmFuncAdj = dvdmFuncAdj
        self.dvdnFuncAdj = dvdnFuncAdj
        self.vFuncAdj2 = vFuncAdj2
        self.dvdaFuncAdj2 = dvdaFuncAdj2
        self.dvdnFuncAdj2 = dvdnFuncAdj2
        self.vFuncAdj3 = vFuncAdj3
        self.dvdaFuncAdj3 = dvdaFuncAdj3
        self.dvdnFuncAdj3 = dvdnFuncAdj3
        self.cFuncFxd = cFuncFxd
        self.ShareFuncFxd = ShareFuncFxd
        self.DFuncFxd = DFuncFxd
        self.vFuncFxd = vFuncFxd
        self.dvdmFuncFxd = dvdmFuncFxd
        self.dvdnFuncFxd = dvdnFuncFxd
        self.dvdsFuncFxd = dvdsFuncFxd
        
        
class RiskyContribConsumerType(RiskyAssetConsumerType):
    """
    TODO: model description
    """
    poststate_vars_ = ['aNrmNow', 'nNrmNow', 'pLvlNow', 'ShareNow', 'AdjustNow']
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + ['AdjustPrb', 'DiscreteShareBool','tau']

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
        IndShockConsumerType.update(self)
        self.updateNGrid()
        self.updateRiskyDstn()
        self.updateShockDstn()
        self.updateShareGrid()
        
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
        
        # Consumption functions: consume all liquid resources
        cFuncAdj_term = IdentityFunction(i_dim=0, n_dims=2)
        cFuncFxd_term = IdentityFunction(i_dim=0, n_dims=3)
        
        # Share functions: irrelevant, set to 0 if possible to adjust
        ShareFuncAdj_term = ConstantFunction(0.)
        ShareFuncFxd_term = IdentityFunction(i_dim=2, n_dims=3),
        
        # Adjustment function: irrelevant, set to 0 if possible to adjust
        DFuncAdj_term = ConstantFunction(0.)
        DFuncFxd_term = ConstantFunction(0.)
        
        # Value function if possible to adjust and derivatives
        vFuncAdj_term = lambda m, n: utility(cFuncAdj_term(m,n), self.CRRA)
        dvdmFuncAdj_term = lambda m, n: utilityP(cFuncAdj_term(m,n), self.CRRA)
        dvdnFuncAdj_term = ConstantFunction(0.)
        
        # Post-consumption value function and derivatives of the adjusting
        # agent.
        # No utility flows after consumption.
        vFuncAdj2_term = ConstantFunction(0.),
        dvdaFuncAdj2_term = ConstantFunction(0.),
        dvdnFuncAdj2_term = ConstantFunction(0.)
        
        # Post-rebalancing value function and derivatives of the adjusting
        # agent.
        # No utility flows after consumption.
        vFuncAdj3_term = ConstantFunction(0.),
        dvdaFuncAdj3_term = ConstantFunction(0.),
        dvdnFuncAdj3_term = ConstantFunction(0.)
        
        # Value function if not possible to adjust, and derivatives
        vFuncFxd_term = lambda m, n, s: utility(cFuncFxd_term(m,n,s), self.CRRA)
        dvdmFuncFxd_term = lambda m, n, s: utilityP(cFuncFxd_term(m,n,s), self.CRRA)
        dvdnFuncFxd_term = ConstantFunction(0.)
        dvdsFuncFxd_term = ConstantFunction(0.)
        
        # Construct the terminal period solution
        self.solution_terminal = RiskyContribSolution(
            cFuncAdj = cFuncAdj_term,
            ShareFuncAdj = ShareFuncAdj_term,
            DFuncAdj = DFuncAdj_term,
            vFuncAdj = vFuncAdj_term,
            dvdmFuncAdj = dvdmFuncAdj_term,
            dvdnFuncAdj = dvdnFuncAdj_term,
            vFuncAdj2 = vFuncAdj2_term,
            dvdaFuncAdj2 = dvdaFuncAdj2_term,
            dvdnFuncAdj2 = dvdnFuncAdj2_term,
            vFuncAdj3 = vFuncAdj3_term,
            dvdaFuncAdj3 = dvdaFuncAdj3_term,
            dvdnFuncAdj3 = dvdnFuncAdj3_term,
            cFuncFxd = cFuncFxd_term,
            ShareFuncFxd = ShareFuncFxd_term,
            DFuncFxd = DFuncFxd_term,
            vFuncFxd = vFuncFxd_term,
            dvdmFuncFxd = dvdmFuncFxd_term,
            dvdnFuncFxd = dvdnFuncFxd_term,
            dvdsFuncFxd = dvdsFuncFxd_term
        )
        
        
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
        self.ShareGrid = np.linspace(0.,1.,self.ShareCount)
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
    
    lb, ub = -1., 1.
    if a == 0.:
        ub = 0.0
    if n == 0.:
        lb = 0.0
    
    if (lb == 0 and ub == 0):
        dopt = 0
    else:
        fobj = lambda d: -1.*rebalanceFobj(d,a,n,v3,tau)
        dopt = minimize_scalar(fobj, bounds=(lb, ub), method='bounded').x
    
    a_til, n_til = rebalanceAssets(dopt,a,n,tau)
    
    return dopt, a_til, n_til
        
    
                
# Define a non-object-oriented one period solver
def solveConsRiskyContrib(solution_next,ShockDstn,IncomeDstn,RiskyDstn,
                          LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                          BoroCnstArt,aXtraGrid,nNrmGrid,ShareGrid,vFuncBool,AdjustPrb,
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
    uInvP = lambda x : utility_invP(x, CRRA)
        
    # Unpack next period's solution
    cFuncAdj_next     = solution_next.cFuncAdj
    ShareFuncAdj_next = solution_next.ShareFuncAdj
    DFuncAdj_next     = solution_next.DFuncAdj
    vFuncAdj_next     = solution_next.vFuncAdj
    dvdmFuncAdj_next  = solution_next.dvdmFuncAdj
    dvdnFuncAdj_next  = solution_next.dvdnFuncAdj
    vFuncAdj2_next    = solution_next.vFuncAdj2
    dvdaFuncAdj2_next = solution_next.dvdaFuncAdj2
    dvdnFuncAdj2_next = solution_next.dvdnFuncAdj2
    vFuncAdj3_next    = solution_next.vFuncAdj3
    dvdaFuncAdj3_next = solution_next.dvdaFuncAdj3
    dvdnFuncAdj3_next = solution_next.dvdnFuncAdj3
    cFuncFxd_next     = solution_next.cFuncFxd
    ShareFuncFxd_next = solution_next.ShareFuncFxd
    DFuncFxd_next     = solution_next.DFuncFxd
    vFuncFxd_next     = solution_next.vFuncFxd
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
    
    zero_bound = (np.min(TranShks_next) == 0.) # Flag for whether the natural borrowing constraint is zero
    if zero_bound:
        aNrmGrid = aXtraGrid
    else:
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0) # Add an asset point at exactly zero
        
    nNrmGrid = np.insert(nNrmGrid, 0, 0.0)    
    
    aNrm_N = aNrmGrid.size
    nNrm_N = nNrmGrid.size
    Share_N = ShareGrid.size
    Shock_N = ShockPrbs_next.size
    
    # FIRST STEP
    # Compute the end-of-period expected future values after all actions
    
    # Create tiled arrays with conforming dimensions. These are used
    # to compute expectations.
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
    
    # Evaluate realizations of the derivatives of next period's value functions
    
    # Always compute the adjusting version
    dvdmAdj_next = dvdmFuncAdj_next(mNrm_next,nNrm_next)
    dvdnAdj_next = dvdnFuncAdj_next(mNrm_next,nNrm_next)
    dvdsAdj_next = np.zeros_like(mNrm_next) # No marginal value of Share if it's a free choice!
    
    # We are interested in marginal values before the realization of the
    # adjustment random variable. Compute those objects
    if AdjustPrb < 1.:
        
        # "Fixed" counterparts
        dvdmFxd_next = dvdmFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdnFxd_next = dvdnFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdsFxd_next = dvdsFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        
        # Expected values with respect to adjustment r.v.
        dvdm_next = AdjustPrb*dvdmAdj_next + (1.-AdjustPrb)*dvdmFxd_next
        dvdn_next = AdjustPrb*dvdnAdj_next + (1.-AdjustPrb)*dvdnFxd_next
        dvds_next = AdjustPrb*dvdsAdj_next + (1.-AdjustPrb)*dvdsFxd_next
        
    else: # Don't bother evaluating if there's no chance that contribution share is fixed
        dvdm_next = dvdmAdj_next
        dvdn_next = dvdnAdj_next
        dvds_next = dvdsAdj_next
        
    # Evaluate realizations of value next period conditional on shocks, assets
    # and shares    
    vAdj_next = vFuncAdj_next(mNrm_next, nNrm_next)
    if AdjustPrb < 1.:
        vFxd_next = vFuncFxd_next(mNrm_next, nNrm_next, Share_next)
        v_next = AdjustPrb*vAdj_next + (1.-AdjustPrb)*vFxd_next
    else: # Don't bother evaluating if there's no chance that portfolio share is fixed
        v_next = vAdj_next
            
    # Calculate end-of-period marginal value of both assets by taking expectations
    temp_fac_A = uP(PermShks_tiled*PermGroFac) # Will use this in a couple places
    EndOfPrddvda = DiscFac*Rfree*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*dvdm_next, axis=3)
    EndOfPrddvdn = DiscFac*Rfree*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*Risky_tiled*dvdn_next, axis=3)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvda)
    EndOfPrddvdnNvrs = uPinv(EndOfPrddvdn)
        
    # Calculate end-of-period value by taking expectations
    temp_fac_B = (PermShks_tiled*PermGroFac)**(1.-CRRA) # Will use this below
    EndOfPrdv = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_B*v_next, axis=3)
    EndOfPrdvNvrs = uInv(EndOfPrdv)
    
    # Compute post-consumption marginal value of contribution share,
    # conditional on shocks
    EndOfPrddvds_cond_undisc = temp_fac_B*( TranShks_tiled*(dvdn_next - dvdm_next) + dvds_next)
    # Discount and integrate over shocks
    EndOfPrddvds = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*EndOfPrddvds_cond_undisc, axis=3)
    
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
        dvdnAdj3Nvrs = EndOfPrddvdnNvrs[a_idx_tiled,n_idx_tiled,opt_idx]
        
    else: # Optimization of Share on continuous interval [0,1]
        # TODO?    
        pass
    
    # Construct interpolators for v3Adj and its derivatives
    vFuncAdj3    = ValueFunc2D(BilinearInterp(vAdj3Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdaFuncAdj3 = MargValueFunc2D(BilinearInterp(dvdaAdj3Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj3 = MargValueFunc2D(BilinearInterp(dvdnAdj3Nvrs, aNrmGrid, nNrmGrid), CRRA)
    
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
    
    # Construct the value function and derivatives
    vAdj2Nvrs = uInv(vFuncAdj3(aTilde_tiled, nTilde_tiled))
    dvdaAdj2Nvrs = uPinv(dvdaFuncAdj3(aTilde_tiled, nTilde_tiled))
    dvdnAdj2Nvrs = uPinv(dvdnFuncAdj3(aTilde_tiled, nTilde_tiled))
    
    vFuncAdj2    = ValueFunc2D(BilinearInterp(vAdj2Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdaFuncAdj2 = MargValueFunc2D(BilinearInterp(dvdaAdj2Nvrs, aNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj2 = MargValueFunc2D(BilinearInterp(dvdnAdj2Nvrs, aNrmGrid, nNrmGrid), CRRA)
    
    # Construct the rebalancing policy function
    DFuncAdj = BilinearInterp(D_tiled, aNrmGrid, nNrmGrid)
    
    # STEP FOUR: EGM inversion to get consumption at endogenous grid, solving
    # the first stage.
    
    # Invert consumption candidates and market resources from the marginal 
    # post-consumption value of liquid assets on the exogenous grid
    cAdj = uPinv(dvdaFuncAdj2(aNrm_tiled, nNrm_tiled))
    mNrmEndog_tiled = aNrm_tiled + cAdj
    
    # Evaluate value function at candidate points (needed for envelope)
    vAdjEndog = u(cAdj) + vFuncAdj2(mNrmEndog_tiled, nNrm_tiled)  
    
    !!!!!!! HERE !!!!!!!!!!!!!!!!!!
    
    # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
    mNrmAdj_now = aNrmGrid + cNrmAdj_now
    
    # Construct the risky share function when the agent can adjust
    if DiscreteShareBool:
        mNrmAdj_mid  = (mNrmAdj_now[1:] + mNrmAdj_now[:-1])/2
        mNrmAdj_plus = mNrmAdj_mid*(1.+1e-12)
        mNrmAdj_comb = (np.transpose(np.vstack((mNrmAdj_mid,mNrmAdj_plus)))).flatten()
        mNrmAdj_comb = np.append(np.insert(mNrmAdj_comb,0,0.0), mNrmAdj_now[-1])
        Share_comb   = (np.transpose(np.vstack((Share_now,Share_now)))).flatten()
        ShareFuncAdj_now = LinearInterp(mNrmAdj_comb, Share_comb)
    else:
        if zero_bound:
            Share_lower_bound = ShareLimit
        else:
            Share_lower_bound = 1.0
        Share_now   = np.insert(Share_now, 0, Share_lower_bound)
        ShareFuncAdj_now = LinearInterp(
                np.insert(mNrmAdj_now,0,0.0),
                Share_now,
                intercept_limit=ShareLimit,
                slope_limit=0.0)
        
    # Construct the consumption function when the agent can adjust
    cNrmAdj_now = np.insert(cNrmAdj_now, 0, 0.0)
    cFuncAdj_now = LinearInterp(np.insert(mNrmAdj_now,0,0.0), cNrmAdj_now)
    
    # Construct the marginal value (of mNrm) function when the agent can adjust
    vPfuncAdj_now = MargValueFunc(cFuncAdj_now, CRRA)
    
    # Construct the consumption function when the agent *can't* adjust the risky share, as well
    # as the marginal value of Share function
    cFuncFxd_by_Share = []
    dvdsFuncFxd_by_Share = []
    for j in range(Share_N):
        cNrmFxd_temp = EndOfPrddvdaNvrs[:,j]
        mNrmFxd_temp = aNrmGrid + cNrmFxd_temp
        cFuncFxd_by_Share.append(LinearInterp(np.insert(mNrmFxd_temp, 0, 0.0), np.insert(cNrmFxd_temp, 0, 0.0)))
        dvdsFuncFxd_by_Share.append(LinearInterp(np.insert(mNrmFxd_temp, 0, 0.0), np.insert(EndOfPrddvds[:,j], 0, EndOfPrddvds[0,j])))
    cFuncFxd_now = LinearInterpOnInterp1D(cFuncFxd_by_Share, ShareGrid)
    dvdsFuncFxd_now = LinearInterpOnInterp1D(dvdsFuncFxd_by_Share, ShareGrid)
    
    # The share function when the agent can't adjust his portfolio is trivial
    ShareFuncFxd_now = IdentityFunction(i_dim=1, n_dims=2)
    
    # Construct the marginal value of mNrm function when the agent can't adjust his share
    dvdmFuncFxd_now = MargValueFunc2D(cFuncFxd_now, CRRA)
    
    # If the value function has been requested, construct it now
    if vFuncBool:
        # First, make an end-of-period value function over aNrm and Share
        EndOfPrdvNvrsFunc = BilinearInterp(EndOfPrdvNvrs, aNrmGrid, ShareGrid)
        EndOfPrdvFunc = ValueFunc2D(EndOfPrdvNvrsFunc, CRRA)
        
        # Construct the value function when the agent can adjust his portfolio
        mNrm_temp  = aXtraGrid # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp  = cFuncAdj_now(mNrm_temp)
        aNrm_temp  = mNrm_temp - cNrm_temp
        Share_temp = ShareFuncAdj_now(mNrm_temp)
        v_temp     = u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Share_temp)
        vNvrs_temp = n(v_temp)
        vNvrsP_temp= uP(cNrm_temp)*nP(v_temp)
        vNvrsFuncAdj = CubicInterp(
                np.insert(mNrm_temp,0,0.0),  # x_list
                np.insert(vNvrs_temp,0,0.0), # f_list
                np.insert(vNvrsP_temp,0,vNvrsP_temp[0])) # dfdx_list
        vFuncAdj_now = ValueFunc(vNvrsFuncAdj, CRRA) # Re-curve the pseudo-inverse value function
        
        # Construct the value function when the agent *can't* adjust his portfolio
        mNrm_temp  = np.tile(np.reshape(aXtraGrid, (aXtraGrid.size, 1)), (1, Share_N))
        Share_temp = np.tile(np.reshape(ShareGrid, (1, Share_N)), (aXtraGrid.size, 1))
        cNrm_temp  = cFuncFxd_now(mNrm_temp, Share_temp)
        aNrm_temp  = mNrm_temp - cNrm_temp
        v_temp     = u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Share_temp)
        vNvrs_temp = n(v_temp)
        vNvrsP_temp= uP(cNrm_temp)*nP(v_temp)
        vNvrsFuncFxd_by_Share = []
        for j in range(Share_N):
            vNvrsFuncFxd_by_Share.append(CubicInterp(
                    np.insert(mNrm_temp[:,0],0,0.0),  # x_list
                    np.insert(vNvrs_temp[:,j],0,0.0), # f_list
                    np.insert(vNvrsP_temp[:,j],0,vNvrsP_temp[j,0]))) #dfdx_list
        vNvrsFuncFxd = LinearInterpOnInterp1D(vNvrsFuncFxd_by_Share, ShareGrid)
        vFuncFxd_now = ValueFunc2D(vNvrsFuncFxd, CRRA)
    
    else: # If vFuncBool is False, fill in dummy values
        vFuncAdj_now = None
        vFuncFxd_now = None

    # Create and return this period's solution
    return PortfolioSolution(
            cFuncAdj = cFuncAdj_now,
            ShareFuncAdj = ShareFuncAdj_now,
            vPfuncAdj = vPfuncAdj_now,
            vFuncAdj = vFuncAdj_now,
            cFuncFxd = cFuncFxd_now,
            ShareFuncFxd = ShareFuncFxd_now,
            dvdmFuncFxd = dvdmFuncFxd_now,
            dvdsFuncFxd = dvdsFuncFxd_now,
            vFuncFxd = vFuncFxd_now
    )
    
    
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
init_riskyContrib = init_risky.copy()
init_riskyContrib['tau']             = 0.1  # Tax rate on risky asset withdrawals
init_riskyContrib['nNrmMin']         = 1e-6 # Use the same parameters for the risky asset grid
init_riskyContrib['nNrmMax']         = 10
init_riskyContrib['nNrmCount']       = 100  #
init_riskyContrib['nNrmNestFac']     = 1    #

# Reduce dimensions while conding the model up
init_riskyContrib['ShareCount']      = 4
init_riskyContrib['aXtraCount']      = 20
init_riskyContrib['nNrmCount']       = 20  #
init_riskyContrib['PermShkCount']    = 3  #
init_riskyContrib['TranShkCount']    = 3 
