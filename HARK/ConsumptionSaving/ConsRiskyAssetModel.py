'''
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
'''
import numpy as np
from copy import deepcopy
from HARK import MetricObject, NullFunc, AgentType # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import(
    IndShockConsumerType,       # PortfolioConsumerType inherits from it
    utility,                    # CRRA utility function
    utility_inv,                # Inverse CRRA utility function
    utilityP,                   # CRRA marginal utility function
    utilityP_inv,               # Inverse CRRA marginal utility function
    init_idiosyncratic_shocks   # Baseline dictionary to build on
)

from HARK.distribution import combineIndepDstns 
from HARK.distribution import Lognormal, Bernoulli # Random draws for simulating agents
from HARK.interpolation import(
        LinearInterp,           # Piecewise linear interpolation
        BilinearInterp,         # 2D interpolator
        TrilinearInterp,        # 3D interpolator
        ConstantFunction,       # Interpolator-like class that returns constant value
        IdentityFunction,        # Interpolator-like class that returns one of its arguments
        ValueFuncCRRA,
        MargValueFuncCRRA
)

from HARK.utilities import makeGridExpMult

class DiscreteInterp2D(MetricObject):
    
    distance_criteria = ['IndexInterp']
    
    def __init__(self, IndexInterp, DiscreteVals):
        
        self.IndexInterp  = IndexInterp
        self.DiscreteVals = DiscreteVals
        self.nVals        = len(self.DiscreteVals)
        
    def __call__(self, x, y):
        
        # Interpolate indices and round to integers
        inds = np.rint(self.IndexInterp(x, y)).astype(int)
        
        # Deal with out-of range indices
        inds[inds < 0]          = 0
        inds[inds >= self.nVals] = self.nVals - 1
        
        # Get values from grid
        return(self.DiscreteVals[inds])


class RiskyAssetConsumerType(IndShockConsumerType):
    """
    A consumer type that has access to a risky asset with lognormal returns
    that are possibly correlated with his income shocks.
    Investment into the risky asset happens through a "share" that represents
    either
    - The share of the agent's total resources allocated to the risky asset.
    - The share of income that the agent diverts to the risky asset
    depending on the model.
    There is a friction that prevents the agent from adjusting his portfolio
    and contribution scheme at any given period with an exogenously given
    probability.
    """
    
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + ['DiscreteShareBool']

    state_vars  = IndShockConsumerType.state_vars + ['ShareNow']
    shock_vars_ = IndShockConsumerType.shock_vars_ + ['AdjustNow','RiskyNow']

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
        
        # self.update is already called inside IndShockConsumerType.__init__
        # Since self is passed to it, we are sure the propper method will be
        # used.
        # self.update()


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
            self.ShockDstn = [combineIndepDstns(self.IncShkDstn[t], self.RiskyDstn[t]) for t in range(self.T_cycle)]
        else:
            self.ShockDstn = [combineIndepDstns(self.IncShkDstn[t], self.RiskyDstn) for t in range(self.T_cycle)]
        self.addToTimeVary('ShockDstn')

        # Mark whether the risky returns and income shocks are independent (they are)
        self.IndepDstnBool = True
        self.addToTimeInv('IndepDstnBool')
    
    def updateAdjustPrb(self):
        '''
        Checks and updates the exogenous probability of the agent being allowed
        to rebalance his portfolio/contribution scheme. It can be time varying.

        Parameters
        ------
        None.

        Returns
        -------
        None.

        '''
        if (type(self.AdjustPrb) is list and (len(self.AdjustPrb) == self.T_cycle)):
            self.addToTimeVary('AdjustPrb')
        elif type(self.AdjustPrb) is list:
            raise AttributeError('If AdjustPrb is time-varying, it must have length of T_cycle!')
        else:
            self.addToTimeInv('AdjustPrb')
            

    def updateShareGrid(self):
        '''
        Creates the attribute ShareGrid as an evenly spaced grid on [0.,1.],
        using the primitive parameter ShareCount.

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
        self.shocks['RiskyNow'] = Lognormal(
                mu, sigma, seed=self.RNG.randint(0, 2**31-1)
            ).draw(1)


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
            
            self.shocks['AdjustNow'] = Bernoulli(
                    self.AdjustPrb, seed=self.RNG.randint(0, 2**31-1)
                ).draw(self.AgentCount)
       
        else: 
            
            AdjustNow = np.zeros(self.AgentCount)  # Initialize shock array
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                N = np.sum(these)
                if N > 0:
                    AdjustPrb = self.AdjustPrb[t - 1]
                    AdjustNow[these] = Bernoulli(AdjustPrb,
                                                 seed=self.RNG.randint(0, 2**31-1)).draw(N)
                    
            self.shocks['AdjustNow'] = AdjustNow

    def initializeSim(self):
        '''
        Initialize the state of simulation attributes.  Simply calls the same
        method for IndShockConsumerType, then initializes the new states/shocks
        AdjustNow and ShareNow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.initializeSim(self)
        self.state_now['ShareNow'] = np.zeros(self.AgentCount)
        self.shocks['AdjustNow'] = np.zeros(self.AgentCount, dtype=bool)


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
        self.state_now['ShareNow'][which_agents] = 0.
        self.AdjustNow[which_agents] = False


    def getShocks(self):
        '''
        Draw idiosyncratic income shocks, just as for IndShockConsumerType, then draw
        a single common value for the risky asset return.  Also draws whether each
        agent is able to adjust their portfolio this period.

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


# Class for the contribution share stage solution
class RiskyContribShaSolution(MetricObject):
    """
    A class for representing the solution to the contribution-share stage of
    the 'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncShaAdj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    ShareFuncAdj : Interp2D
        Income contribution share function over normalized liquid resources
        and normalized iliquid resources when the agent is able to adjust his
        portfolio.
    dvdmFuncShaAdj : MargValueFunc2D
        Marginal value function over normalized liquid resources when the agent
        is able to adjust his portfolio.
    dvdnFuncShaAdj : MargValueFunc2D
        Marginal value function over normalized iliquid resources when the
        agent is able to adjust his portfolio.
    vFuncShaFxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
    ShareFuncFxd : Interp3D
        Income contribution share function over normalized liquid resources,
        iliquid resources, and income contribution share when the agent is not
        able to adjust his portfolio.
        Should be an IdentityFunc.
    dvdmFuncShaFxd : MargValueFunc3D
        Marginal value function over normalized liquid resources when the agent
        is not able to adjust his portfolio.
    dvdnFuncShaFxd : MargValueFunc3D
        Marginal value function over normalized iliquid resources when the
        agent is not able to adjust his portfolio.
    dvdsFuncShaFxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to adjust his portfolio
    """
    
    distance_criteria = ['dvdmFuncShaAdj','dvdnFuncShaAdj']
    
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
class RiskyContribRebSolution(MetricObject):
    """
    A class for representing the solution to the asset-rebalancing stage of
    the 'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncRebAdj : ValueFunc2D
        Stage value function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    DFuncAdj : Interp2D
        Deposit function over normalized liquid resources and normalized
        iliquid resources when the agent is able to adjust his portfolio.
    dvdmFuncRebAdj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    dvdnFuncRebAdj : MargValueFunc2D
        Marginal value over normalized liquid resources when the agent is able
        to adjust his portfolio.
    vFuncRebFxd : ValueFunc3D
        Stage value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share when the agent is
        not able to adjust his portfolio.
    DFuncFxd : Interp2D
        Deposit function over normalized liquid resources, normalized iliquid
        resources, and income contribution share when the agent is not able to
        adjust his portfolio.
        Must be ConstantFunction(0.0)
    dvdmFuncRebFxd : MargValueFunc3D
        Marginal value over normalized liquid resources when the agent is not
        able to adjust his portfolio.
    dvdnFuncRebFxd : MargValueFunc3D
        Marginal value over normalized iliquid resources when the agent is not
        able to adjust his portfolio.
    dvdsFuncRebFxd : Interp3D
        Marginal value function over income contribution share when the agent
        is not able to ajust his portfolio.
    """
    
    distance_criteria = ['dvdmFuncRebAdj','dvdnFuncRebAdj']
    
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
        
# Class for the consumption stage solution
class RiskyContribCnsSolution(MetricObject):
    """
    A class for representing the solution to the consumption stage of the
    'RiskyContrib' model.
    
    Parameters
    ----------
    vFuncCns : ValueFunc3D
        Stage-value function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    cFunc : Interp3D
        Consumption function over normalized liquid resources, normalized
        iliquid resources, and income contribution share.
    dvdmFuncCns : MargValueFunc3D
        Marginal value function over normalized liquid resources.
    dvdnFuncCns : MargValueFunc3D
        Marginal value function over normalized iliquid resources.
    dvdsFuncCns : Interp3D
        Marginal value function over income contribution share.
    """
    
    distance_criteria = ['dvdmFuncCns','dvdnFuncCns']
    
    def __init__(self,
                
        # Consumption stage
        vFuncCns = None,
        cFunc = None,
        dvdmFuncCns = None,
        dvdnFuncCns = None,
        dvdsFuncCns = None
        
    ):
                
        # Consumption stage
        if vFuncCns is None:
            vFuncCns = NullFunc()
        if cFunc is None:
            cFunc = NullFunc()
        if dvdmFuncCns is None:
            dvdmFuncCns = NullFunc()
        if dvdnFuncCns is None:
            dvdmFuncCns = NullFunc()
        if dvdsFuncCns is None:
            dvdsFuncCns = NullFunc()
        
        # Consumption stage
        self.vFuncCns = vFuncCns
        self.cFunc = cFunc
        self.dvdmFuncCns = dvdmFuncCns
        self.dvdnFuncCns = dvdnFuncCns
        self.dvdsFuncCns = dvdsFuncCns

# Class for the solution of a whole period
class RiskyContribSolution(MetricObject):
    
    # Declare that the distance metric will be an object called 'ConvCriterion'
    # TODO: this should just be stageSols, but HARK's distance code can not
    # deal with dictionaries at the moment: only lists. This is therefore a
    # workarround.
    distance_criteria = ['ConvCriterion']

    def __init__(self, Reb, Sha, Cns):
        
        # Dictionary of stage solutions
        self.stageSols = {'Reb': Reb, 'Sha': Sha, 'Cns': Cns}
        
        # And convergence criterion. This is the object that will be checked
        # for convergence in the infinite horizon solution.
        # We inted the 'distance' to be an aggregate of the distance of 
        # each stage's distance.
        self.ConvCriterion = list(self.stageSols.values())
    

class RiskyContribConsumerType(RiskyAssetConsumerType):
    """
    TODO: model description
    """
    
    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)

    state_vars  = RiskyAssetConsumerType.state_vars + ['mNrmTildeNow','nNrmTildeNow', 'ShareNow']
    shock_vars_ = RiskyAssetConsumerType.shock_vars_

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
        self.solveOnePeriod = solveRiskyContrib
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
        
        # Construct the terminal solution backwards.
        
        # Start with the consumption stage. All liquid resources are consumed.
        cFunc_term = IdentityFunction(i_dim = 0, n_dims = 3)
        vFuncCns_term = ValueFuncCRRA(cFunc_term, CRRA = self.CRRA)
        # Marginal values
        dvdmFuncCns_term = MargValueFuncCRRA(cFunc_term, CRRA = self.CRRA)
        dvdnFuncCns_term = ConstantFunction(0.0)
        dvdsFuncCns_term = ConstantFunction(0.0)
        
        CnsStageSol = RiskyContribCnsSolution(
            # Consumption stage
            vFuncCns = vFuncCns_term,
            cFunc = cFunc_term,
            dvdmFuncCns = dvdmFuncCns_term,
            dvdnFuncCns = dvdnFuncCns_term,
            dvdsFuncCns = dvdsFuncCns_term)
        
        # Share stage
        
        # It's irrelevant because there is no future period. Set share to 0.
        # Create a dummy 2-d consumption function to get value function and marginal
        c2d = IdentityFunction(i_dim = 0, n_dims = 2)
        ShaStageSol = RiskyContribShaSolution(
            # Adjust
            vFuncShaAdj = ValueFuncCRRA(c2d, CRRA = self.CRRA),
            ShareFuncAdj = ConstantFunction(0.0),
            dvdmFuncShaAdj = MargValueFuncCRRA(c2d, CRRA = self.CRRA),
            dvdnFuncShaAdj = ConstantFunction(0.0),
            
            # Fixed
            vFuncShaFxd = vFuncCns_term,
            ShareFuncFxd = IdentityFunction(i_dim = 2, n_dims = 3),
            dvdmFuncShaFxd = dvdmFuncCns_term,
            dvdnFuncShaFxd = dvdnFuncCns_term,
            dvdsFuncShaFxd = dvdsFuncCns_term
        )
        
        # Rabalancing stage

        # Adjusting agent: 
        # Withdraw everything from the pension fund and consume everything
        DFuncAdj_term = ConstantFunction(-1.0)
        
        # Find the withdrawal penalty
        if type(self.tau) is list:
            tau = self.tau[-1]
        else:
            tau = self.tau
        
        # Value and marginal value function of the adjusting agent
        vFuncRebAdj_term = ValueFuncCRRA(lambda m,n: m + n/(1+tau), self.CRRA)
        dvdmFuncRebAdj_term = MargValueFuncCRRA(lambda m,n: m + n/(1+tau), self.CRRA)
        # A marginal unit of n will be withdrawn and put into m. Then consumed.
        dvdnFuncRebAdj_term = lambda m,n: dvdmFuncRebAdj_term(m,n)/(1+tau)
        
        RebStageSol = RiskyContribRebSolution(
            # Rebalancing stage
            vFuncRebAdj = vFuncRebAdj_term,
            DFuncAdj = DFuncAdj_term,
            dvdmFuncRebAdj = dvdmFuncRebAdj_term,
            dvdnFuncRebAdj = dvdnFuncRebAdj_term,
            
            # Adjusting stage
            vFuncRebFxd = vFuncCns_term,
            DFuncFxd = ConstantFunction(0.0),
            dvdmFuncRebFxd = dvdmFuncCns_term,
            dvdnFuncRebFxd = dvdnFuncCns_term,
            dvdsFuncRebFxd = dvdsFuncCns_term)
        
        # Construct the terminal period solution
        self.solution_terminal = RiskyContribSolution(RebStageSol,
                                                      ShaStageSol,
                                                      CnsStageSol)
        
    
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
            self.ShockDstn = [combineIndepDstns(self.IncShkDstn[t], self.RiskyDstn[t]) for t in range(self.T_cycle)]
        else:
            self.ShockDstn = [combineIndepDstns(self.IncShkDstn[t], self.RiskyDstn) for t in range(self.T_cycle)]
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
        self.dGrid = np.linspace(0,1,self.dCount)
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
        # these need to be set because "post states",
        # but are a control variable and shock, respectively
        self.controls['ShareNow'] = np.zeros(self.AgentCount)
        self.shocks['AdjustNow'] = np.zeros(self.AgentCount, dtype=bool)
        IndShockConsumerType.initializeSim(self)
    
    
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
        self.state_now['nNrmTildeNow'][which_agents] = 0.
        self.state_now['ShareNow'][which_agents] = 0.
        # here a shock is being used as a 'post state'
        self.shocks['AdjustNow'][which_agents] = False
        
            
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
                
        # Mortality and birth happens only at the start of the period
        self.getMortality()
            
        # Shocks are drawn in the first stage
        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.readShocks()
        else:  # Otherwise, draw shocks as usual according to subclass-specific method
            self.getShocks()
        
        # Stages in chronological order
        stages = ['Reb','Sha','Cns']
        
        setStates = {'Reb': self.getStatesReb,
                     'Sha': self.getStatesSha,
                     'Cns': self.getStatesCns}
        
        setControls = {'Reb': self.getControlsReb,
                       'Sha': self.getControlsSha,
                       'Cns': self.getControlsCns}
        
        for s in stages:
            setStates[s]()
            setControls[s]()
    
        self.getPostStates()
        
        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end
        
    
    def getStatesReb(self):
        """
        Get states for the first stage: rebalancing.
        """
        pLvlPrev      = self.state_now['pLvlNow']
        aNrmPrev      = self.state_now['aNrmNow']
        nNrmTildePrev = self.state_now['nNrmTildeNow']
        RfreeNow      = self.Rfree
        RriskNow      = self.shocks['RiskyNow']
        
        # Calculate new states:
        
        # Permanent income
        self.state_now['pLvlNow'] = (
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
        
        bNrmNow = RfEffNow * aNrmPrev  # Liquid balances before labor income
        gNrmNow = RrEffNow * nNrmTildePrev  # Iliquid balances before labor income
        
        # Liquid balances after labor income
        self.state_now['mNrmNow'] = (
            bNrmNow + self.shocks["TranShkNow"] * (1 - self.state_now['ShareNow'])
        )
        # Iliquid balances after labor income
        self.state_now['nNrmNow'] = (
            gNrmNow + self.shocks["TranShkNow"] * self.state_now['ShareNow']
        )
        
        return None

    
    def getControlsReb(self):
        """
        """
        DNrmNow = np.zeros(self.AgentCount) + np.nan
        
        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            
            # Find agents in this period-stage
            these = t == self.t_cycle
                           
            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.shocks['AdjustNow'])
            DNrmNow[those] = self.solution[t].stageSols['Reb'].DFuncAdj(
                self.state_now['mNrmNow'][those],
                self.state_now['nNrmNow'][those]
            )
                
            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks['AdjustNow']))
            DNrmNow[those] = self.solution[t].stageSols['Reb'].DFuncFxd(
                self.state_now['mNrmNow'][those],
                self.state_now['nNrmNow'][those],
                self.state_now['ShareNow'][those]
            )

        # Store controls as attributes of self
        self.controls['DNrmNow'] = DNrmNow
        
    def getStatesSha(self):
        """
        """
        
        # Post-states are assets after rebalancing

        if not 'tau' in self.time_vary:
        
            mNrmTildeNow, nNrmTildeNow = rebalanceAssets(self.controls['DNrmNow'],
                                                         self.state_now['mNrmNow'],
                                                         self.state_now['nNrmNow'], self.tau)
        
        else:
            
            # Initialize
            mNrmTildeNow = np.zeros_like(self.state_now['mNrmNow']) + np.nan
            nNrmTildeNow = np.zeros_like(self.state_now['mNrmNow']) + np.nan
            
            # Loop over each period of the cycle, getting controls separately depending on "age"
            for t in range(self.T_cycle):
            
                # Find agents in this period-stage
                these = t == self.t_cycle
                
                if np.sum(these) > 0:
                    tau = self.tau[t]
                    
                    mNrmTildeNow[these], nNrmTildeNow[these] = rebalanceAssets(self.controls['DNrmNow'][these],
                                                                               self.state_now['mNrmNow'][these],
                                                                               self.state_now['nNrmNow'][these],
                                                                               tau)
        
        self.state_now['mNrmTildeNow'] = mNrmTildeNow
        self.state_now['nNrmTildeNow'] = nNrmTildeNow
    
    def getControlsSha(self):
        """
        """
        
        ShareNow = np.zeros(self.AgentCount) + np.nan
        
        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            
            # Find agents in this period-stage
            these = t == self.t_cycle
                           
            # Get controls for agents who *can* adjust.
            those = np.logical_and(these, self.shocks['AdjustNow'])
            ShareNow[those] = self.solution[t].stageSols['Sha'].ShareFuncAdj(
                self.state_now['mNrmTildeNow'][those],
                self.state_now['nNrmTildeNow'][those]
            )
                
            # Get Controls for agents who *can't* adjust.
            those = np.logical_and(these, np.logical_not(self.shocks['AdjustNow']))
            ShareNow[those] = self.solution[t].stageSols['Sha'].ShareFuncFxd(
                self.state_now['mNrmTildeNow'][those],
                self.state_now['nNrmTildeNow'][those],
                self.state_now['ShareNow'][those]
            )

        # Store controls as attributes of self
        self.controls['ShareNow'] = ShareNow     
        # Share is also a state
        # TODO: Ask Seb how this is handled
        self.state_now['ShareNow'] = ShareNow
        
    def getStatesCns(self):
        # No new states need to be computed in the consumption stage
        pass
        
    def getControlsCns(self):
        """
        """
        
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        
        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            
            # Find agents in this period-stage
            these = t == self.t_cycle
                           
            # Get consumption
            cNrmNow[these] = self.solution[t].stageSols['Cns'].cFunc(
                self.state_now['mNrmTildeNow'][these],
                self.state_now['nNrmTildeNow'][these],
                self.state_now['ShareNow'][these]
            )
            
        # Store controls as attributes of self
        # Since agents might be willing to end the period with a = 0, make
        # sure consumption does not go over m because of some numerical error.
        self.controls['cNrmNow'] = np.minimum(cNrmNow,self.state_now['mNrmTildeNow'])            
        
    def getPostStates(self):
        """
        """
        self.state_now['aNrmNow'] = self.state_now['mNrmTildeNow'] - self.controls['cNrmNow']
         
    
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
    mTil[inds] = m[inds] - d[inds]*n[inds]*(1 - tau)
    nTil[inds] = n[inds]*(1+d[inds])
    
    return (mTil, nTil)




# Consumption stage solver
def solveRiskyContribCnsStage(solution_next,ShockDstn,IncShkDstn,RiskyDstn,
                              LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                              BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                              ShareGrid,vFuncBool,AdjustPrb,
                              DiscreteShareBool,IndepDstnBool,
                              **unused_params):
    
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
        
        ShockDstn = combineIndepDstns(IncShkDstn, RiskyDstn)
    
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
    if vFuncBool:
        vAdj_next    = vFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdmAdj_next = dvdmFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdnAdj_next = dvdnFuncRebAdj_next(mNrm_next,nNrm_next)
    dvdsAdj_next = np.zeros_like(mNrm_next)# No marginal value of Share if it's a free choice!
    
    # We are interested in marginal values before the realization of the
    # adjustment random variable. Compute those objects
    if AdjustPrb < 1.:
        
        # "Fixed" counterparts
        dvdmFxd_next = dvdmFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdnFxd_next = dvdnFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        dvdsFxd_next = dvdsFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
        
        # Expected values with respect to adjustment r.v.
        dvdm_next = AdjustPrb*dvdmAdj_next + (1.-AdjustPrb)*dvdmFxd_next
        dvdn_next = AdjustPrb*dvdnAdj_next + (1.-AdjustPrb)*dvdnFxd_next
        dvds_next = AdjustPrb*dvdsAdj_next + (1.-AdjustPrb)*dvdsFxd_next
        
        # Value function if needed
        if vFuncBool:
            vFxd_next = vFuncRebFxd_next(mNrm_next, nNrm_next, Share_next)
            v_next    = AdjustPrb*vAdj_next    + (1.-AdjustPrb)*vFxd_next
        
    else: # Don't evaluate if there's no chance that contribution share is fixed
        
        dvdm_next = dvdmAdj_next
        dvdn_next = dvdnAdj_next
        dvds_next = dvdsAdj_next
        
        if vFuncBool:
            v_next    = vAdj_next
        
    # Calculate end-of-period marginal value of both assets by taking expectations
    temp_fac_A = uP(PermShks_tiled*PermGroFac) # Will use this in a couple places
    EndOfPrddvda = DiscFac*Rfree*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*dvdm_next, axis=3)
    EndOfPrddvdn = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_A*Risky_tiled*dvdn_next, axis=3)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvda)
    EndOfPrddvdnNvrs = uPinv(EndOfPrddvdn)
        
    # Calculate end-of-period value if needed
    temp_fac_B = (PermShks_tiled*PermGroFac)**(1.-CRRA) # Will use this below
    if vFuncBool:
        EndOfPrdv = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*temp_fac_B*v_next, axis=3)
        EndOfPrdvNvrs = uInv(EndOfPrdv)
        
        # Construct an interpolator for EndOfPrdV. It will be used later.
        EndOfPrdvFunc = ValueFuncCRRA(TrilinearInterp(EndOfPrdvNvrs, aNrmGrid,
                                                    nNrmGrid, ShareGrid),
                                    CRRA)
    
    # Find EndOfPrddvds.
    
    # There are two parts to it:
    # - If income > 0, it shifts resources from m to n. Call this 'distribution
    #   effect'
    # - Since s can be fixed in the future there is also a marginal effect coming
    #   from the future.
        
    # Find distribution effect. Initialize at 0.
    distribEffect = np.zeros_like(TranShks_tiled)
    # Find the effect where income is not 0
    inds = TranShks_next > 0.0
    distribEffect[:,:,:,inds] = TranShks_tiled[:,:,:,inds] * (dvdn_next[:,:,:,inds] - dvdm_next[:,:,:,inds])
    
    # Add the two effects
    EndOfPrddvds_cond_undisc = temp_fac_B*(distribEffect + dvds_next)    
    # Discount and integrate over shocks
    EndOfPrddvds = DiscFac*LivPrb*np.sum(ShockPrbs_tiled*EndOfPrddvds_cond_undisc, axis=3)
    
    # STEP TWO:
    # Solve the consumption problem and create interpolators for c, vCns,
    # and its derivatives.
    
    # Recast a, n, and s now that the shock dimension has been integrated over
    aNrm_tiled  = aNrm_tiled[:,:,:,0]
    nNrm_tiled  = nNrm_tiled[:,:,:,0]
    Share_tiled = Share_tiled[:,:,:,0]
    
    # Apply EGM over liquid resources at every (n,s) to find consumption.
    c_end    = EndOfPrddvdaNvrs
    mNrm_end = aNrm_tiled + c_end
    
    # Now construct interpolators for c and the derivatives of vCns.
    # The m grid is different for every (n,s). We interpolate the object of
    # interest on the regular m grid for every (n,s). At the end we will have
    # values of the functions of interest on a regular (m,n,s) grid. We use
    # trilinear interpolation on those points.
    
    # Expand the regular m grid to contain 0.
    mNrmGrid = np.insert(mNrmGrid,0,0)
    mNrm_N = len(mNrmGrid)
    
    # Dimensions might have changed, so re-create tiled arrays
    mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1,1)), (1,nNrm_N,Share_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (mNrm_N,1,Share_N))
    Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (mNrm_N,nNrm_N,1))
    
    # Initialize arrays
    c_vals        = np.zeros_like(mNrm_tiled)
    dvdnNvrs_vals = np.zeros_like(mNrm_tiled)
    dvds_vals     = np.zeros_like(mNrm_tiled)
    
    for nInd in range(nNrm_N):
        for sInd in range(Share_N):
            
            # Extract the endogenous m grid for particular (n,s).
            m_ns = mNrm_end[:,nInd,sInd]
            
            # Check if there is a natural constraint
            if m_ns[0] == 0.0:
                
                # There's no need to insert points since we have m==0.0
                
                # c
                c_vals[:,nInd,sInd] = LinearInterp(m_ns,c_end[:,nInd,sInd])(mNrmGrid)
                
                # dvdnNvrs
                dvdnNvrs_vals[:,nInd,sInd] = LinearInterp(m_ns, EndOfPrddvdnNvrs[:,nInd,sInd])(mNrmGrid)
                
                # dvds
                # TODO: this might returns NaN when m=n=0. This might propagate.
                dvds_vals[:,nInd,sInd] = LinearInterp(m_ns, EndOfPrddvds[:,nInd,sInd])(mNrmGrid)
                
            else:
                
                # We know that:
                # -The lowest gridpoints of both a and n are 0.
                # -Consumption at m < m0 is m.
                # -dvdnFxd at (m,n) for m < m0(n) is dvdnFxd(m0,n)
                # -Same is true for dvdsFxd
                
                # c
                c_vals[:,nInd,sInd] = LinearInterp(
                    np.insert(m_ns,0,0),
                    np.insert(c_end[:,nInd,sInd],0,0)
                    )(mNrmGrid)
                
                # dvdnNvrs
                dvdnNvrs_vals[:,nInd,sInd] = LinearInterp(
                    np.insert(m_ns,0,0),
                    np.insert(EndOfPrddvdnNvrs[:,nInd,sInd],0,EndOfPrddvdnNvrs[0,nInd,sInd])
                    )(mNrmGrid)
                
                # dvds
                dvds_vals[:,nInd,sInd] = LinearInterp(
                    np.insert(m_ns,0,0),
                    np.insert(EndOfPrddvds[:,nInd,sInd],0,EndOfPrddvds[0,nInd,sInd])
                    )(mNrmGrid)
                
    # With the arrays filled, create 3D interpolators
    
    # Consumption interpolator
    cFunc = TrilinearInterp(c_vals, mNrmGrid, nNrmGrid, ShareGrid)
    # dvdmCns interpolator
    dvdmFuncCns = MargValueFuncCRRA(cFunc, CRRA)
    # dvdnCns interpolator
    dvdnNvrsFunc = TrilinearInterp(dvdnNvrs_vals, mNrmGrid, nNrmGrid, ShareGrid)
    dvdnFuncCns = MargValueFuncCRRA(dvdnNvrsFunc, CRRA)
    # dvdsCns interpolator
    # TODO: dvds might be NaN. Check and fix?
    dvdsFuncCns = TrilinearInterp(dvds_vals, mNrmGrid, nNrmGrid, ShareGrid)
    
    # Compute value function if needed
    if vFuncBool:
        # Consumption in the regular grid
        aNrm_reg = mNrm_tiled - c_vals
        vCns = u(c_vals) + EndOfPrdvFunc(aNrm_reg, nNrm_tiled, Share_tiled) 
        vNvrsCns = uInv(vCns)
        vNvrsFuncCns = TrilinearInterp(vNvrsCns, mNrmGrid, nNrmGrid, ShareGrid)
        vFuncCns     = ValueFuncCRRA(vNvrsFuncCns, CRRA)
    else:
        vFuncCns = NullFunc()
        
    solution = RiskyContribCnsSolution(
        vFuncCns = vFuncCns,
        cFunc = cFunc,
        dvdmFuncCns = dvdmFuncCns,
        dvdnFuncCns = dvdnFuncCns,
        dvdsFuncCns = dvdsFuncCns
    )
    
    return solution


# Solver for the contribution stage
def solveRiskyContribShaStage(solution_next,CRRA,AdjustPrb,
                              mNrmGrid,nNrmGrid,ShareGrid,
                              DiscreteShareBool, vFuncBool,
                              **unused_params):
    
    # Unpack solution from the next sub-stage
    vFuncCns_next    = solution_next.vFuncCns
    cFunc_next       = solution_next.cFunc
    dvdmFuncCns_next = solution_next.dvdmFuncCns
    dvdnFuncCns_next = solution_next.dvdnFuncCns
    dvdsFuncCns_next = solution_next.dvdsFuncCns
    
    # Define temporary functions for utility and its derivative and inverse
    uPinv = lambda x : utilityP_inv(x, CRRA)
    
    # Create tiled grids    

    # Add 0 to the m and n grids
    nNrmGrid = np.insert(nNrmGrid, 0, 0.0)  
    nNrm_N = len(nNrmGrid)
    mNrmGrid = np.insert(mNrmGrid,0,0)
    mNrm_N = len(mNrmGrid)
    Share_N = len(ShareGrid)
    
    if AdjustPrb == 1.0:
        # If the readjustment probability is 1, set the share to 0:
        # - If there is a withdrawal tax: better for the agent to observe
        #   income before rebalancing.
        # - If there is no tax: all shares should yield the same value.
        mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1)), (1,nNrm_N))
        nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N)), (mNrm_N,1))
        
        optIdx   = np.zeros_like(mNrm_tiled, dtype = int)
        optShare = ShareGrid[optIdx]
        
        if vFuncBool:
            vNvrsSha = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, optShare)
        
    else:
        
        # Figure out optimal share by evaluating all alternatives at all
        # (m,n) combinations
        m_idx_tiled = np.tile(np.reshape(np.arange(mNrm_N), (mNrm_N,1)), (1,nNrm_N))
        n_idx_tiled = np.tile(np.reshape(np.arange(nNrm_N), (1,nNrm_N)), (mNrm_N,1))
        
        mNrm_tiled = np.tile(np.reshape(mNrmGrid, (mNrm_N,1,1)), (1,nNrm_N,Share_N))
        nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,nNrm_N,1)), (mNrm_N,1,Share_N))
        Share_tiled = np.tile(np.reshape(ShareGrid, (1,1,Share_N)), (mNrm_N,nNrm_N,1))
        
        if DiscreteShareBool:
        
            # Evaluate value function to optimize over shares.
            # Do it in inverse space
            vNvrs = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, Share_tiled)
            
            # Find the optimal share at each (m,n).
            optIdx = np.argmax(vNvrs, axis = 2)
            
            # Compute objects needed for the value function and its derivatives
            vNvrsSha     = vNvrs[m_idx_tiled, n_idx_tiled, optIdx]
            optShare     = ShareGrid[optIdx]
            
            # Project grids
            mNrm_tiled = mNrm_tiled[:,:,0]
            nNrm_tiled = nNrm_tiled[:,:,0]
        
        else:
            
            # Evaluate the marginal value of the contribution share at
            # every (m,n,s) gridpoint
            dvds = dvdsFuncCns_next(mNrm_tiled, nNrm_tiled, Share_tiled)
            
            # If the derivative is negative at the lowest share, then s[0] is optimal
            constrained_bot = dvds[:,:,0] <= 0.0
            # If it is poitive at the highest share, then s[-1] is optimal
            constrained_top = dvds[:,:,-1] >= 0.0
            
            # Find indices at which the derivative crosses 0 for the 1st time
            # will be 0 if it never does, but "constrained_top/bot" deals with that
            crossings = np.logical_and(dvds[:,:, :-1] >= 0.0, dvds[:,:, 1:] <= 0.0)
            idx = np.argmax(crossings, axis = 2)
            
            # Linearly interpolate the optimal share
            idx1 = idx+1
            slopes = (dvds[m_idx_tiled, n_idx_tiled,idx1] - dvds[m_idx_tiled, n_idx_tiled,idx]) / (ShareGrid[idx1] - ShareGrid[idx])
            optShare = ShareGrid[idx] - dvds[m_idx_tiled, n_idx_tiled,idx]/slopes
            
            # Replace the ones we knew were constrained
            optShare[constrained_bot] = ShareGrid[0]
            optShare[constrained_top] = ShareGrid[-1]
        
            # Project grids
            mNrm_tiled = mNrm_tiled[:,:,0]
            nNrm_tiled = nNrm_tiled[:,:,0]
            
            # Evaluate the inverse value function at the optimal shares
            if vFuncBool:
                vNvrsSha = vFuncCns_next.func(mNrm_tiled, nNrm_tiled, optShare)
    
    dvdmNvrsSha  = cFunc_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnSha      = dvdnFuncCns_next(mNrm_tiled, nNrm_tiled, optShare)
    dvdnNvrsSha  = uPinv(dvdnSha)
    
    # Interpolators
    
    # Value function if needed
    if vFuncBool:
        vNvrsFuncSha    = BilinearInterp(vNvrsSha, mNrmGrid, nNrmGrid)
        vFuncSha        = ValueFuncCRRA(vNvrsFuncSha, CRRA)
    else:
        vFuncSha = NullFunc()
        
    # Contribution share function
    if DiscreteShareBool:
        ShareFunc = DiscreteInterp2D(BilinearInterp(optIdx, mNrmGrid, nNrmGrid),
                                     ShareGrid)
    else:
        ShareFunc       = BilinearInterp(optShare, mNrmGrid, nNrmGrid)
        
    dvdmNvrsFuncSha = BilinearInterp(dvdmNvrsSha, mNrmGrid, nNrmGrid)
    dvdmFuncSha     = MargValueFuncCRRA(dvdmNvrsFuncSha, CRRA)
    dvdnNvrsFuncSha = BilinearInterp(dvdnNvrsSha, mNrmGrid, nNrmGrid)
    dvdnFuncSha     = MargValueFuncCRRA(dvdnNvrsFuncSha, CRRA)
    
    solution = RiskyContribShaSolution(
        vFuncShaAdj = vFuncSha,
        ShareFuncAdj = ShareFunc,
        dvdmFuncShaAdj = dvdmFuncSha,
        dvdnFuncShaAdj = dvdnFuncSha,
        
        # The fixed agent does nothing at this stage,
        # so his value functions are the next problem's
        vFuncShaFxd = vFuncCns_next,
        ShareFuncFxd = IdentityFunction(i_dim = 2, n_dims = 3),
        dvdmFuncShaFxd = dvdmFuncCns_next,
        dvdnFuncShaFxd = dvdnFuncCns_next,
        dvdsFuncShaFxd = dvdsFuncCns_next
    )
    
    return solution
    
# Solver for the asset rebalancing stage
def solveRiskyContribRebStage(solution_next,
                              CRRA,tau,
                              nNrmGrid,mNrmGrid,dGrid,vFuncBool,
                              **unused_params):
    
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
    
    # Duplicate d so that possible values are -dGrid,dGrid. Duplicate 0 is
    # intentional since the tax causes a discontinuity. We need the value
    # from the left and right.
    dGrid = np.concatenate((-1*np.flip(dGrid),dGrid))
    
    # It will be useful to pre-evaluate marginals at every (m,n,d) combination
    
    # Create tiled arrays for every d,m,n option
    d_N2 = len(dGrid)
    d_tiled    = np.tile(np.reshape(dGrid,    (d_N2,1,1)), (1, mNrm_N,nNrm_N))
    mNrm_tiled = np.tile(np.reshape(mNrmGrid, (1,mNrm_N,1)), (d_N2,1,nNrm_N))
    nNrm_tiled = np.tile(np.reshape(nNrmGrid, (1,1,nNrm_N)), (d_N2,mNrm_N,1))
    
    # Get post-rebalancing assets the m_tilde, n_tilde.
    m_tilde, n_tilde = rebalanceAssets(d_tiled, mNrm_tiled, nNrm_tiled, tau)
    
    # Now the marginals, in inverse space
    dvdmNvrs = dvdmFuncAdj_next.cFunc(m_tilde, n_tilde)
    dvdnNvrs = dvdnFuncAdj_next.cFunc(m_tilde, n_tilde)
    
    # Pre-evaluate the inverse of (1-tau)
    taxNvrs = uPinv(1-tau)
    # Create a tiled array of the tax
    taxNvrs_tiled = np.tile(np.reshape(np.concatenate([np.repeat(taxNvrs,d_N),
                                                       np.ones(d_N, dtype=np.double)]),
                                       (d_N2,1,1)),
                            (1, mNrm_N,nNrm_N))
    
    # The FOC is dvdn = tax*dvdm or dvdnNvrs = taxNvrs*dvdmNvrs
    dvdDNvrs = dvdnNvrs - taxNvrs_tiled*dvdmNvrs
    # The optimal d will be at the first point where dvdD < 0. The inverse
    # transformation flips the sign.
    
    # If the derivative is negative (inverse positive) at the lowest d,
    # then d == -1.0 is optimal
    constrained_bot = dvdDNvrs[0,:,:] >= 0.0
    # If it is positive (inverse negative) at the highest d, then d[-1] = 1.0
    # is optimal
    constrained_top = dvdDNvrs[-1,:,:,] <= 0.0
            
    # Find indices at which the derivative crosses 0 for the 1st time
    # will be 0 if it never does, but "constrained_top/bot" deals with that
    crossings = np.logical_and(dvdDNvrs[:-1,:,:] <= 0.0, dvdDNvrs[1:,:, :] >= 0.0)
    idx = np.argmax(crossings, axis = 0)
    
    m_idx_tiled = np.tile(np.reshape(np.arange(mNrm_N), (mNrm_N,1)), (1,nNrm_N))
    n_idx_tiled = np.tile(np.reshape(np.arange(nNrm_N), (1,nNrm_N)), (mNrm_N,1))
    
    # Linearly interpolate the optimal withdrawal percentage d
    idx1 = idx+1
    slopes = (dvdDNvrs[idx1,m_idx_tiled, n_idx_tiled] - dvdDNvrs[idx, m_idx_tiled, n_idx_tiled]) / (dGrid[idx1] - dGrid[idx])
    dOpt = dGrid[idx] - dvdDNvrs[idx, m_idx_tiled, n_idx_tiled]/slopes
            
    # Replace the ones we knew were constrained
    dOpt[constrained_bot] = dGrid[0]
    dOpt[constrained_top] = dGrid[-1]

    # Find m_tilde and n_tilde
    mtil_opt, ntil_opt = rebalanceAssets(dOpt, mNrm_tiled[0], nNrm_tiled[0], tau)
            
    # Now the derivatives. These are not straight forward because of corner
    # solutions with partial derivatives that change the limits. The idea then
    # is to evaluate the possible uses of the marginal unit of resources and
    # take the maximum.
    
    # An additional unit of m
    marg_m      = dvdmFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n kept in n
    marg_n      = dvdnFuncAdj_next(mtil_opt, ntil_opt)
    # An additional unit of n withdrawn to m
    marg_n_to_m = marg_m*(1-tau)
    
    # Marginal value is the maximum of the marginals in their possible uses 
    dvdmAdj     = np.maximum(marg_m, marg_n)
    dvdmNvrsAdj = uPinv(dvdmAdj)
    dvdnAdj     = np.maximum(marg_n, marg_n_to_m)
    dvdnNvrsAdj = uPinv(dvdnAdj)
    
    # Interpolators
    
    # Value
    if vFuncBool:
        vNvrsAdj = vFuncAdj_next.func(mtil_opt, ntil_opt)
        vNvrsFuncAdj = BilinearInterp(vNvrsAdj, mNrmGrid, nNrmGrid)
        vFuncAdj     = ValueFuncCRRA(vNvrsFuncAdj, CRRA)
    else:
        vFuncAdj = NullFunc()
        
    # Marginals
    dvdmFuncAdj = MargValueFuncCRRA(BilinearInterp(dvdmNvrsAdj, mNrmGrid, nNrmGrid), CRRA)
    dvdnFuncAdj = MargValueFuncCRRA(BilinearInterp(dvdnNvrsAdj, mNrmGrid, nNrmGrid), CRRA)
    
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

def solveRiskyContrib(solution_next,ShockDstn,IncShkDstn,RiskyDstn,
                      LivPrb,DiscFac,CRRA,Rfree,PermGroFac,tau,
                      BoroCnstArt,aXtraGrid,nNrmGrid,mNrmGrid,
                      ShareGrid,dGrid,vFuncBool,AdjustPrb,
                      DiscreteShareBool,IndepDstnBool):
    
     # Pack parameters to be passed to stage-specific solvers
     kws = {'ShockDstn': ShockDstn, 'IncShkDstn': IncShkDstn,
            'RiskyDstn': RiskyDstn, 'LivPrb': LivPrb, 'DiscFac': DiscFac,
            'CRRA': CRRA, 'Rfree': Rfree, 'PermGroFac': PermGroFac, 'tau': tau,
            'BoroCnstArt': BoroCnstArt, 'aXtraGrid': aXtraGrid,
            'nNrmGrid': nNrmGrid, 'mNrmGrid': mNrmGrid, 'ShareGrid': ShareGrid,
            'dGrid': dGrid, 'vFuncBool': vFuncBool, 'AdjustPrb': AdjustPrb,
            'DiscreteShareBool': DiscreteShareBool, 'IndepDstnBool': IndepDstnBool}
     
     # Stages of the problem in chronological order
     Stages = ['Reb', 'Sha', 'Cns']
     n_stages = len(Stages)
     # Solvers, indexed by stage names
     Solvers = {'Reb': solveRiskyContribRebStage,
                'Sha': solveRiskyContribShaStage,
                'Cns': solveRiskyContribCnsStage}
     
     # Initialize empty solution
     stageSols = {}
     # Solve stages backwards
     for i in reversed(range(n_stages)):
         stage = Stages[i]
         
         # In the last stage, the next solution is the first stage of the next
         # period. Otherwise, its the next stage of his period.
         if i == n_stages - 1:
             sol_next_stage = solution_next.stageSols[Stages[0]]  
         else:
             sol_next_stage = stageSols[Stages[i+1]]
        
         # Solve
         stageSols[stage] = Solvers[stage](sol_next_stage, **kws)
     
     # Assemble stage solutions into period solution
     periodSol = RiskyContribSolution(**stageSols)
     
     return(periodSol)

# %% Useful parameter sets

# %% Base risky asset dictionary

# Make a dictionary to specify a risky asset consumer type
init_risky = init_idiosyncratic_shocks.copy()
init_risky['RiskyAvg']          = 1.08 # Average return of the risky asset
init_risky['RiskyStd']          = 0.20 # Standard deviation of (log) risky returns
init_risky['RiskyCount']        = 5    # Number of integration nodes to use in approximation of risky returns
init_risky['ShareCount']        = 25   # Number of discrete points in the risky share approximation
init_risky['AdjustPrb']         = 1.0  # Probability that the agent can adjust their risky portfolio share each period
init_risky['DiscreteShareBool'] = False # Flag for whether to optimize risky share on a discrete grid only
init_risky['vFuncBool']         = False

# Adjust some of the existing parameters in the dictionary
init_risky['aXtraMax']        = 100  # Make the grid of assets go much higher...
init_risky['aXtraCount']      = 300  # ...and include many more gridpoints...
init_risky['aXtraNestFac']    = 1    # ...which aren't so clustered at the bottom
init_risky['BoroCnstArt']     = 0.0  # Artificial borrowing constraint must be turned on
init_risky['CRRA']            = 5.0  # Results are more interesting with higher risk aversion
init_risky['DiscFac']         = 0.90 # And also lower patience

# %% Base risky-contrib dictionary

# TODO: these parameters are preliminary and arbitrary!
init_riskyContrib = init_risky.copy()
init_riskyContrib['ShareMax']        = 0.9  # You don't want to put 100% of your wage into pensions.

# Regular grids in m and n
init_riskyContrib['mNrmMin']         = 1e-6
init_riskyContrib['mNrmMax']         = 100
init_riskyContrib['mNrmCount']       = 300
init_riskyContrib['mNrmNestFac']     = 1

init_riskyContrib['nNrmMin']         = 1e-6
init_riskyContrib['nNrmMax']         = 100
init_riskyContrib['nNrmCount']       = 300  
init_riskyContrib['nNrmNestFac']     = 1    

# Number of grid-points for finding the optimal asset rebalance
init_riskyContrib['dCount'] = 20

# Params from the life-cycle agent
init_riskyContrib['AdjustPrb']  = [1.0]
init_riskyContrib['tau']        = [0.1]  # Tax rate on risky asset withdrawals

# TODO: Reduce dimensions while conding the model up
init_riskyContrib['ShareCount']      = 10
init_riskyContrib['aXtraCount']      = 40
init_riskyContrib['nNrmCount']       = 40  #
init_riskyContrib['mNrmCount']       = 45  #
init_riskyContrib['PermShkCount']    = 3  #
init_riskyContrib['TranShkCount']    = 3 
