# FIXME RiskyShareLimitFunc currently doesn't work for time varying CRRA,
# Rfree and Risky-parameters. This should be possible by creating a list of
# functions instead.

import math
import scipy.optimize as sciopt
import scipy.integrate
import scipy.stats as stats

from HARK import Solution, NullFunc, AgentType

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType, IndShockConsumerType, solveConsIndShock, ConsIndShockSolver, ValueFunc, MargValueFunc, ConsumerSolution
from HARK.utilities import approxLognormal, combineIndepDstns
from HARK.interpolation import LinearInterp, LowerEnvelope
from HARK.simulation import drawLognormal

from copy import deepcopy
import numpy as np

#
# def CambellVicApprox()
#
#         # We assume fixed distribution of risky shocks throughout, so we can
#         # calculate the limiting solution once and for all.
#         phi = math.log(self.RiskyAvg/self.Rfree)
#         RiskyAvgSqrd = self.RiskyAvg**2
#         RiskyVar = self.RiskyStd**2
# #
# # mu = math.log(self.RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
# # sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))
# #
# # RiskyShareLimit = phi/(self.CRRA*sigma**2)


def PerfForesightLogNormalPortfolioShare(Rfree, RiskyAvg, RiskyStd, CRRA):
   PortfolioObjective = lambda share: PerfForesightLogNormalPortfolioObjective(share,
                                                                      Rfree,
                                                                      RiskyAvg,
                                                                      RiskyStd,
                                                                      CRRA)
   return sciopt.minimize_scalar(PortfolioObjective, bounds=(0.0, 1.0), method='bounded').x

def PerfForesightDiscretePortfolioShare(Rfree, RiskyDstn, CRRA):
   PortfolioObjective = lambda share: PerfForesightDiscretePortfolioObjective(share,
                                                                      Rfree,
                                                                      RiskyDstn,
                                                                      CRRA)
   return sciopt.minimize_scalar(PortfolioObjective, bounds=(0.0, 1.0), method='bounded').x


# Switch here based on knowledge about risky.
# It can either be "discrete" in which case it is only the number of draws that
# are used, or it can be continuous in which case bounds and a pdf has to be supplied.
def PerfForesightLogNormalPortfolioIntegrand(share, R0, RiskyAvg, sigma, rho):
   muNorm = np.log(RiskyAvg/np.sqrt(1+sigma**2/RiskyAvg**2))
   sigmaNorm = np.sqrt(np.log(1+sigma**2/RiskyAvg**2))
   sharedobjective = lambda r: (R0+share*(r-R0))**(1-rho)
   pdf = lambda r: stats.lognorm.pdf(r, s=sigmaNorm, scale=np.exp(muNorm))

   integrand = lambda r: sharedobjective(r)*pdf(r)
   return integrand

def PerfForesightLogNormalPortfolioObjective(share, R0, RiskyAvg, sigma, rho):
   integrand = PerfForesightLogNormalPortfolioIntegrand(share, R0, RiskyAvg, rho, sigma)
   a = 0.0 # Cannot be negative
   b = 5.0 # This is just an upper bound. pdf should be 0 here.
   return -((1-rho)**-1)*scipy.integrate.quad(integrand, a, b)[0]


def PerfForesightDiscretePortfolioObjective(share, R0, RiskyDstn, rho):

   vals = (R0+share*(RiskyDstn[1]-R0))**(1-rho)
   weights = RiskyDstn[0]

   return -((1-rho)**-1)*np.dot(vals, weights)

class PortfolioSolution(Solution):
    distance_criteria = ['cFunc']

    def __init__(self, cFunc=None, vFunc=None,
                       vPfunc=None, RiskyShareFunc=None, vPPfunc=None,
                       mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None):
        # Change any missing function inputs to NullFunc
        if cFunc is None:
            cFunc = NullFunc()
        if vFunc is None:
            vFunc = NullFunc()
        if RiskyShareFunc is None:
            RiskyShareFunc = NullFunc()
        if vPfunc is None:
            vPfunc = NullFunc()
        if vPPfunc is None:
            vPPfunc = NullFunc()
        self.cFunc        = cFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.RiskyShareFunc = RiskyShareFunc
        # self.vPPfunc      = vPPfunc
        # self.mNrmMin      = mNrmMin
        # self.hNrm         = hNrm
        # self.MPCmin       = MPCmin
        # self.MPCmax       = MPCmax


# These domains are convenient for switching to relavent code paths internally.
# It might be simpler to just pass in vectors instead of DiscreteDomain.
class ContinuousDomain(object):
    def __init__(self, lower, upper, points = [np.nan]):
        if lower > upper:
            raise Exception('lower bounds is larger than upper bound')
        else:
            self.lower = lower
            self.upper = upper
            self.points = points

    def getPoints(self):
        return self.points

    def len(self):
        return len(self.points)

class DiscreteDomain(object):
    def __init__(self, points):
        self.points = points
        self.lower = np.amin(points)
        self.upper = np.amax(points)

    def len(self):
        return len(self.points)

    def getPoints(self):
        return self.points

class PortfolioConsumerType(IndShockConsumerType):

    # We add CantAdjust to the standard set of poststate_vars_ here. We call it
    # CantAdjust over CanAdjust, because this allows us to index into the
    # "CanAdjust = 1- CantAdjust" at all times (it's the 0th offset).
    poststate_vars_ = ['aNrmNow', 'pLvlNow', 'RiskyShareNow', 'CantAdjust']
    time_inv_ = deepcopy(IndShockConsumerType.time_inv_)
    time_inv_ = time_inv_ + ['approxRiskyDstn', 'RiskyCount', 'RiskyShareCount']
    time_inv_ = time_inv_ + ['RiskyShareLimitFunc', 'PortfolioDomain']
    time_inv_ = time_inv_ + ['AdjustPrb', 'PortfolioGrid', 'AdjustCount']

    def __init__(self,cycles=1,time_flow=True,verbose=False,quiet=False,**kwds):

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,
        verbose=verbose,quiet=quiet, **kwds)

        # Check that an adjustment probability is set. If not, default to always.
        if not hasattr(self, 'AdjustPrb'):
            self.AdjustPrb = 1.0
            self.AdjustCount = 1
        elif self.AdjustPrb == 1.0:
            # Always adjust, so there's just one possibility
            self.AdjustCount = 1
        else:
            # If AdjustPrb was set and was below 1.0, there's a chance that
            # the consumer cannot adjust in a given period.
            self.AdjustCount = 2

        if not hasattr(self, 'PortfolioDomain'):
            if self.AdjustPrb < 1.0:
                raise Exception('Please supply a PortfolioDomain when setting AdjustPrb < 1.0.')
            else:
                self.PortfolioDomain = ContinuousDomain(0,1)
        if isinstance(self.PortfolioDomain, DiscreteDomain):
            if self.vFuncBool == False:
                if self.verbose:
                    print('Setting vFuncBool to True to accomodate dicrete portfolio optimization.')

                self.vFuncBool = True
        else:
            if self.AdjustPrb < 1.0:
                raise Exception('Occational inability to re-optimize portfolio (AdjustPrb < 1.0) is currently not possible with continuous choice of the portfolio share.')

        # Now we can set up the PortfolioGrid! This is the portfolio values
        # you can enter the period with. It's exact for discrete , for continuous
        # domain it's the interpolation points.
        self.PortfolioGrid = self.PortfolioDomain.getPoints()

        if self.BoroCnstArt is not 0.0:
            if self.verbose:
                print("Setting BoroCnstArt to 0.0 as this is required by PortfolioConsumerType.")
            self.BoroCnstArt = 0.0



        # Chose specialized solver for Portfolio choice model
        self.solveOnePeriod = solveConsPortfolio

        self.update()

        self.RiskyShareLimitFunc = lambda RiskyDstn: PerfForesightDiscretePortfolioShare(self.Rfree, RiskyDstn, self.CRRA)

    def preSolve(self):
        AgentType.preSolve(self)
        self.updateSolutionTerminal()

    def updateSolutionTerminal(self):
        '''
        Updates the terminal period solution for a portfolio shock consumer.
        Only fills in the consumption function and marginal value function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cFunc_terminal = LinearInterp([0.0, 1.0], [0.0,1.0]) # c=m in terminal period
        vFunc_terminal = LinearInterp([0.0, 1.0], [0.0,0.0]) # This is overwritten
        RiskyShareFunc_terminal = LinearInterp([0.0, 1.0], [0.0,0.0]) # c=m in terminal period

        if isinstance(self.PortfolioDomain, DiscreteDomain):
            PortfolioGridCount = len(self.PortfolioDomain.points)
        else:
            # This should be "PortfolioGridCount" that was set earlier,
            PortfolioGridCount = 1

        cFunc_terminal = PortfolioGridCount*[cFunc_terminal]
        RiskyShareFunc_terminal = PortfolioGridCount*[RiskyShareFunc_terminal]

        vFunc_terminal   = PortfolioGridCount*[ValueFunc(self.cFunc_terminal_,self.CRRA)]
        vPfunc_terminal  = PortfolioGridCount*[MargValueFunc(self.cFunc_terminal_,self.CRRA)]

        # repeat according to number of portfolio adjustment situations
        # TODO FIXME this is technically incorrect, too many in [0]
        cFunc_terminal = self.AdjustCount*[cFunc_terminal]
        RiskyShareFunc_terminal = self.AdjustCount*[RiskyShareFunc_terminal]

        vFunc_terminal   = self.AdjustCount*[vFunc_terminal]
        vPfunc_terminal  = self.AdjustCount*[vPfunc_terminal]

        self.solution_terminal = PortfolioSolution(cFunc = cFunc_terminal,
                                                   RiskyShareFunc = RiskyShareFunc_terminal,
                                                   vFunc = vFunc_terminal,
                                                   vPfunc = vPfunc_terminal,
                                                   mNrmMin=0.0, hNrm=None,
                                                   MPCmin=None, MPCmax=None)


    def getPostStates(self):
        '''
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Calculate post decision ressources
        self.aNrmNow = self.mNrmNow - self.cNrmNow
        self.aLvlNow = self.aNrmNow*self.pLvlNow   # Useful in some cases to precalculate asset level

        # We calculate the risky share given post decision assets aNrmNow. We
        # do this for all agents that have self.CantAdjust == 0 and save the
        # non-adjusters for the next section.
        RiskyShareNow = np.zeros(self.AgentCount) + np.nan

        for t in range(self.T_cycle):
            # We need to take into account whether they have drawn a portfolio
            # adjust shock or not.
            these_adjust = self.CantAdjust == 0
            these_t = t == self.t_cycle

            # First take adjusters
            these = np.logical_and(these_adjust, these_t)
            RiskyShareNow[these] = self.solution[t].RiskyShareFunc[0][0](self.aNrmNow[these]) # should be redefined on mNrm in solve and calculated in getControls

            these_cant_adjust = self.CantAdjust == 1
            these = np.logical_and(these_cant_adjust, these_t)
            RiskyShareNow[these] = self.RiskySharePrev[these] # should be redefined on mNrm in solve and calculated in getControls


        # Store the result in self
        self.RiskyShareNow = RiskyShareNow
        return None


    # Simulation methods
    def getStates(self):
        '''
        Calculates updated values of normalized market resources and permanent income level for each
        agent.  Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        RiskySharePrev = self.RiskyShareNow
        self.RiskySharePrev = RiskySharePrev # Save this for the non-adjusters!
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow

        RfreeNow = self.getRfree()
        # In the current interpretation, everyone gets the same random return.
        # This is because they all invest in the same stock/market index.
        # As a result, we simply draw *one* realization from RiskyDstn.
        RiskyNow = self.getRisky()

        # Calculate the portfolio return from last period to current period.
        RportNow = RfreeNow + RiskySharePrev*(RiskyNow-RfreeNow)

        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income level
        self.PlvlAggNow = self.PlvlAggNow*self.PermShkAggNow # Updated aggregate permanent productivity level
        ReffNow      = RportNow/self.PermShkNow # "Effective" interest factor on normalized assets
        self.bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        self.mNrmNow = self.bNrmNow + self.TranShkNow # Market resources after income

        # Figure out who can adjust their portfolio this period.
        self.CantAdjust = stats.bernoulli.rvs(1-self.AdjustPrb, size=self.AgentCount)

        # New agents are always allowed to optimize their portfolio, because they
        # have no past portfolio to "keep".
        self.CantAdjust[self.new_agents] = 0.0
        return None

    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        self.aNrmNow[which_agents] = drawLognormal(N,mu=self.aNrmInitMean,sigma=self.aNrmInitStd,seed=self.RNG.randint(0,2**31-1))
        pLvlInitMeanNow = self.pLvlInitMean + np.log(self.PlvlAggNow) # Account for newer cohorts having higher permanent income
        self.pLvlNow[which_agents] = drawLognormal(N,mu=pLvlInitMeanNow,sigma=self.pLvlInitStd,seed=self.RNG.randint(0,2**31-1))

        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent is currently in
        self.new_agents = which_agents # store for portfolio choice forced to be allowed in first period
        return None

    def getControls(self):
        '''
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow  = np.zeros(self.AgentCount) + np.nan

        these_cant_adjust = self.CantAdjust == 1
        these_can_adjust = self.CantAdjust == 0

        for t in range(self.T_cycle):
            these_t = t == self.t_cycle
            these = np.logical_and(these_t, these_can_adjust)
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc[0][0].eval_with_derivative(self.mNrmNow[these])

            if any(these_cant_adjust):
                for portfolio_index, portfolio_value in enumerate(self.ShareNow[AdjustIndex]):
                    these_portfolio = np.equal(portfolio_value, self.RiskySharePrev)
                    these = np.logical_and(these_t, these_portfolio)

                    cNrmNow[these], MPCnow[these] = self.solution[t].cFunc[1][portfolio_index].eval_with_derivative(self.mNrmNow[these])

        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        return None

    def getRisky(self):
        return self.drawRiskyFunc()

class ConsIndShockPortfolioSolver(ConsIndShockSolver):
    def __init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                      PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool,
                      approxRiskyDstn, RiskyCount, RiskyShareCount, RiskyShareLimitFunc,
                      AdjustPrb, PortfolioGrid, AdjustCount, PortfolioDomain):

        ConsIndShockSolver.__init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                          PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool)


        self.PortfolioDomain = PortfolioDomain
        if isinstance(self.PortfolioDomain, DiscreteDomain):
            self.DiscreteCase = True
        else:
            self.DiscreteCase = False

        self.AdjustPrb = AdjustPrb
        self.PortfolioGrid = PortfolioGrid
        self.AdjustCount = AdjustCount
        self.ShareNowCount = [1]
        if self.DiscreteCase:
            self.ShareNow = self.PortfolioDomain.getPoints()
            self.ShareNowCount.append(len(self.PortfolioDomain.getPoints()))

        # Store the Risky asset shock distribution
        self.RiskyDstn = approxRiskyDstn(RiskyCount)
        self.RiskyShareLimit = RiskyShareLimitFunc(self.RiskyDstn)

        # Store the number of grid points used approximate the FOC in the port-
        # folio sub-problem.
        self.RiskyShareCount = RiskyShareCount

        self.vFuncsNext = solution_next.vFunc
        self.vPfuncsNext = solution_next.vPfunc

        self.updateShockDstn()
        self.makeRshareGrid()


    def makeEndOfPrdvFunc(self, AdjustIndex, ShareIndex):
        '''
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        if not self.DiscreteCase:
            raise Exception("vFuncBool == True is not supported for continuous portfolio choice.")

        # We will need to index vFuncNext wrt the state next period given choices
        # today.
        VLvlNext            = (self.PermShkVals_temp**(1.0-self.CRRA)*\
                               self.PermGroFac**(1.0-self.CRRA))*self.vFuncsNext[AdjustIndex][ShareIndex](self.mNrmNext[AdjustIndex][ShareIndex])
        EndOfPrdv           = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs       = self.uinv(EndOfPrdv) # value transformed through inverse utility
        # Manually input (0,0) pair
        EndOfPrdvNvrs       = np.insert(EndOfPrdvNvrs,0,0.0)
        aNrm_temp           = np.insert(self.aNrmNow,0,0.0)

        EndOfPrdvNvrsFunc   = LinearInterp(aNrm_temp,EndOfPrdvNvrs)
        self.EndOfPrdvFunc  = ValueFunc(EndOfPrdvNvrsFunc,self.CRRA)


    def makevFunc(self,solution, AdjustIndex, ShareIndex):
        '''
        Creates the value function for this period, defined over market resources m.
        self must have the attribute EndOfPrdvFunc in order to execute.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        vFuncNow : ValueFunc
            A representation of the value function for this period, defined over
            normalized market resources m: v = vFuncNow(m).
        '''
        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp   = self.mNrmMinNow + self.aXtraGrid
        cNrmNow     = solution.cFunc[AdjustIndex][ShareIndex](mNrm_temp)
        aNrmNow     = mNrm_temp - cNrmNow
        vNrmNow     = self.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow)

        # Construct the beginning-of-period value function
        vNvrs        = self.uinv(vNrmNow) # value transformed through inverse utility
        # Manually insert (0,0) pair.
        mNrm_temp    = np.insert(mNrm_temp,0,0.0) # np.insert(mNrm_temp,0,self.mNrmMinNow)
        vNvrs        = np.insert(vNvrs,0,0.0)
        vNvrsFuncNow = LinearInterp(mNrm_temp,vNvrs)
        vFuncNow     = ValueFunc(vNvrsFuncNow,self.CRRA)
        return vFuncNow

    def addvFunc(self,solution):
        '''
        Creates the value function for this period and adds it to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        '''
        if not self.DiscreteCase:
            raise Exception('You\'re not supposed to be here. Continuous choice portfolio domain does not support vFuncBool == True or AdjustPrb < 1.0.')

        vFunc = self.AdjustCount*[[]]

        for AdjustIndex in range(self.AdjustCount): # nonadjuster possible!
            # this is where we add to vFunc based on non-adjustment.
            # Basically repeat the above with the share updated to be the "prev"
            # share an. We need to keep mNrmNext at two major indeces: adjust and
            # non-adjust. Adjust will just have one element, but non-adjust will need
            # one for each of the possible current ("prev") values.
            for ShareIndex in range(self.ShareNowCount[AdjustIndex]): # for all share level indeces in the adjuster (1) case
                self.makeEndOfPrdvFunc(AdjustIndex, ShareIndex)
                vFunc[AdjustIndex].append(self.makevFunc(solution, AdjustIndex, ShareIndex))

        solution.vFunc = vFunc
        return solution

    def updateShockDstn(self):
        self.ShockDstn = combineIndepDstns(self.IncomeDstn, self.RiskyDstn)

    def makeRshareGrid(self):
        # We set this up such that attempts to use RshareGrid will fail hard
        # if we're in the discrete case
        if not self.DiscreteCase:
            self.RshareGrid = np.linspace(0, 1, self.RiskyShareCount)
            return self.RshareGrid
        return []

    def prepareToCalcRiskyShare(self):
        if self.DiscreteCase:
            self.prepareToCalcRiskyShareDiscrete()
        else:
            self.prepareToCalcRiskyShareContinuous()

    def prepareToCalcRiskyShareContinuous(self):
        # Hard restriction on aNrm. We'd need to define more elaborate model
        # specifics if a could become negative (or a positive return shock
        # would make you worse off!)
        aNrmPort = self.aXtraGrid[self.aXtraGrid >= 0]

        self.aNrmPort = aNrmPort
        RshareGrid = self.makeRshareGrid()
        self.RshareNow = np.array([])
        vHatP = np.zeros((len(aNrmPort), len(RshareGrid)))

        # Evaluate the non-constant part of the first order conditions wrt the
        # portfolio share. This requires the implied resources tomorrow given
        # todays shocks to be evaluated.
        i_a = 0
        for a in aNrmPort:
            # for all possible a's today
            i_s = 0
            for s in RshareGrid:
                Rtilde = self.RiskyShkValsNext - self.Rfree
                Reff = self.Rfree + Rtilde*s
                mNext = a*Reff/(self.PermGroFac*self.PermShkValsNext) + self.TranShkValsNext
                vHatP_a_s = Rtilde*self.PermShkValsNext**(-self.CRRA)*self.vPfuncNext(mNext)
                vHatP[i_a, i_s] = np.dot(vHatP_a_s, self.ShkPrbsNext)
                i_s += 1
            i_a += 1

        self.vHatP = vHatP

    def prepareToCalcRiskyShareDiscrete(self):
        # Hard restriction on aNrm. We'd need to define more elaborate model
        # specifics if a could become negative (or a positive return shock
        # would make you worse off!)
        aNrmPort = self.aXtraGrid[self.aXtraGrid >= 0]

        self.aNrmPort = aNrmPort
        RshareGrid = self.ShareNow
        self.RshareNow = np.array([])
        vHat = np.zeros((len(aNrmPort), len(RshareGrid)))

        # Evaluate the non-constant part of the first order conditions wrt the
        # portfolio share. This requires the implied resources tomorrow given
        # todays shocks to be evaluated.
        i_a = 0
        for a in aNrmPort:
            # for all possible a's today
            i_s = 0
            for s in RshareGrid:
                Rtilde = self.RiskyShkValsNext - self.Rfree
                Reff = self.Rfree + Rtilde*s
                mNrmNext = a*Reff/(self.PermGroFac*self.PermShkValsNext) + self.TranShkValsNext


                VLvlNext = (self.PermShkValsNext**(1.0-self.CRRA)*\
                            self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(mNrmNext)
                vHat_a_s = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbsNext,axis=0)

                vHat[i_a, i_s] = vHat_a_s
                i_s += 1
            i_a += 1

        self.vHat = vHat

    def calcRiskyShare(self):
        if self.DiscreteCase:
            RiskyShareFunc = self.calcRiskyShareDiscrete()
        else:
            RiskyShareFunc = self.calcRiskyShareContinuous()

        return RiskyShareFunc


    def calcRiskyShareContinuous(self):
        # This should be fixed by an insert 0
        aGrid = np.array([0.0,])
        Rshare = np.array([1.0,])

        i_a = 0
        for a in self.aNrmPort:
            aGrid = np.append(aGrid, a)
            if self.vHatP[i_a, -1] >= 0.0:
                Rshare = np.append(Rshare, 1.0)
            elif self.vHatP[i_a, 0] < 0.0:
                Rshare = np.append(Rshare, 0.0)
            else:
                residual = LinearInterp(self.RshareGrid, self.vHatP[i_a, :])
                zero  = sciopt.fsolve(residual, Rshare[-1])
                Rshare = np.append(Rshare, zero)
            i_a += 1
        RiskyShareFunc = LinearInterp(aGrid, Rshare,intercept_limit=self.RiskyShareLimit, slope_limit=0) # HAVE to specify the slope limit
        return RiskyShareFunc

    def calcRiskyShareDiscrete(self):
        # Based on the end-of-period value function, we calculate the best
        # choice today for a range of a values (those given in aNrmPort).

        # Should just use insert below ( at 0)
        aGrid = np.array([0.0,])
        Rshare = np.array([1.0,]) # is it true for AdjustPrb < 1?

        i_a = 0
        # For all positive aNrms
        for a in self.aNrmPort:
            # all values at portfolio shares should be calculated
            # argmax gives optimal portfolio
            share_argmax = np.argmax(self.vHat[i_a, :])
            Rshare = np.append(Rshare, self.ShareNow[share_argmax])
            i_a += 1

        # TODO FIXME find limiting share for perf foresight
        RiskyShareFunc = scipy.interpolate.interp1d(np.insert(self.aNrmPort, 0, 0.0), Rshare, kind='zero',bounds_error=False, fill_value=Rshare[-1])
        return RiskyShareFunc

    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period. This method adds extra steps because it first
        solves the portfolio problem given the end-of-period assets to be able
        to get next period resources.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        '''

        # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) uconstrained consumption function, and the artificially con-
        # strained consumption function.
        aNrmNow     = np.asarray(self.aXtraGrid)
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.tile(aNrmNow,(ShkCount,1))

        # Tile arrays of the income shocks and put them into useful shapes
        aNrmCount         = aNrmNow.shape[0]
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aNrmCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aNrmCount,1))).transpose()
        RiskyShkVals_temp  = (np.tile(self.RiskyShkValsNext,(aNrmCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aNrmCount,1))).transpose()

        mNrmNext = self.AdjustCount*[[]]
        for AdjustIndex in range(self.AdjustCount):
            for ShareIndex in range(self.ShareNowCount[AdjustIndex]):
            # Calculate share at current aNrm. If non-adjusting, it's just
            # self.RiskySharePrev, else we use the recently calculated RiskyShareFunc
            # First generate mNrmNext for adjusters
                if AdjustIndex == 0: # adjust
                    sAt_aNrm = self.RiskyShareFunc(aNrmNow)
                # Then generate for non-adjusters
                else: # non-adjuster
                    sAt_aNrm = self.ShareNow[ShareIndex]

                # Get cash on hand next period.
                # Compose possible return factors
                self.Rtilde = RiskyShkVals_temp - self.Rfree
                # Combine into effective returns factors, taking into account the share
                self.Reff = (self.Rfree + self.Rtilde*sAt_aNrm)
                # Apply the permanent growth factor and possible permanent shocks
                mNrmPreTran = self.Reff/(self.PermGroFac*PermShkVals_temp)*aNrm_temp
                # Add transitory income
                mNrmNext[AdjustIndex].append(mNrmPreTran + TranShkVals_temp)

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext
        self.aNrmNow           = aNrmNow
        return aNrmNow

    def calcEndOfPrdvP(self):
        '''
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        '''

        EndOfPrdvP = self.AdjustCount*[[]]
        for AdjustIndex in range(self.AdjustCount):
            for ShareIndex in range(self.ShareNowCount[AdjustIndex]):
                mNrmNext = self.mNrmNext[AdjustIndex][ShareIndex]
                EndOfPrdvP[AdjustIndex].append(np.sum(self.DiscFacEff*self.Reff*self.PermGroFac**(-self.CRRA)*
                              self.PermShkVals_temp**(-self.CRRA)*
                              self.vPfuncNext(mNrmNext)*self.ShkPrbs_temp,axis=0))

        return EndOfPrdvP



    def setAndUpdateValues(self,solution_next,IncomeDstn,LivPrb,DiscFac):
        '''
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        '''
        self.DiscFacEff       = DiscFac*LivPrb # "effective" discount factor
        self.ShkPrbsNext  = self.ShockDstn[0] # but ConsumtionSolver doesn't store the risky shocks
        self.PermShkValsNext  = self.ShockDstn[1] # but ConsumtionSolver doesn't store the risky shocks
        self.TranShkValsNext  = self.ShockDstn[2] # but ConsumtionSolver doesn't store the risky shocks
        self.RiskyShkValsNext  = self.ShockDstn[3] # but ConsumtionSolver doesn't store the risky shocks
        self.PermShkMinNext   = np.min(self.PermShkValsNext)
        self.TranShkMinNext   = np.min(self.TranShkValsNext)
        self.vPfuncNext       = solution_next.vPfunc
        self.WorstIncPrb      = np.sum(self.ShkPrbsNext[
                                (self.PermShkValsNext*self.TranShkValsNext)==
                                (self.PermShkMinNext*self.TranShkMinNext)])

        if self.CubicBool:
            self.vPPfuncNext  = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext    = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        # self.PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        # self.MPCminNow    = 1.0/(1.0 + self.PatFac/solution_next.MPCmin)
        # self.ExIncNext    = np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext)
        # self.hNrmNow      = self.PermGroFac/self.Rfree*(self.ExIncNext + solution_next.hNrm)
        # self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))*
        #                                 self.PatFac/solution_next.MPCmax)




    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        '''
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = 0.0 #(self.solution_next.mNrmMin - self.TranShkMinNext)*\
                           #(self.PermGroFac*self.PermShkMinNext)/self.Rfree

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat,BoroCnstArt])

        # Can we just put in *some value* for MPCmaxNow here?
        # if self.BoroCnstNat < self.mNrmMinNow:
        #     self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        # else:
        #     self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(np.array([0.0, 1.0]),
                                         np.array([0.0, 1.0]))


    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income and
        a portfolio choice over a riskless and a risky asset.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        '''

        # TODO FIXME
        # This code is a mix of looping over states ( for the Risky Share funcs)
        # and implicit looping such as in prepareToCalcEndOfPrdvP. I think it would
        # be best to simply have a major loop here, and keep the methods atomic.

        cFuncs = self.AdjustCount*[[]]
        vPfuncs = self.AdjustCount*[[]]
        RiskyShareFuncs = self.AdjustCount*[[]]

        for AdjustIndex in range(self.AdjustCount):
            for PortfolioGridIdx in range(self.ShareNowCount[AdjustIndex]):
                if self.DiscreteCase:
                    self.vFuncNext = self.vFuncsNext[AdjustIndex][PortfolioGridIdx]

                self.vPfuncNext = self.vPfuncsNext[AdjustIndex][PortfolioGridIdx]

                self.PortfolioGridIdx = PortfolioGridIdx

                # If it can adjust, solve the first sub-problem: the portfolio choice
                if AdjustIndex == 0:
                    self.prepareToCalcRiskyShare()
                    self.RiskyShareFunc = self.calcRiskyShare()
                else:
                    self.RiskyShareFunc = lambda a: PortfolioGrid[PortfolioGridIdx]

                RiskyShareFuncs[AdjustIndex].append(self.RiskyShareFunc)


        # Then solve the consumption choice given optimal portfolio choice
        aNrm = self.prepareToCalcEndOfPrdvP()

        # Seems like this could already have been done above? TODO:Calculate
        # and save both and compare.
        EndOfPrdvP = self.calcEndOfPrdvP()

        # Todo!
        self.cFuncLimitIntercept = None
        self.cFuncLimitSlope = None

        # Generate all the solutions
        for AdjustIndex in range(self.AdjustCount): # stupid name, should be related to adjusting
            for PortfolioGridIdx in range(self.ShareNowCount[AdjustIndex]):
                cs_solution = self.makeBasicSolution(EndOfPrdvP[AdjustIndex][PortfolioGridIdx],aNrm,self.makeLinearcFunc)
                cFuncs[AdjustIndex].append(cs_solution.cFunc)
                vPfuncs[AdjustIndex].append(cs_solution.vPfunc)
                # This is a good place to make it defined at m!!!

        # solution   = self.addMPCandHumanWealth(solution)
        solution = PortfolioSolution(cFunc=cFuncs,
                                     vPfunc=vPfuncs,
                                     RiskyShareFunc=RiskyShareFuncs)

        if self.vFuncBool:
            solution = self.addvFunc(solution)

        return solution


# The solveOnePeriod function!

def solveConsPortfolio(solution_next, IncomeDstn, LivPrb, DiscFac,
                       CRRA, Rfree, PermGroFac, BoroCnstArt,
                       aXtraGrid, vFuncBool, CubicBool, approxRiskyDstn,
                       RiskyCount, RiskyShareCount, RiskyShareLimitFunc,
                       AdjustPrb, PortfolioGrid, AdjustCount, PortfolioDomain):


    # construct solver instance
    solver = ConsIndShockPortfolioSolver(solution_next, IncomeDstn, LivPrb,
                                         DiscFac, CRRA, Rfree, PermGroFac,
                                         BoroCnstArt, aXtraGrid, vFuncBool,
                                         CubicBool, approxRiskyDstn, RiskyCount,
                                         RiskyShareCount, RiskyShareLimitFunc,
                                         AdjustPrb, PortfolioGrid, AdjustCount,
                                         PortfolioDomain)

    # Do some preparatory work
    solver.prepareToSolve()

    # Solve and return solution
    portsolution = solver.solve()

    return portsolution

def RiskyDstnFactory(RiskyAvg=1.0, RiskyStd=0.0):
    RiskyAvgSqrd = RiskyAvg**2
    RiskyVar = RiskyStd**2

    mu = math.log(RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
    sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))

    return lambda RiskyCount: approxLognormal(RiskyCount, mu=mu, sigma=sigma)

def RiskyDrawFactory(RiskyAvg=1.0, RiskyStd=0.0):
    RiskyAvgSqrd = RiskyAvg**2
    RiskyVar = RiskyStd**2

    mu = math.log(RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
    sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))

    return lambda: drawLognormal(1, mu=mu, sigma=sigma)

def LogNormalRiskyDstnDraw(RiskyAvg=1.0, RiskyStd=0.0):
    RiskyAvgSqrd = RiskyAvg**2
    RiskyVar = RiskyStd**2

    mu = math.log(RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
    sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))

    return drawLognormal(1, mu=mu, sigma=sigma)

class LogNormalPortfolioConsumerType(PortfolioConsumerType):

#    time_inv_ = PortfolioConsumerType.time_inv_ + ['approxRiskyDstn', 'RiskyCount', 'RiskyShareCount', 'RiskyShareLimitFunc', 'AdjustPrb', 'PortfolioGrid']

    def __init__(self,cycles=1,time_flow=True,verbose=False,quiet=False,**kwds):

        PortfolioConsumerType.__init__(self,cycles=cycles, time_flow=time_flow,
                                      verbose=verbose, quiet=quiet, **kwds)

        self.approxRiskyDstn = RiskyDstnFactory(RiskyAvg=self.RiskyAvg,
                                                RiskyStd=self.RiskyStd)
        # Needed to simulate. Is a function that given 0 inputs it returns a draw
        # from the risky asset distribution. Only one is needed, because everyone
        # draws the same shock.
        self.drawRiskyFunc = RiskyDrawFactory(RiskyAvg=self.RiskyAvg,
                                                 RiskyStd=self.RiskyStd)

        self.RiskyShareLimitFunc = lambda _: PerfForesightLogNormalPortfolioShare(self.Rfree, self.RiskyAvg, self.RiskyStd, self.CRRA)

def matchPortfolio(portType, moments):
    """
    Match a portfolio model to a set of moments. Requires that the simulations
    produce `RiskyShareNow`.
    """
    return portType
#    optimize...




# def ConsIndShockPortfolioSolver(ConsIndShockSolver):
#
# #    def setAndUpdateValues(self):
#         # great but wait
#         #   calc sUnderbar -> mertonsammuelson
#         #   calc MPC kappaUnderbar
#         #   calc human wealth
#

####################
####################
####################
####################

###
