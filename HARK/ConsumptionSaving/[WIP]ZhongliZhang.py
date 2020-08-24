import numpy as np
from HARK.core import AgentType
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, ValueFunc, MargValueFunc, ConsumerSolution
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distribution import *
from HARK.interpolation import *
from HARK.utilities import *


class LaborSearchConsumerType(MarkovConsumerType):
    '''
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    When he is unemployed, he will decide his probability of becoming re-employed. When he
    is employed, he is influenced by permanent and trasnsitory shock and exogenous probability
    of becoming unemployed.
    '''
    def __init__(self, cycles=1, **kwds):
        '''
        Instantiate a new consumer type with given data.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        '''
        MarkovConsumerType.__init__(cycles=cycles, **kwds)
        self.solveOnePeriod = self.solveConsLaborSearch

    def checkMarkovInputs(self):
        '''
        This overwrites a method of the parent class that would not work as intended in this model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        pass

    def updatePermGroFac(self):
        '''
        Construct the attribute PermGroFac as a time-varying. It should have the length of unemployment state + 1.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        PermGroFac = list()
        for j in range(0, self.T_cycle):
            PermGroFac.append([])
            PermGroFac[j].append(self.PermGroFacEmp[j])
            for n in range(0, len(self.StateCount)):
                PermGroFac[j].append(self.PermGroFacUnemp[n])
            PermGroFac[j] = np.array(PermGroFac[j])

        self.PermGroFac = PermGroFac.copy()
        self.addToTimeVary('PermGroFac')

    def update(self):
        '''
        Update the income process, the assets grid, the terminal solution, and growth factor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        MarkovConsumerType.update(self)
        self.updatePermGroFac()

    def updateIncomeProcess(self):
        '''
        constructs the attributes IncomeDstn, PermShkDstn, and TranShkDstn using the primitive attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IncomeDstn = list()
        TranShkDstn = list()
        PermShkDstn = list()

        IncUnemp_all = self.IncUnemp.copy()
        UnempPrb = self.UnempPrb

        self.IncUnemp = list()
        for i in range(0, len(IncUnemp_all)):
            self.IncUnemp.append(0)
        self.UnempPrb = 0

        IndShockConsumerType.updateIncomeProcess(self)
        IncomeDstn_emp = self.IncomeDstn
        PermShkDstn_emp = self.PermShkDstn
        TranShkDstn_emp = self.TranShkDstn

        degnrteProb = np.array([1])
        PermShkDstn_unemp = DiscreteDistribution(degnrteProb, np.array([1]))
        TranShkDstn_unemp = list()
        IncomeDstn_by_unemp = list()
        for n in range(0, self.StateCount):
            degnrteX = np.array(IncUnemp_all[n])
            TranShkDstn_unemp.append(DiscreteDistribution(degnrteProb, degnrteX))
            instance = combineIndepDstns(PermShkDstn_unemp, DiscreteDistribution(degnrteProb, degnrteX))
            IncomeDstn_by_unemp.append(instance)
            del instance, degnrteX

        for j in range(0, self.T_cycle):
            IncomeDstn.append([])
            TranShkDstn.append([])
            PermShkDstn.append([])
            IncomeDstn[j].append(IncomeDstn_emp[j])
            TranShkDstn[j].append(TranShkDstn_emp[j])
            PermShkDstn[j].append(PermShkDstn_emp[j])
            for n in range(0, len(IncomeDstn_by_unemp)):
                IncomeDstn[j].append(IncomeDstn_by_unemp[n])
                TranShkDstn[j].append(TranShkDstn_unemp[n])
                PermShkDstn[j].append(PermShkDstn_unemp)

        self.TranShkDstn = TranShkDstn.copy()
        self.IncomeDstn = IncomeDstn.copy()
        self.PermShkDstn = PermShkDstn.copy()

        self.addToTimeVary('TranShkDstn', 'PermShkDstn', 'IncomeDstn')

        self.IncUnemp = IncUnemp_all.copy()
        self.UnempPrb = UnempPrb.copy()

    def preSolve(self):
        '''
        Check to make sure that the inputs that are specific to LaborSearchConsumerType
        are of the right shape (if arrays) or length (if lists). If they are of right shape,
        then update the parameters used for solver function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        AgentType.preSolve(self)
        self.update()

    def solveConsLaborSearch(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt,
                             SearchCost0, SearchCost1, aXtraGrid, UnempPrb):
        '''
        Solver function that solves for one period.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one-period problem.
        IncomeDstn: [[np.array]]
            A length N list of income distributions in each succeeding unemployment state.
            Each income distribution contains three arrays of floats,representing a discrete
            approximation to the income process at the beginning of the succeeding period.
            Order: event probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : list
            Expected permanent income growth factor in each unemployment state.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        SearchCost0: float
            One of the cost of employment parameter at this age.
        SearchCost1: float
            The other cost of employment parameter at this age.
        aXtraGrid: np.array
            Gridpoints represents “normalized assets above minimum”, at which
            the model will be solved by the endogenous grid method.
        UnempPrb: float
            The exogenous probability of being unemployment at this age.

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''
        # 1
        u = lambda c , s: CRRAutility(c, CRRA) - SearchCost0 * s ** SearchCost1
        u_ = lambda c: CRRAutility(c, CRRA)  
        u_inv = lambda c: CRRAutility_inv(c, CRRA)
        u_prime = lambda u: CRRAutilityP(u, CRRA)
        u_prime_inv = lambda uP: CRRAutilityP_inv(uP, CRRA)
        # 2
        EndOfPrdvFunc_by_state_next = list()
        EndOfPrdvPfunc_by_state_next = list()
        # 3
        BoroCnstNat = list()
        for i in range(self.StateCount):
            BoroCnstNat.append(np.nan)
        BoroCnstNat = np.array(BoroCnstNat)
        # 4
        for n in range(self.StateCount):
            # a
            shkProb = IncomeDstn[n].pmf
            permVal = IncomeDstn[n].X[0]
            tranVal = IncomeDstn[n].X[1]
            # b
            shkProb_tiled = np.tile(shkProb, (aXtraGrid.size, 1))
            permVal_tiled = np.tile(permVal, (aXtraGrid.size, 1))
            tranVal_tiled = np.tile(tranVal, (aXtraGrid.size, 1))
            # c
            permVal_min = np.min(permVal_tiled)
            tranVal_min = np.min(tranVal_tiled)
            Rinv = 1 / Rfree
            BoroCnstNat[n] = (solution_next.mNrmMin[n] - tranVal_min) * (PermGroFac[n] * permVal_min) * Rinv
            # d
            aNrmNowGrid = aXtraGrid + BoroCnstNat[n]
            nums = shkProb_tiled.size
            aNrmNowGrid_tiled = np.tile(aNrmNowGrid, (nums, 1))
            aNrmNowGrid_tiled = aNrmNowGrid_tiled.transpose()
            # e
            mNrmNext = Rfree / (PermGroFac[n] * permVal_tiled) * aNrmNowGrid_tiled + tranVal_tiled
            # f
            vFunc = solution_next.vFunc[n](mNrmNext)
            vPfunc = solution_next.vPfunc[n](mNrmNext)
            # g
            EndOfPrdv = DiscFac * LivPrb * PermGroFac[n] ** (1 - CRRA) * \
                        np.sum(tranVal_tiled ** (1 - CRRA) * vFunc * shkProb_tiled, axis=1)
            EndOfPrdvP = DiscFac * LivPrb * PermGroFac[n] ** (-CRRA) * Rfree * \
                         np.sum(permVal_tiled ** (-CRRA) * vPfunc * shkProb_tiled, axis=1)
            # h
            EndOfPrdvNvrs_cond = u_inv(EndOfPrdv)
            EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond, 0, 0.0)
            EndOfPrdvPnvrs_cond = u_prime_inv(EndOfPrdvP)
            EndOfPrdvPnvrs_cond = np.insert(EndOfPrdvPnvrs_cond, 0, 0.0)
            # i
            aNrmNowGrid = np.insert(aNrmNowGrid, 0, BoroCnstNat[n])
            EndOfPrdvNvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvNvrs_cond)
            EndOfPrdvPnvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvPnvrs_cond)
            # j
            EndOfPrdvFunc_cond = ValueFunc(EndOfPrdvNvrsFunc_cond, CRRA)
            EndOfPrdvPfunc_cond = MargValueFunc(EndOfPrdvPnvrsFunc_cond, CRRA)
            # k
            EndOfPrdvFunc_by_state_next.append(EndOfPrdvFunc_cond)
            EndOfPrdvPfunc_by_state_next.append(EndOfPrdvPfunc_cond)
        # 5
        cFunc_by_state = list()
        sFunc_by_state = list()
        vFunc_by_state = list()
        vPfunc_by_state = list()
        # 6
        mNrmMinNow = list()
        for n in range(1, self.StateCount):
            # a
            if BoroCnstArt != None:
                if BoroCnstArt > BoroCnstNat[n]:
                    mNrmMinNow[n] = BoroCnstArt
                else:
                    mNrmMinNow[n] = BoroCnstNat[n]
            else:
                mNrmMinNow[n] = BoroCnstNat[n]
            # b
            aNrmGrid_temp = aXtraGrid + BoroCnstNat[n]
            if BoroCnstArt != None:
                for i in range(aNrmGrid_temp.size):
                    if aNrmGrid_temp[i] >= BoroCnstArt:
                        break
                aNrmGrid_temp = np.insert(aNrmGrid_temp, i, BoroCnstArt)
            EndOfPrdv_unemp = EndOfPrdvFunc_by_state_next[n](aNrmGrid_temp)
            EndOfPrdvP_unemp = EndOfPrdvPfunc_by_state_next[n](aNrmGrid_temp)
            # c
            EndOfPrdv_emp = EndOfPrdvFunc_by_state_next[0](aNrmGrid_temp)
            EndOfPrdvP_emp = EndOfPrdvPfunc_by_state_next[0](aNrmGrid_temp)
            # d
            sGrid_temp = ((EndOfPrdv_emp - EndOfPrdv_unemp) / (SearchCost0 * SearchCost1)) ** (1 / (SearchCost1 - 1))
            cNrmGrid_temp = (sGrid_temp * EndOfPrdvP_emp + (1 - sGrid_temp) * EndOfPrdvP_unemp) ** (-1 / CRRA)
            # f
            mNrmGrid_temp = aNrmGrid_temp + cNrmGrid_temp
            mNrmGrid_temp = np.insert(mNrmGrid_temp, 0, BoroCnstNat)
            cNrmGrid_temp = np.insert(cNrmGrid_temp, 0, 0.0)
            sGrid_temp = np.insert(sGrid_temp, 0, 0.0)
            cFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, cNrmGrid_temp)
            sFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, sGrid_temp)
            # g
            cFunc_cnst = LinearInterp(np.array([mNrmMinNow[n], mNrmMinNow[n] + 1]), np.array([0.0, 1.0]))
            # h
            cFunc = LowerEnvelope(cFuncUnc_at_this_state, cFunc_cnst)
            cFunc_by_state.append(cFunc)
            # i
            if BoroCnstArt != None:
                s_rplce = sFuncUnc_at_this_state(BoroCnstArt)
                for i in range(sGrid_temp.size):
                    if aNrmGrid_temp[i] <= BoroCnstArt:
                        sGrid_temp = s_rplce
            # j
            sFunc = LinearInterp(mNrmGrid_temp, sGrid_temp)
            sFunc_by_state.append(sFunc)
            # k
            vPfunc = MargValueFunc(cFunc_by_state[n], CRRA)
            vPfunc_by_state.append(vPfunc)
            # l
            mGrid_temp = aXtraGrid + mNrmMinNow[n]
            cVal = cFunc_by_state[n](mGrid_temp)
            sVal = sFunc_by_state[n](mGrid_temp)
            aVal = mGrid_temp - cVal
            UnempVal = EndOfPrdvFunc_by_state_next[n](aVal)
            EmpVal = EndOfPrdvFunc_by_state_next[0](aVal)
            vFuncVal = u(cVal, sVal) + sVal * EmpVal + (1 - sVal) * UnempVal
            # m
            vFuncNrvs = u_inv(vFuncVal)
            vFuncNrvs = np.insert(vFuncNrvs, 0, 0.0)
            mGrid_temp = np.insert(mGrid_temp, 0, mNrmMinNow[n])
            vFuncNrvs_func = LinearInterp(mGrid_temp, vFuncNrvs)
            vFunc = ValueFunc(vFuncNrvs_func, CRRA)
            vFunc_by_state.append(vFunc)

        # 7
        if BoroCnstArt != None:
            if BoroCnstArt > BoroCnstNat[0]:
                mNrmMinNow = np.insert(mNrmMinNow, 0, BoroCnstArt)
            else:
                mNrmMinNow = np.insert(mNrmMinNow, 0, BoroCnstNat[0])
        else:
            mNrmMinNow = np.insert(mNrmMinNow, 0, BoroCnstNat[0])
        mNrmMinNow_0 = mNrmMinNow[0]

        EndOfPrdvP_emp_0 = EndOfPrdvFunc_by_state_next[0](aXtraGrid + BoroCnstNat[0])
        EndOfPrdvP_unemp_0 = EndOfPrdvPfunc_by_state_next[1](aXtraGrid + BoroCnstNat[0])
        cNrmGrid_temp_0 = ((1 - UnempPrb) * EndOfPrdvP_emp_0 + UnempPrb * EndOfPrdvP_unemp_0) ** (-1 / CRRA)
        mNrmGrid_temp_0 = aXtraGrid + BoroCnstNat[0] + cNrmGrid_temp_0
        cFunc_0 = LinearInterp(mNrmGrid_temp_0, cNrmGrid_temp_0)
        cFunc_0 = LowerEnvelope(cFunc_0, LinearInterp(np.array([mNrmMinNow_0, mNrmMinNow_0 + 1]), np.array([0.0, 1.0])))
        cFunc_by_state = np.insert(cFunc_by_state, 0, cFunc_0)

        # search function is constant at zero
        sFunc_0 = LinearInterp(np.array([0.0, np.max(mNrmGrid_temp_0)]), np.array([0.0, 0.0]))
        sFunc_by_state = np.insert(sFunc_by_state, 0, sFunc_0)

        vPfunc_0 = MargValueFunc(cFunc_by_state[0], CRRA)
        vPfunc_by_state = np.insert(vPfunc_by_state, 0, vPfunc_0)

        mGrid_0 = aXtraGrid + mNrmMinNow_0
        cVal_0 = cFunc_by_state[0](mGrid_0)
        UnempVal_0 = EndOfPrdvFunc_by_state_next[1](mGrid_0 - cVal_0)
        EmpVal_0 = EndOfPrdvFunc_by_state_next[0](mGrid_0 - cVal_0)
        vFuncVal_0 = u_(mGrid_0 - cVal_0) + (1 - UnempPrb) * EmpVal_0 + UnempPrb * UnempVal_0
        vFuncNvrs_0 = u_inv(vFuncVal_0)
        vFuncNvrs_0 = np.insert(vFuncNvrs_0, 0, 0.0)
        mGrid_0 = np.insert(mGrid_0, 0, mNrmMinNow_0)
        vFuncNrvs_func_0 = LinearInterp(mGrid_0, vFuncNvrs_0)
        vFunc_0 = ValueFunc(vFuncNrvs_func_0, CRRA)
        vFunc_by_state = np.insert(vFunc_by_state, 0, vFunc_0)

        solution_now = ConsumerSolution()
        solution_now.vFunc = vFunc_by_state
        solution_now.vPfunc = vPfunc_by_state
        solution_now.cFunc = cFunc_by_state
        solution_now.sFunc = sFunc_by_state
        solution_now.mNrmMin = mNrmMinNow

        return solution_now
