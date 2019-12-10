'''
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
'''
from copy import deepcopy
import numpy as np
from HARK import AgentType
from HARK.ConsumptionSaving.ConsIndShockModel import ConsIndShockSolver, ValueFunc, \
                             MargValueFunc, ConsumerSolution, IndShockConsumerType
from HARK.simulation import drawDiscrete, drawUniform
from HARK.interpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARK.utilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, \
                           CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class ConsMarkovSolver(ConsIndShockSolver):
    '''
    A class to solve a single period consumption-saving problem with risky income
    and stochastic transitions between discrete states, in a Markov fashion.
    Extends ConsIndShockSolver, with identical inputs but for a discrete
    Markov state, whose transition rule is summarized in MrkvArray.  Markov
    states can differ in their interest factor, permanent growth factor, live probability, and
    income distribution, so the inputs Rfree, PermGroFac, IncomeDstn, and LivPrb are
    now arrays or lists specifying those values in each (succeeding) Markov state.
    '''
    def __init__(self,solution_next,IncomeDstn_list,LivPrb,DiscFac,
                      CRRA,Rfree_list,PermGroFac_list,MrkvArray,BoroCnstArt,
                      aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for a one period problem with risky income
        and transitions between discrete Markov states.  In the descriptions below,
        N is the number of discrete states.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn_list : [[np.array]]
            A length N list of income distributions in each succeeding Markov
            state.  Each income distribution contains three arrays of floats,
            representing a discrete approximation to the income process at the
            beginning of the succeeding period. Order: event probabilities,
            permanent shocks, transitory shocks.
        LivPrb : np.array
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period for each Markov state.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree_list : np.array
            Risk free interest factor on end-of-period assets for each Markov
            state in the succeeding period.
        PermGroFac_list : np.array
            Expected permanent income growth factor at the end of this period
            for each Markov state in the succeeding period.
        MrkvArray : np.array
            An NxN array representing a Markov transition matrix between discrete
            states.  The i,j-th element of MrkvArray is the probability of
            moving from state i in period t to state j in period t+1.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.

        Returns
        -------
        None
        '''
        # Set basic attributes of the problem
        ConsIndShockSolver.assignParameters(self,solution_next,np.nan,LivPrb,DiscFac,CRRA,np.nan,
                                            np.nan,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defUtilityFuncs()

        # Set additional attributes specific to the Markov model
        self.IncomeDstn_list      = IncomeDstn_list
        self.Rfree_list           = Rfree_list
        self.PermGroFac_list      = PermGroFac_list
        self.MrkvArray            = MrkvArray
        self.StateCount           = MrkvArray.shape[0]

    def solve(self):
        '''
        Solve the one period problem of the consumption-saving model with a Markov state.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a marg-
            inal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        '''
        # Find the natural borrowing constraint in each current state
        self.defBoundary()

        # Initialize end-of-period (marginal) value functions
        self.EndOfPrdvFunc_list  = []
        self.EndOfPrdvPfunc_list = []
        self.ExIncNextAll        = np.zeros(self.StateCount) + np.nan # expected income conditional on the next state
        self.WorstIncPrbAll      = np.zeros(self.StateCount) + np.nan # probability of getting the worst income shock in each next period state

        # Loop through each next-period-state and calculate the end-of-period
        # (marginal) value function
        for j in range(self.StateCount):
            # Condition values on next period's state (and record a couple for later use)
            self.conditionOnState(j)
            self.ExIncNextAll[j]   = np.dot(self.ShkPrbsNext,self.PermShkValsNext*self.TranShkValsNext)
            self.WorstIncPrbAll[j] = self.WorstIncPrb

            # Construct the end-of-period marginal value function conditional
            # on next period's state and add it to the list of value functions
            EndOfPrdvPfunc_cond = self.makeEndOfPrdvPfuncCond()
            self.EndOfPrdvPfunc_list.append(EndOfPrdvPfunc_cond)

            # Construct the end-of-period value functional conditional on next
            # period's state and add it to the list of value functions
            if self.vFuncBool:
                EndOfPrdvFunc_cond = self.makeEndOfPrdvFuncCond()
                self.EndOfPrdvFunc_list.append(EndOfPrdvFunc_cond)

        # EndOfPrdvP_cond is EndOfPrdvP conditional on *next* period's state.
        # Take expectations to get EndOfPrdvP conditional on *this* period's state.
        self.calcEndOfPrdvP()

        # Calculate the bounding MPCs and PDV of human wealth for each state
        self.calcHumWealthAndBoundingMPCs()

        # Find consumption and market resources corresponding to each end-of-period
        # assets point for each state (and add an additional point at the lower bound)
        aNrm = np.asarray(self.aXtraGrid)[np.newaxis,:] + np.array(self.BoroCnstNat_list)[:,np.newaxis]
        self.getPointsForInterpolation(self.EndOfPrdvP,aNrm)
        cNrm = np.hstack((np.zeros((self.StateCount,1)),self.cNrmNow))
        mNrm = np.hstack((np.reshape(self.mNrmMin_list,(self.StateCount,1)),self.mNrmNow))

        # Package and return the solution for this period
        self.BoroCnstNat = self.BoroCnstNat_list
        solution = self.makeSolution(cNrm,mNrm)
        return solution

    def defBoundary(self):
        '''
        Find the borrowing constraint for each current state and save it as an
        attribute of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.BoroCnstNatAll          = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext         = np.min(self.IncomeDstn_list[j][1])
            TranShkMinNext         = np.min(self.IncomeDstn_list[j][2])
            self.BoroCnstNatAll[j] = (self.solution_next.mNrmMin[j] - TranShkMinNext)*\
                                     (self.PermGroFac_list[j]*PermShkMinNext)/self.Rfree_list[j]

        self.BoroCnstNat_list   = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list       = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount,self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states         = self.MrkvArray[i,:] > 0
            self.BoroCnstNat_list[i]     = np.max(self.BoroCnstNatAll[possible_next_states])

            # Explicitly handle the "None" case:
            if self.BoroCnstArt is None:
                self.mNrmMin_list[i]         = self.BoroCnstNat_list[i]
            else:
                self.mNrmMin_list[i]         = np.max([self.BoroCnstNat_list[i],self.BoroCnstArt])
            self.BoroCnstDependency[i,:] = self.BoroCnstNat_list[i] == self.BoroCnstNatAll
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.

    def conditionOnState(self,state_index):
        '''
        Temporarily assume that a particular Markov state will occur in the
        succeeding period, and condition solver attributes on this assumption.
        Allows the solver to construct the future-state-conditional marginal
        value function (etc) for that future state.

        Parameters
        ----------
        state_index : int
            Index of the future Markov state to condition on.

        Returns
        -------
        none
        '''
        # Set future-state-conditional values as attributes of self
        self.IncomeDstn     = self.IncomeDstn_list[state_index]
        self.Rfree          = self.Rfree_list[state_index]
        self.PermGroFac     = self.PermGroFac_list[state_index]
        self.vPfuncNext     = self.solution_next.vPfunc[state_index]
        self.mNrmMinNow     = self.mNrmMin_list[state_index]
        self.BoroCnstNat    = self.BoroCnstNatAll[state_index]
        self.setAndUpdateValues(self.solution_next,self.IncomeDstn,self.LivPrb,self.DiscFac)
        self.DiscFacEff     = self.DiscFac # survival probability LivPrb represents probability from
                                           # *current* state, so DiscFacEff is just DiscFac for now

        # These lines have to come after setAndUpdateValues to override the definitions there
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        if self.CubicBool:
            self.vPPfuncNext= self.solution_next.vPPfunc[state_index]
        if self.vFuncBool:
            self.vFuncNext  = self.solution_next.vFunc[state_index]

    def calcEndOfPrdvPP(self):
        '''
        Calculates end-of-period marginal marginal value using a pre-defined
        array of next period market resources in self.mNrmNext.

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvPP : np.array
            End-of-period marginal marginal value of assets at each value in
            the grid of assets.
        '''
        EndOfPrdvPP = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)*\
                      np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*self.vPPfuncNext(self.mNrmNext)
                      *self.ShkPrbs_temp,axis=0)
        return EndOfPrdvPP

    def makeEndOfPrdvFuncCond(self):
        '''
        Construct the end-of-period value function conditional on next period's
        state.  NOTE: It might be possible to eliminate this method and replace
        it with ConsIndShockSolver.makeEndOfPrdvFunc, but the self.X_cond
        variables must be renamed.

        Parameters
        ----------
        none

        Returns
        -------
        EndofPrdvFunc_cond : ValueFunc
            The end-of-period value function conditional on a particular state
            occuring in the next period.
        '''
        VLvlNext               = (self.PermShkVals_temp**(1.0-self.CRRA)*
                                  self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mNrmNext)
        EndOfPrdv_cond         = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs_cond     = self.uinv(EndOfPrdv_cond)
        EndOfPrdvNvrsP_cond    = self.EndOfPrdvP_cond*self.uinvP(EndOfPrdv_cond)
        EndOfPrdvNvrs_cond     = np.insert(EndOfPrdvNvrs_cond,0,0.0)
        EndOfPrdvNvrsP_cond    = np.insert(EndOfPrdvNvrsP_cond,0,EndOfPrdvNvrsP_cond[0])
        aNrm_temp              = np.insert(self.aNrm_cond,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc_cond = CubicInterp(aNrm_temp,EndOfPrdvNvrs_cond,EndOfPrdvNvrsP_cond)
        EndofPrdvFunc_cond     = ValueFunc(EndOfPrdvNvrsFunc_cond,self.CRRA)
        return EndofPrdvFunc_cond


    def calcEndOfPrdvPcond(self):
        '''
        Calculate end-of-period marginal value of assets at each point in aNrmNow
        conditional on a particular state occuring in the next period.

        Parameters
        ----------
        None

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets.
        '''
        EndOfPrdvPcond = ConsIndShockSolver.calcEndOfPrdvP(self)
        return EndOfPrdvPcond


    def makeEndOfPrdvPfuncCond(self):
        '''
        Construct the end-of-period marginal value function conditional on next
        period's state.

        Parameters
        ----------
        None

        Returns
        -------
        EndofPrdvPfunc_cond : MargValueFunc
            The end-of-period marginal value function conditional on a particular
            state occuring in the succeeding period.
        '''
        # Get data to construct the end-of-period marginal value function (conditional on next state)
        self.aNrm_cond      = self.prepareToCalcEndOfPrdvP()
        self.EndOfPrdvP_cond= self.calcEndOfPrdvPcond()
        EndOfPrdvPnvrs_cond = self.uPinv(self.EndOfPrdvP_cond) # "decurved" marginal value
        if self.CubicBool:
            EndOfPrdvPP_cond = self.calcEndOfPrdvPP()
            EndOfPrdvPnvrsP_cond = EndOfPrdvPP_cond*self.uPinvP(self.EndOfPrdvP_cond) # "decurved" marginal marginal value

        # Construct the end-of-period marginal value function conditional on the next state.
        if self.CubicBool:
            EndOfPrdvPnvrsFunc_cond = CubicInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,
                                                  EndOfPrdvPnvrsP_cond,lower_extrap=True)
        else:
            EndOfPrdvPnvrsFunc_cond = LinearInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,
                                                   lower_extrap=True)
        EndofPrdvPfunc_cond = MargValueFunc(EndOfPrdvPnvrsFunc_cond,self.CRRA) # "recurve" the interpolated marginal value function
        return EndofPrdvPfunc_cond

    def calcEndOfPrdvP(self):
        '''
        Calculates end of period marginal value (and marginal marginal) value
        at each aXtra gridpoint for each current state, unconditional on the
        future Markov state (i.e. weighting conditional end-of-period marginal
        value by transition probabilities).

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        # Find unique values of minimum acceptable end-of-period assets (and the
        # current period states for which they apply).
        aNrmMin_unique, state_inverse = np.unique(self.BoroCnstNat_list,return_inverse=True)
        self.possible_transitions     = self.MrkvArray > 0

        # Calculate end-of-period marginal value (and marg marg value) at each
        # asset gridpoint for each current period state
        EndOfPrdvP                    = np.zeros((self.StateCount,self.aXtraGrid.size))
        EndOfPrdvPP                   = np.zeros((self.StateCount,self.aXtraGrid.size))
        for k in range(aNrmMin_unique.size):
            aNrmMin       = aNrmMin_unique[k]   # minimum assets for this pass
            which_states  = state_inverse == k  # the states for which this minimum applies
            aGrid         = aNrmMin + self.aXtraGrid # assets grid for this pass
            EndOfPrdvP_all  = np.zeros((self.StateCount,self.aXtraGrid.size))
            EndOfPrdvPP_all = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if np.any(np.logical_and(self.possible_transitions[:,j],which_states)): # only consider a future state if one of the relevant states could transition to it
                    EndOfPrdvP_all[j,:] = self.EndOfPrdvPfunc_list[j](aGrid)
                    if self.CubicBool: # Add conditional end-of-period (marginal) marginal value to the arrays
                        EndOfPrdvPP_all[j,:] = self.EndOfPrdvPfunc_list[j].derivative(aGrid)
            # Weight conditional marginal (marginal) values by transition probs
            # to get unconditional marginal (marginal) value at each gridpoint.
            EndOfPrdvP_temp = np.dot(self.MrkvArray,EndOfPrdvP_all)
            EndOfPrdvP[which_states,:] = EndOfPrdvP_temp[which_states,:] # only take the states for which this asset minimum applies
            if self.CubicBool:
                EndOfPrdvPP_temp = np.dot(self.MrkvArray,EndOfPrdvPP_all)
                EndOfPrdvPP[which_states,:] = EndOfPrdvPP_temp[which_states,:]

        # Store the results as attributes of self, scaling end of period marginal value by survival probability from each current state
        LivPrb_tiled = np.tile(np.reshape(self.LivPrb,(self.StateCount,1)),(1,self.aXtraGrid.size))
        self.EndOfPrdvP = LivPrb_tiled*EndOfPrdvP
        if self.CubicBool:
            self.EndOfPrdvPP = LivPrb_tiled*EndOfPrdvPP

    def calcHumWealthAndBoundingMPCs(self):
        '''
        Calculates human wealth and the maximum and minimum MPC for each current
        period state, then stores them as attributes of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        # Upper bound on MPC at lower m-bound
        WorstIncPrb_array = self.BoroCnstDependency*np.tile(np.reshape(self.WorstIncPrbAll,
                            (1,self.StateCount)),(self.StateCount,1))
        temp_array        = self.MrkvArray*WorstIncPrb_array
        WorstIncPrbNow    = np.sum(temp_array,axis=1) # Probability of getting the "worst" income shock and transition from each current state
        ExMPCmaxNext      = (np.dot(temp_array,self.Rfree_list**(1.0-self.CRRA)*
                            self.solution_next.MPCmax**(-self.CRRA))/WorstIncPrbNow)**\
                            (-1.0/self.CRRA)
        DiscFacEff_temp   = self.DiscFac*self.LivPrb
        self.MPCmaxNow    = 1.0/(1.0 + ((DiscFacEff_temp*WorstIncPrbNow)**
                            (1.0/self.CRRA))/ExMPCmaxNext)
        self.MPCmaxEff    = self.MPCmaxNow
        self.MPCmaxEff[self.BoroCnstNat_list < self.mNrmMin_list] = 1.0
        # State-conditional PDV of human wealth
        hNrmPlusIncNext   = self.ExIncNextAll + self.solution_next.hNrm
        self.hNrmNow      = np.dot(self.MrkvArray,(self.PermGroFac_list/self.Rfree_list)*
                            hNrmPlusIncNext)
        # Lower bound on MPC as m gets arbitrarily large
        temp              = (DiscFacEff_temp*np.dot(self.MrkvArray,self.solution_next.MPCmin**
                            (-self.CRRA)*self.Rfree_list**(1.0-self.CRRA)))**(1.0/self.CRRA)
        self.MPCminNow    = 1.0/(1.0 + temp)

    def makeSolution(self,cNrm,mNrm):
        '''
        Construct an object representing the solution to this period's problem.

        Parameters
        ----------
        cNrm : np.array
            Array of normalized consumption values for interpolation.  Each row
            corresponds to a Markov state for this period.
        mNrm : np.array
            Array of normalized market resource values for interpolation.  Each
            row corresponds to a Markov state for this period.

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a marg-
            inal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        '''
        solution = ConsumerSolution() # An empty solution to which we'll add state-conditional solutions
        # Calculate the MPC at each market resource gridpoint in each state (if desired)
        if self.CubicBool:
            dcda          = self.EndOfPrdvPP/self.uPP(np.array(self.cNrmNow))
            MPC           = dcda/(dcda+1.0)
            self.MPC_temp = np.hstack((np.reshape(self.MPCmaxNow,(self.StateCount,1)),MPC))
            interpfunc    = self.makeCubiccFunc
        else:
            interpfunc    = self.makeLinearcFunc

        # Loop through each current period state and add its solution to the overall solution
        for i in range(self.StateCount):
            # Set current-period-conditional human wealth and MPC bounds
            self.hNrmNow_j   = self.hNrmNow[i]
            self.MPCminNow_j = self.MPCminNow[i]
            if self.CubicBool:
                self.MPC_temp_j  = self.MPC_temp[i,:]

            # Construct the consumption function by combining the constrained and unconstrained portions
            self.cFuncNowCnst = LinearInterp([self.mNrmMin_list[i], self.mNrmMin_list[i]+1.0],
                                             [0.0,1.0])
            cFuncNowUnc       = interpfunc(mNrm[i,:],cNrm[i,:])
            cFuncNow          = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

            # Make the marginal value function and pack up the current-state-conditional solution
            vPfuncNow     = MargValueFunc(cFuncNow,self.CRRA)
            solution_cond = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow,
                                             mNrmMin=self.mNrmMinNow)
            if self.CubicBool: # Add the state-conditional marginal marginal value function (if desired)
                solution_cond = self.addvPPfunc(solution_cond)

            # Add the current-state-conditional solution to the overall period solution
            solution.appendSolution(solution_cond)

        # Add the lower bounds of market resources, MPC limits, human resources,
        # and the value functions to the overall solution
        solution.mNrmMin = self.mNrmMin_list
        solution         = self.addMPCandHumanWealth(solution)
        if self.vFuncBool:
            vFuncNow = self.makevFunc(solution)
            solution.vFunc = vFuncNow

        # Return the overall solution to this period
        return solution


    def makeLinearcFunc(self,mNrm,cNrm):
        '''
        Make a linear interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.LinearInterp
        '''
        cFuncUnc = LinearInterp(mNrm,cNrm,self.MPCminNow_j*self.hNrmNow_j,self.MPCminNow_j)
        return cFuncUnc


    def makeCubiccFunc(self,mNrm,cNrm):
        '''
        Make a cubic interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.CubicInterp
        '''
        cFuncUnc = CubicInterp(mNrm,cNrm,self.MPC_temp_j,self.MPCminNow_j*self.hNrmNow_j,
                               self.MPCminNow_j)
        return cFuncUnc

    def makevFunc(self,solution):
        '''
        Construct the value function for each current state.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Must
            have a consumption function cFunc (using cubic or linear splines) as
            a list with elements corresponding to the current Markov state.  E.g.
            solution.cFunc[0] is the consumption function when in the i=0 Markov
            state this period.

        Returns
        -------
        vFuncNow : [ValueFunc]
            A list of value functions (defined over normalized market resources
            m) for each current period Markov state.
        '''
        vFuncNow = [] # Initialize an empty list of value functions
        # Loop over each current period state and construct the value function
        for i in range(self.StateCount):
            # Make state-conditional grids of market resources and consumption
            mNrmMin       = self.mNrmMin_list[i]
            mGrid         = mNrmMin + self.aXtraGrid
            cGrid         = solution.cFunc[i](mGrid)
            aGrid         = mGrid - cGrid

            # Calculate end-of-period value at each gridpoint
            EndOfPrdv_all   = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if self.possible_transitions[i,j]:
                    EndOfPrdv_all[j,:] = self.EndOfPrdvFunc_list[j](aGrid)
            EndOfPrdv     = np.dot(self.MrkvArray[i,:],EndOfPrdv_all)

            # Calculate (normalized) value and marginal value at each gridpoint
            vNrmNow       = self.u(cGrid) + EndOfPrdv
            vPnow         = self.uP(cGrid)

            # Make a "decurved" value function with the inverse utility function
            vNvrs        = self.uinv(vNrmNow) # value transformed through inverse utility
            vNvrsP       = vPnow*self.uinvP(vNrmNow)
            mNrm_temp    = np.insert(mGrid,0,mNrmMin) # add the lower bound
            vNvrs        = np.insert(vNvrs,0,0.0)
            vNvrsP       = np.insert(vNvrsP,0,self.MPCmaxEff[i]**(-self.CRRA/(1.0-self.CRRA)))
            MPCminNvrs   = self.MPCminNow[i]**(-self.CRRA/(1.0-self.CRRA))
            vNvrsFunc_i  = CubicInterp(mNrm_temp,vNvrs,vNvrsP,MPCminNvrs*self.hNrmNow[i],MPCminNvrs)

            # "Recurve" the decurved value function and add it to the list
            vFunc_i     = ValueFunc(vNvrsFunc_i,self.CRRA)
            vFuncNow.append(vFunc_i)
        return vFuncNow


def solveConsMarkov(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                 MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncomeDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn_list : [[np.array]]
        A length N list of income distributions in each succeeding Markov
        state.  Each income distribution contains three arrays of floats,
        representing a discrete approximation to the income process at the
        beginning of the succeeding period. Order: event probabilities,
        permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_list : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac_list : float
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin, normalized human wealth hNrm, and bounding
        MPCs MPCmin and MPCmax.  It might also have a value function vFunc
        and marginal marginal value function vPPfunc.  All of these attributes
        are lists or arrays, with elements corresponding to the current
        Markov state.  E.g. solution.cFunc[0] is the consumption function
        when in the i=0 Markov state this period.
    '''
    solver = ConsMarkovSolver(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                              PermGroFac,MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
    solution_now = solver.solve()
    return solution_now



####################################################################################################
####################################################################################################

class MarkovConsumerType(IndShockConsumerType):
    '''
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    '''
    time_vary_ = IndShockConsumerType.time_vary_ + ['MrkvArray']
    shock_vars_ = IndShockConsumerType.shock_vars_ + ['MrkvNow']

    def __init__(self,cycles=1,time_flow=True,**kwds):
        IndShockConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        self.solveOnePeriod = solveConsMarkov
        self.poststate_vars += ['MrkvNow']
        if not hasattr(self, 'global_markov'):
            self.global_markov = False

    def checkMarkovInputs(self):
        '''
        Many parameters used by MarkovConsumerType are arrays.  Make sure those arrays are the
        right shape.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        StateCount = self.MrkvArray[0].shape[0]

        # Check that arrays are the right shape
        assert self.Rfree.shape      == (StateCount,),'Rfree not the right shape!'

        # Check that arrays in lists are the right shape
        for MrkvArray_t in self.MrkvArray:
            assert MrkvArray_t.shape  == (StateCount,StateCount),'MrkvArray not the right shape!'
        for LivPrb_t in self.LivPrb:
            assert LivPrb_t.shape == (StateCount,),'Array in LivPrb is not the right shape!'
        for PermGroFac_t in self.LivPrb:
            assert PermGroFac_t.shape == (StateCount,),'Array in PermGroFac is not the right shape!'

        # Now check the income distribution.
        # Note IncomeDstn is (potentially) time-varying, so it is in time_vary.
        # Therefore it is a list, and each element of that list responds to the income distribution
        # at a particular point in time.  Each income distribution at a point in time should itself
        # be a list, with each element corresponding to the income distribution
        # conditional on a particular Markov state.
        for IncomeDstn_t in self.IncomeDstn:
            assert len(IncomeDstn_t) == StateCount,'List in IncomeDstn is not the right length!'

    def preSolve(self):
        """
        Check to make sure that the inputs that are specific to MarkovConsumerType
        are of the right shape (if arrays) or length (if lists).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        AgentType.preSolve(self)
        self.checkMarkovInputs()

    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        IndShockConsumerType.updateSolutionTerminal(self)

        # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
        StateCount = self.MrkvArray[0].shape[0]
        self.solution_terminal.cFunc   = StateCount*[self.cFunc_terminal_]
        self.solution_terminal.vFunc   = StateCount*[self.solution_terminal.vFunc]
        self.solution_terminal.vPfunc  = StateCount*[self.solution_terminal.vPfunc]
        self.solution_terminal.vPPfunc = StateCount*[self.solution_terminal.vPPfunc]
        self.solution_terminal.mNrmMin = np.zeros(StateCount)
        self.solution_terminal.hRto    = np.zeros(StateCount)
        self.solution_terminal.MPCmax  = np.ones(StateCount)
        self.solution_terminal.MPCmin  = np.ones(StateCount)

    def initializeSim(self):
        IndShockConsumerType.initializeSim(self)
        if self.global_markov:  #Need to initialize markov state to be the same for all agents
            base_draw = drawUniform(1,seed=self.RNG.randint(0,2**31-1))
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.MrkvNow = np.ones(self.AgentCount)*np.searchsorted(Cutoffs,base_draw).astype(int)
        self.MrkvNow = self.MrkvNow.astype(int)

    def simDeath(self):
        '''
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        '''
        # Determine who dies
        LivPrb = np.array(self.LivPrb)[self.t_cycle-1,self.MrkvNow] # Time has already advanced, so look back one
        DiePrb = 1.0 - LivPrb
        DeathShks = drawUniform(N=self.AgentCount,seed=self.RNG.randint(0,2**31-1))
        which_agents = DeathShks < DiePrb
        if self.T_age is not None: # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents,too_old)
        return which_agents

    def simBirth(self,which_agents):
        '''
        Makes new Markov consumer by drawing initial normalized assets, permanent income levels, and
        discrete states. Calls IndShockConsumerType.simBirth, then draws from initial Markov distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        IndShockConsumerType.simBirth(self,which_agents) # Get initial assets and permanent income
        if not self.global_markov:  #Markov state is not changed if it is set at the global level
            N = np.sum(which_agents)
            base_draws = drawUniform(N,seed=self.RNG.randint(0,2**31-1))
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.MrkvNow[which_agents] = np.searchsorted(Cutoffs,base_draws).astype(int)

    def getShocks(self):
        '''
        Gets new Markov states and permanent and transitory income shocks for this period.  Samples
        from IncomeDstn for each period-state in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Get new Markov states for each agent
        if self.global_markov:
            base_draws = np.ones(self.AgentCount)*drawUniform(1,seed=self.RNG.randint(0,2**31-1))
        else:
            base_draws = self.RNG.permutation(np.arange(self.AgentCount,dtype=float)/self.AgentCount + 1.0/(2*self.AgentCount))
        newborn = self.t_age == 0 # Don't change Markov state for those who were just born (unless global_markov)
        MrkvPrev = self.MrkvNow
        MrkvNow = np.zeros(self.AgentCount,dtype=int)
        for t in range(self.T_cycle):
            Cutoffs = np.cumsum(self.MrkvArray[t],axis=1)
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(self.t_cycle == t,MrkvPrev == j)
                MrkvNow[these] = np.searchsorted(Cutoffs[j,:],base_draws[these]).astype(int)
        if not self.global_markov:
                MrkvNow[newborn] = MrkvPrev[newborn]
        self.MrkvNow = MrkvNow.astype(int)

        # Now get income shocks for each consumer, by cycle-time and discrete state
        PermShkNow = np.zeros(self.AgentCount) # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                N = np.sum(these)
                if N > 0:
                    IncomeDstnNow    = self.IncomeDstn[t-1][j] # set current income distribution
                    PermGroFacNow    = self.PermGroFac[t-1][j] # and permanent growth factor
                    Indices          = np.arange(IncomeDstnNow[0].size) # just a list of integers
                    # Get random draws of income shocks from the discrete distribution
                    EventDraws       = drawDiscrete(N,X=Indices,P=IncomeDstnNow[0],exact_match=False,seed=self.RNG.randint(0,2**31-1))
                    PermShkNow[these] = IncomeDstnNow[1][EventDraws]*PermGroFacNow # permanent "shock" includes expected growth
                    TranShkNow[these] = IncomeDstnNow[2][EventDraws]
        newborn = self.t_age == 0
        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0
        self.PermShkNow = PermShkNow
        self.TranShkNow = TranShkNow

    def readShocks(self):
        '''
        A slight modification of AgentType.readShocks that makes sure that MrkvNow is int, not float.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.readShocks(self)
        self.MrkvNow = self.MrkvNow.astype(int)

    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with interest factor that varies with discrete state.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.Rfree[self.MrkvNow]
        return RfreeNow

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
        for t in range(self.T_cycle):
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(t == self.t_cycle, j == self.MrkvNow)
                cNrmNow[these] = self.solution[t].cFunc[j](self.mNrmNow[these])
        self.cNrmNow = cNrmNow
        return None

    def calcBoundingValues(self):
        '''
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  Results are all
        np.array with elements corresponding to each Markov state.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        raise NotImplementedError()

    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncomeDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncomeDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None
        '''
        raise NotImplementedError()



###############################################################################

def main():
    import HARK.ConsumptionSaving.ConsumerParameters as Params
    from HARK.utilities import plotFuncs
    from time import clock
    from copy import copy
    mystr = lambda number : "{:.4f}".format(number)

    do_simulation           = True

    # Define the Markov transition matrix for serially correlated unemployment
    unemp_length = 5         # Averange length of unemployment spell
    urate_good = 0.05        # Unemployment rate when economy is in good state
    urate_bad = 0.12         # Unemployment rate when economy is in bad state
    bust_prob = 0.01         # Probability of economy switching from good to bad
    recession_length = 20    # Averange length of bad state
    p_reemploy =1.0/unemp_length
    p_unemploy_good = p_reemploy*urate_good/(1-urate_good)
    p_unemploy_bad = p_reemploy*urate_bad/(1-urate_bad)
    boom_prob = 1.0/recession_length
    MrkvArray = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),
                           (1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                          [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),
                           p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                          [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,
                           (1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                          [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,
                           p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])

    # Make a consumer with serially correlated unemployment, subject to boom and bust cycles
    init_serial_unemployment = copy(Params.init_idiosyncratic_shocks)
    init_serial_unemployment['MrkvArray'] = [MrkvArray]
    init_serial_unemployment['UnempPrb'] = 0 # to make income distribution when employed
    init_serial_unemployment['global_markov'] = False
    SerialUnemploymentExample = MarkovConsumerType(**init_serial_unemployment)
    SerialUnemploymentExample.cycles = 0
    SerialUnemploymentExample.vFuncBool = False # for easy toggling here

    # Replace the default (lognormal) income distribution with a custom one
    employed_income_dist   = [np.ones(1),np.ones(1),np.ones(1)] # Definitely get income
    unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)] # Definitely don't
    SerialUnemploymentExample.IncomeDstn = [[employed_income_dist,unemployed_income_dist,employed_income_dist,
                              unemployed_income_dist]]

    # Interest factor, permanent growth rates, and survival probabilities are constant arrays
    SerialUnemploymentExample.Rfree = np.array(4*[SerialUnemploymentExample.Rfree])
    SerialUnemploymentExample.PermGroFac = [np.array(4*SerialUnemploymentExample.PermGroFac)]
    SerialUnemploymentExample.LivPrb = [SerialUnemploymentExample.LivPrb*np.ones(4)]

    # Solve the serial unemployment consumer's problem and display solution
    SerialUnemploymentExample.timeFwd()
    start_time = clock()
    SerialUnemploymentExample.solve()
    end_time = clock()
    print('Solving a Markov consumer with serially correlated unemployment took ' + mystr(end_time-start_time) + ' seconds.')
    print('Consumption functions for each discrete state:')
    plotFuncs(SerialUnemploymentExample.solution[0].cFunc,0,50)
    if SerialUnemploymentExample.vFuncBool:
        print('Value functions for each discrete state:')
        plotFuncs(SerialUnemploymentExample.solution[0].vFunc,5,50)

    # Simulate some data; results stored in cHist, mNrmNow_hist, cNrmNow_hist, and MrkvNow_hist
    if do_simulation:
        SerialUnemploymentExample.T_sim = 120
        SerialUnemploymentExample.MrkvPrbsInit = [0.25,0.25,0.25,0.25]
        SerialUnemploymentExample.track_vars = ['mNrmNow','cNrmNow']
        SerialUnemploymentExample.makeShockHistory() # This is optional
        SerialUnemploymentExample.initializeSim()
        SerialUnemploymentExample.simulate()

###############################################################################

    # Make a consumer who occasionally gets "unemployment immunity" for a fixed period
    UnempPrb    = 0.05  # Probability of becoming unemployed each period
    ImmunityPrb = 0.01  # Probability of becoming "immune" to unemployment
    ImmunityT   = 6     # Number of periods of immunity

    StateCount = ImmunityT+1   # Total number of Markov states
    IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0/(1.0-UnempPrb),0.0])] # Ordinary income distribution
    IncomeDstnImm = [np.array([1.0]), np.array([1.0]), np.array([1.0])] # Income distribution when unemployed
    IncomeDstn = [IncomeDstnReg] + ImmunityT*[IncomeDstnImm] # Income distribution for each Markov state, in a list

    # Make the Markov transition array.  MrkvArray[i,j] is the probability of transitioning
    # to state j in period t+1 from state i in period t.
    MrkvArray = np.zeros((StateCount,StateCount))
    MrkvArray[0,0] = 1.0 - ImmunityPrb   # Probability of not becoming immune in ordinary state: stay in ordinary state
    MrkvArray[0,ImmunityT] = ImmunityPrb # Probability of becoming immune in ordinary state: begin immunity periods
    for j in range(ImmunityT):
        MrkvArray[j+1,j] = 1.0  # When immune, have 100% chance of transition to state with one fewer immunity periods remaining

    init_unemployment_immunity = copy(Params.init_idiosyncratic_shocks)
    init_unemployment_immunity['MrkvArray'] = [MrkvArray]
    ImmunityExample = MarkovConsumerType(**init_unemployment_immunity)
    ImmunityExample.assignParameters(Rfree = np.array(np.array(StateCount*[1.03])), # Interest factor same in all states
                                  PermGroFac = [np.array(StateCount*[1.01])],    # Permanent growth factor same in all states
                                  LivPrb = [np.array(StateCount*[0.98])],        # Same survival probability in all states
                                  BoroCnstArt = None,                            # No artificial borrowing constraint
                                  cycles = 0)                                    # Infinite horizon
    ImmunityExample.IncomeDstn = [IncomeDstn]

    # Solve the unemployment immunity problem and display the consumption functions
    start_time = clock()
    ImmunityExample.solve()
    end_time = clock()
    print('Solving an "unemployment immunity" consumer took ' + mystr(end_time-start_time) + ' seconds.')
    print('Consumption functions for each discrete state:')
    mNrmMin = np.min([ImmunityExample.solution[0].mNrmMin[j] for j in range(StateCount)])
    plotFuncs(ImmunityExample.solution[0].cFunc,mNrmMin,10)

###############################################################################

    # Make a consumer with serially correlated permanent income growth
    UnempPrb = 0.05    # Unemployment probability
    StateCount = 5     # Number of permanent income growth rates
    Persistence = 0.5  # Probability of getting the same permanent income growth rate next period

    IncomeDstnReg = [np.array([1-UnempPrb,UnempPrb]), np.array([1.0,1.0]), np.array([1.0,0.0])]
    IncomeDstn = StateCount*[IncomeDstnReg] # Same simple income distribution in each state

    # Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
    MrkvArray = Persistence*np.eye(StateCount) + (1.0/StateCount)*(1.0-Persistence)*np.ones((StateCount,StateCount))

    init_serial_growth = copy(Params.init_idiosyncratic_shocks)
    init_serial_growth['MrkvArray'] = [MrkvArray]
    SerialGroExample = MarkovConsumerType(**init_serial_growth)
    SerialGroExample.assignParameters(Rfree = np.array(np.array(StateCount*[1.03])),    # Same interest factor in each Markov state
                                   PermGroFac = [np.array([0.97,0.99,1.01,1.03,1.05])], # Different permanent growth factor in each Markov state
                                   LivPrb = [np.array(StateCount*[0.98])],              # Same survival probability in all states
                                   cycles = 0)
    SerialGroExample.IncomeDstn = [IncomeDstn]


    # Solve the serially correlated permanent growth shock problem and display the consumption functions
    start_time = clock()
    SerialGroExample.solve()
    end_time = clock()
    print('Solving a serially correlated growth consumer took ' + mystr(end_time-start_time) + ' seconds.')
    print('Consumption functions for each discrete state:')
    plotFuncs(SerialGroExample.solution[0].cFunc,0,10)

###############################################################################

    # Make a consumer with serially correlated interest factors
    SerialRExample = deepcopy(SerialGroExample) # Same as the last problem...
    SerialRExample.assignParameters(PermGroFac = [np.array(StateCount*[1.01])],   # ...but now the permanent growth factor is constant...
                                 Rfree = np.array([1.01,1.02,1.03,1.04,1.05])) # ...and the interest factor is what varies across states

    # Solve the serially correlated interest rate problem and display the consumption functions
    start_time = clock()
    SerialRExample.solve()
    end_time = clock()
    print('Solving a serially correlated interest consumer took ' + mystr(end_time-start_time) + ' seconds.')
    print('Consumption functions for each discrete state:')
    plotFuncs(SerialRExample.solution[0].cFunc,0,10)


if __name__ == '__main__':
    main()
