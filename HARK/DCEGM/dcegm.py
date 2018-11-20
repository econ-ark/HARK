# NOTE TO SELF: we manually place 0 (1e-6) in the grids, and that helps with
# interpolating on common grid

import numpy
import matplotlib.pyplot as plt
from collections import namedtuple
from HARK import Solution, AgentType
from HARK.interpolation import LinearInterp
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv
# might as well alias them utility, as we need CRRA
# should use these instead of hardcoded log (CRRA=1)
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityP_inv  = CRRAutilityP_inv
utility_inv   = CRRAutility_inv

def dcegmUpperEnvelope(M, C, V):

    return None

def dcegmEGM(M, C, V):

    return None

def DCEGM(M, C, V):

    return None

RetiringDeatonParameters = namedtuple('RetiringDeatonParamters',
                                      'DiscFac CRRA DisUtil Rfree y eta sigma')


def nonlinspace(low, high, n, phi = 1.1):
    '''
    Recursively constructs a non-linearly spaces grid that starts at low, ends at high, has n
    elements, and has smaller step lengths towards low if phi > 1 and smaller
    step lengths towards high if phi < 1.
    Taken from original implementation of G2EGM.
    Parameters
    ----------
    low : float
        first value in the returned grid
    high : float
        last value in the returned grid
    n : int
        number of elements in grid
    phi : float
        degree of non-linearity. Default is 1.5 (smaller steps towards `low`)
    Returns
    -------
    x : numpy.array
        the non-linearly spaced grid
    '''
    x    = numpy.ones(n)*numpy.nan
    x[0] = low
    for i in range(1,n):
        x[i] = x[i-1] + (high-x[i-1])/((n-i)**phi)
    return x

class RetiringDeatonSolution(Solution):
    def __init__(self, rs, ws, M, C, V_T, P):
        self.rs = rs
        self.ws = ws
        self.M = M
        self.C = C
        self.V_T = V_T
        self.P = P

class RetiredDeatonSolution(Solution):
    def __init__(self, C, M, V_T):
        self.C = C
        self.M = M
        self.V_T = V_T

class WorkingDeatonSolution(Solution):
    def __init__(self, M, C, V_T):
        self.M = M
        self.C = C
        self.V_T = V_T

class RetiringDeaton(AgentType):
    def __init__(self, DiscFac=0.95, Rfree=1.05, DisUtil=1.0,
                 shiftPoints=False, T=20, CRRA=1.0, sigma=0.0,
                 eta=40.0,
                 y=0.0, # normalized relative to work
                 Nm=2000,
                 aLims=(1e-6, 500), Na=1800,
                 saveCommon=False,
                 **kwds):

        AgentType.__init__(self, **kwds)
        self.T = T
        self.saveCommon = saveCommon
        # check conditions for analytic solution
        if not CRRA == 1.0:
            # err
            return None
            # Todo the DisUtil is wrong below, look up in paper
        if DiscFac*Rfree > 1.0 or DisUtil > (1+DiscFac)*numpy.log(1+DiscFac):
            # err
            return None

        self.time_inv = ['aGrid', 'mGrid', 'EGMVector', 'par', 'Util', 'UtilP',
                         'UtilP_inv', 'saveCommon']
        self.time_vary = ['age']

        self.age = list(range(T-1))

        self.par = RetiringDeatonParameters(DiscFac, CRRA, DisUtil, Rfree, y, eta, sigma)
        self.aLims = aLims
        self.Na = Na
        self.Nm = Nm
        # d == 2 is working
        # - 10.0 moves curve down to improve lienar interpolation
        self.Util = lambda c, d: numpy.log(c) + self.par.DisUtil*(2-d) - 10.0
        self.UtilP = lambda c, d: c**(-self.par.CRRA) # we require CRRA 1.0 for now...
        self.UtilP_inv = lambda u, d: u**(-1.0/self.par.CRRA) # ... so ...

        self.preSolve = self.updateLast
        self.solveOnePeriod = solveRetiringDeaton

    def updateLast(self):
        """
        Updates grids and functions according to the given model parameters, and
        solves the last period.

        Parameters
        ---------
        None

        Returns
        -------
        None
        """
        self.aGrid = nonlinspace(self.aLims[0], self.aLims[1], self.Na)
        self.mGrid = nonlinspace(self.aLims[0], self.aLims[1]*1.5, self.Na)
        self.EGMVector = numpy.zeros(self.Nm)

        rs = self.solveLastRetired()
        ws = self.solveLastWorking()

        commonM = ws.M

        if self.saveCommon:
            # To save the pre-disrete choice expected consumption and value function,
            # we need to interpolate onto the same grid between the two. We know that
            # ws.C and ws.V_T are on the ws.M grid, so we use that to interpolate.
            Crs = LinearInterp(rs.M, rs.C)(commonM)
            Cws = ws.C
    #        Cws = LinearInterp(rs.M, rs.C)(mGrid)

            V_Trs = LinearInterp(rs.M, rs.V_T)(commonM)
            V_Tws = ws.V_T

            # use vstack to stack the choice specific value functions by
            # value of cash-on-hand
            V_T, P = discreteEnvelope(numpy.vstack((V_Trs, V_Tws)), self.par.sigma)

            # find the expected consumption prior to discrete choice by element-
            # wise multiplication between choice distribution and choice specific
            # consumption functions
            C = (P*numpy.vstack((Crs, Cws))).sum(axis=0)
        else:
            C, V_T, P = None, None, None

        self.solution_terminal = RetiringDeatonSolution(rs, ws, commonM, C, V_T, P)#M, C, -1.0/V, P)

    def solveLastRetired(self):
        """
        Solves the last period of a retired agent.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        choice = 1
        C = self.mGrid # consume everything
        M = self.mGrid # everywhere
        # this transformation is different than in our G2EGM, we
        # need to figure out which is better
        V_T = -1.0/self.Util(self.mGrid, choice)
        return RetiredDeatonSolution(C, M, V_T)

    def solveLastWorking(self):
        """
        Solves the last period of a working agent.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        choice = 2
        C = self.mGrid # consume everything
        M = self.mGrid # everywhere
        # this transformation is different than in our G2EGM, we
        # need to figure out which is better
        V_T = -1.0/self.Util(self.mGrid, choice)
        return WorkingDeatonSolution(M, C, V_T)

    def plotV(self, t, d, plot_range = None):

        if t == self.T:
            sol = self.solution_terminal
        else:
            sol = self.solution[self.T-1-t]

        if d == 1:
            sol = sol.rs
            choice_str = "retired"
        elif d == 2:
            sol = sol.ws
            choice_str = "working"

        if plot_range == None:
            plot_range = range(len(sol.M))
        else:
            plot_range = range(plot_range[0], plot_range[1])
        plt.plot(sol.M[plot_range], numpy.divide(-1.0, sol.V_T[plot_range]), label="{} (t = {})".format(choice_str, t))
        plt.legend()
        plt.xlabel("M")
        plt.ylabel("C(M)")
        plt.title('Choice specific value functions')


    def plotC(self, t, d, plot_range = None, **kwds):

        if t == self.T:
            sol = self.solution_terminal
        else:
            sol = self.solution[self.T-1-t]

        if d == 1:
            sol = sol.rs
            choice_str = "retired"
        elif d == 2:
            sol = sol.ws
            choice_str = "working"

        if plot_range == None:
            plot_range = range(len(sol.M))
        else:
            plot_range = range(plot_range[0], plot_range[1])
        plt.plot(sol.M[plot_range], sol.C[plot_range], label="{} (t = {})".format(choice_str, t), **kwds)
        plt.legend()
        plt.xlabel("M")
        plt.ylabel("C(M)")
        plt.title('Choice specific consumption functions')

def solveRetiringDeaton(solution_next, aGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv, age, saveCommon):
    """
    Solves a period of problem defined by the RetiringDeaton AgentType. It uses
    DCEGM to solve the mixed choice problem.

    Parameters
    ----------

    Returns
    -------

    """
    rs = solveRetiredDeaton(solution_next, aGrid, EGMVector, par, Util, UtilP, UtilP_inv)
    ws = solveWorkingDeaton(solution_next, aGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv)

    if saveCommon:
        # To save the pre-disrete choice expected consumption and value function,
        # we need to interpolate onto the same grid between the two. We know that
        # ws.C and ws.V_T are on the ws.M grid, so we use that to interpolate.
        Crs = LinearInterp(rs.M, rs.C)(ws.M)
        Cws = ws.C
#        Cws = LinearInterp(rs.M, rs.C)(mGrid)

        V_Trs = LinearInterp(rs.M, rs.V_T)(ws.M)
        V_Tws = ws.V_T

        V_T, P = discreteEnvelope(numpy.vstack((V_Trs, V_Tws)), par.sigma)

        C = (P*numpy.vstack((Crs, Cws))).sum(axis=0)
    else:
        C, V_T, P = None, None, None

    return RetiringDeatonSolution(rs, ws, mGrid, C, V_T, P)#-1.0/V, P) #0,0 is m and c

def solveRetiredDeaton(solution_next, aGrid, EGMVector, par, Util, UtilP, UtilP_inv):
    choice = 1
    rs_tp1 = solution_next.rs
    M = numpy.copy(EGMVector)
    C = numpy.copy(EGMVector)
    Eu = numpy.copy(EGMVector)
    Ev = numpy.copy(EGMVector)

    aLen = len(aGrid)
    conLen = len(M)-aLen
    # Next-period initial wealth given exogenous aGrid
    M_tp1 = par.Rfree*aGrid + par.y

    # Prepare variables for EGM step
    # Augmented M
    augM = numpy.insert(rs_tp1.M, 0, 0.0)
    augC = numpy.insert(rs_tp1.C, 0, 0.0)
    augV_T = numpy.insert(rs_tp1.V_T, 0, 0.0)
    C_tp1 = LinearInterp(augM, augC, lower_extrap=True)(M_tp1)
    V_T_tp1 = LinearInterp(augM, augV_T, lower_extrap=True)(M_tp1)     #interpvtp1
    V_tp1 = numpy.divide(-1.0, V_T_tp1)


    # Calculate the expected marginal utility and expected value function
    Eu[conLen:] = par.Rfree*UtilP(C_tp1, choice)
    Ev[conLen:] = V_tp1

    # EGM step
    C[conLen:] = UtilP_inv(par.DiscFac*Eu[conLen:], choice)
    M[conLen:] = aGrid + C[conLen:]

    # Add points to M (and C) to solve between 0 and the first point EGM finds
    # (that is add the solution in the segment where the agent is constrained)
    M[0:conLen] = numpy.linspace(numpy.min(aGrid), M[conLen]*0.99, len(M)-aLen)
    C[0:conLen] = M[0:conLen]

    Ev[0:conLen] = Ev[conLen+1]

    V_T = numpy.divide(-1.0, Util(C, choice) + par.DiscFac*Ev)

    return RetiredDeatonSolution(C, M, V_T)

def solveWorkingDeaton(solution_next, aGrid, mGrid, EGMVector, par, Util, UtilP, UtilP_inv):
    choice = 2
    rs_tp1 = solution_next.rs
    ws_tp1 = solution_next.ws
    M = numpy.copy(EGMVector)
    C = numpy.copy(EGMVector)
    Eu = numpy.copy(EGMVector)
    Ev = numpy.copy(EGMVector)

    aLen = len(aGrid)
    conLen = len(M)-aLen
    # Next-period initial wealth given exogenous aGrid
    M_tp1 = par.Rfree*aGrid + par.eta
    # Prepare variables for EGM step
    rs_augM = numpy.insert(rs_tp1.M, 0, 0.0)
    rs_augC = numpy.insert(rs_tp1.C, 0, 0.0)
    rs_augV_T = numpy.insert(rs_tp1.V_T, 0, 0.0)
    ws_augM = numpy.insert(ws_tp1.M, 0, 0.0)
    ws_augC = numpy.insert(ws_tp1.C, 0, 0.0)
    ws_augV_T = numpy.insert(ws_tp1.V_T, 0, 0.0)

    Cr_tp1 = LinearInterp(rs_augM, rs_augC, lower_extrap=True)(M_tp1)
    Cw_tp1 = LinearInterp(ws_augM, ws_augC, lower_extrap=True)(M_tp1)
    Vr_T = LinearInterp(rs_augM, rs_augV_T, lower_extrap=True)(M_tp1)
    Vw_T = LinearInterp(ws_augM, ws_augV_T, lower_extrap=True)(M_tp1)

    # Due to the transformation on V being monotonic increasing, we can just as
    # well use the transformed values to do this discrete envelope step.
    V_T, P_tp1 = discreteEnvelope(numpy.vstack((Vr_T, Vw_T)), par.sigma)
    # Calculate the expected marginal utility and expected value function
    Eu[conLen:] = par.Rfree*(P_tp1[0, :]*UtilP(Cr_tp1, 1) + P_tp1[1, :]*UtilP(Cw_tp1, 2))
    Ev[conLen:] = numpy.divide(-1.0, V_T)

    # EGM step
    C[conLen:] = UtilP_inv(par.DiscFac*Eu[conLen:], choice)
    M[conLen:] = aGrid + C[conLen:]

    # Add points to M (and C) to solve between 0 and the first point EGM finds
    # (that is add the solution in the segment where the agent is constrained)
    M[0:conLen] = numpy.linspace(numpy.min(aGrid), M[conLen]*0.99, len(M)-aLen)
    C[0:conLen] = M[0:conLen]

    Ev[0:conLen] = Ev[conLen+1]

    V_T = numpy.divide(-1.0, Util(C, choice) + par.DiscFac*Ev)
    # We do the envelope step in transformed value space for accuracy. The values
    # keep their monotonicity under our transformation.
    M, C, V_T = multilineEnvelope(M, C, V_T, mGrid)

    return WorkingDeatonSolution(M, C, V_T)

def discreteEnvelope(Vs, sigma):
    '''
    Returns the final optimal value and policies given the choice specific value
    functions Vs. Policies are degenerate if sigma == 0.0.
    Parameters
    ----------
    Vs : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the taste shocks
    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    P : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    '''
    if sigma == 0.0:
        # Is there a way to fuse these?
        Pflat = numpy.argmax(Vs, axis=0)
        P = numpy.zeros(Vs.shape)
        for i in range(Vs.shape[0]):
            P[i][Pflat==i] = 1
        V = numpy.amax(Vs, axis=0)
        return V, P
    maxV = Vs.max()
    P = numpy.divide(numpy.exp((Vs-Vs[0])/sigma), numpy.sum(numpy.exp((Vs-Vs[0])/sigma), axis=0))
    # GUARD THIS USING numpy error level whatever
    sumexp = numpy.sum(numpy.exp((Vs-maxV)/sigma), axis=0)
    V = numpy.log(sumexp)
    V = maxV + sigma*V
    return V, P

def rise_and_fall(x, v):
    """
    Find index vectors `rise` and `fall` such that `rise` holds the indeces `i`
    such that x[i+1]>x[i] and `fall` holds indeces `j` such that either
    x[j+1] < x[j] or x[j]>x[j-1] but v[j]<v[j-1].

    Parameters
    ----------
    x : numpy.ndarray
        array of points where `v` is evaluated
    v : numpy.ndarray
        array of values of some function of `x`

    Returns
    -------
    rise : numpy.ndarray
        see description above
    fall : numpy.ndarray
        see description above
    """
    # NOTE: assumes that the first segment is in fact increasing (forced in EGM
    # by augmentation with the constrained segment).
    # elements in common grid g
    x_len = len(x)

    # Identify index intervals of falling and rising regions
    # We need these to construct the upper envelope because we need to discard
    # solutions from the inverted Euler equations that do not represent optimal
    # choices (the FOCs are only necessary in these models).
    #
    # `fall` is a vector of indeces that represent the first elements in all
    # of the falling segments (the curve can potentially fold several times)
    fall = numpy.empty(0, dtype=int) # initialize with empty and then add the last point below while-loop

    rise = numpy.array([0]) # Initialize such thatthe lowest point is the first grid point
    i = 1 # Initialize
    while i <= x_len - 2:
        # Check if the next (`ip1` stands for i plus 1) grid point is below the
        # current one, such that the line is folding back.
        ip1_falls = x[i+1] < x[i] # true if grid decreases on index increment
        i_rose = x[i] > x[i-1] # true if grid decreases on index decrement
        val_fell = v[i] < v[i-1] # true if value rises on index decrement

        if (ip1_falls and i_rose) or (val_fell and i_rose):

            # we are in a region where the endogenous grid is decreasing or
            # the value function rises by stepping back in the grid.
            fall = numpy.append(fall, i) # add the index to the vector

            # We now iterate from the current index onwards until we find point
            # where resources rises again. Unfortunately, we need to check
            # each points, as there can be multiple spells of falling endogenous
            # grids, so we cannot use bisection or some other fast algorithm.
            k = i
            while x[k+1] < x[k]:
                k = k + 1
            # k now holds either the next index the starts a new rising
            # region, or it holds the length of M, `m_len`.

            rise = numpy.append(rise, k)

            # Set the index to the point where resources again is rising
            i = k

        i = i + 1
    return rise, fall


# think! nanargmax makes everythign super ugly because numpy changed the wraning
# in all nan slices to a valueerror...it's nans, aaarghgghg
def multilineEnvelope(M, C, V_T, mGrid):
    """
    Do the envelope step of DCEGM.

    Parameters
    ----------

    Returns
    -------


    """
    m_len = len(mGrid)
    rise, fall = rise_and_fall(M, V_T)


    # Add the last point to the vector for convenience below
    fall = numpy.append(fall, m_len)
    # The numbre of kinds are the number of time the grid falls
    num_kinks = len(fall)

    # Use these segments to sequentially find upper envelopes
    mV_T = numpy.empty((m_len, num_kinks))
    mV_T[:] = numpy.nan
    mC = numpy.empty((m_len, num_kinks))
    mC[:] = numpy.nan
    # understand this : # TAKE THE FIRST ONE BY HAND: prevent all the NaN-stuff..
    for j in range(num_kinks):
        # Find all common grid
        below = M[rise[j]] >= mGrid
        above = M[fall[j]] <= mGrid
        in_range = below + above == 0 # neither above nor below
        idxs = range(rise[j], fall[j]+1)
        m_idx_j = M[idxs]
        m_eval = mGrid[in_range]
        mV_T[in_range,j] = LinearInterp(m_idx_j, V_T[idxs], lower_extrap=True)(m_eval)
        mC[in_range,j]  = LinearInterp(m_idx_j, C[idxs], lower_extrap=True)(m_eval) # Interpolat econsumption also. May not be nesserary
    is_all_nan = numpy.array([numpy.all(numpy.isnan(mvrow)) for mvrow in mV_T])
    # Now take the max of all these functions. Since the mV_T
    # is either NaN or very low number outside the range of the actual line-segment this works "globally"
    idx_max = numpy.zeros(mGrid.size, dtype = int) # this one might be wrong if is_all_nan[0] == True
    idx_max[is_all_nan == False] = numpy.nanargmax(mV_T[is_all_nan == False], axis=1)

    # prefix with upper for variable that are "upper enveloped"
    upperV_T = numpy.zeros(mGrid.size)
    upperV_T[:] = numpy.nan

    upperV_T[is_all_nan == False] = numpy.nanmax(mV_T[is_all_nan == False, :], axis=1)
    upperM = numpy.copy(mGrid)
    # Ad the zero point in the bottom
    if numpy.isnan(upperV_T[0]):
        upperV_T[0] = 0 # Since M=0 here
        mC[0,0]  = upperM[1]

    # Extrapolate if NaNs are introduced due to the common grid
    # going outside all the sub-line segments
    IsNaN = numpy.isnan(upperV_T)
    upperV_T[IsNaN] = LinearInterp(upperM[IsNaN == False], upperV_T[IsNaN == False], lower_extrap=True)(upperM[IsNaN])
    upperV_T[IsNaN] = LinearInterp(upperM[IsNaN == False], upperV_T[IsNaN == False], lower_extrap=True)(upperM[IsNaN])
    LastBeforeNaN = numpy.append(numpy.diff(IsNaN)>0, 0)
    LastId = LastBeforeNaN*idx_max # Find last id-number
    idx_max[IsNaN] = LastId[IsNaN]

    # Linear index used to get optimal consumption based on "id"  from max
    ncols = mC.shape[1]
    rowidx = numpy.cumsum(ncols*numpy.ones(len(mGrid), dtype=int))-ncols
    idx_linear = numpy.unravel_index(rowidx+idx_max, mC.shape)
    upperC = mC[idx_linear]
    upperC[IsNaN] = LinearInterp(upperM[IsNaN==0], upperC[IsNaN==0])(upperM[IsNaN])

    # TODO calculate cross points of line segments to get the true vertical drops

    return upperM, upperC, upperV_T
