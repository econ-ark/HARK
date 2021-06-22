import numpy as np
npoints = 20   # number of integer support points of the distribution minus 1
npointsh = npoints // 2
npointsf = float(npoints)
nbound = 4   # bounds for the truncated normal
normbound = (1+1/npointsf) * nbound   # actual bounds of truncated normal
grid = np.arange(-npointsh, npointsh+2, 1)   # integer grid
gridlimitsnorm = (grid-0.5) / npointsh * nbound   # bin limits for the truncnorm
gridlimits = grid - 0.5   # used later in the analysis
grid = grid[:-1]
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
gridint = grid

normdiscrete = stats.rv_discrete(values=(gridint, np.round(probs, decimals=7)), name='normdiscrete')


# The function depends on RVs in the distribution but
# all that we pass to the function is a matrix.

transition_eqns = {
#    'Disc': 'DiscFac * DiscShk',
#    'RFac': 'RFac = Rfree * RShk',
    'RFac': 'RFac = Rfree * RShk',
    'RNrm': 'RNrm = (RFac/(PermGroFac*permShk))',
    'bNrm': 'bNrm = aNrm * RNrm',
    'yNrm': 'yNrm = tranShk',
    'mNrm': 'mNrm = bNrm + yNrm'
    }

transition_eqns_inverse = {
    'aNrmMin': 'aNrmMin = np.asarray(aXtraGrid) + BoroCnstNat_tp1'
    'mNrmMin': 'mNrmMin = aNrmMin + cNrmMin',
    'bNrmMin': 'bNrmMin = mNrmMin - tranShkMin',
    'aNrmGrid': 'aNrmGrid = np.asarray(aXtraGrid) + aNrmMin',
    }

pars = self.soln_crnt.pars

for key in transition_eqns.keys():
    eval(eqns[value], {}, {**bilt.__dict__, **vars})


mNrm_tp1_from_a_t_bcst


def transition_EOP_to_BOP(EOP_states, EOP_shocks, EOP_params, ):
    aNrm = EOP_states[0]
    perm = EOP_shocks[pars.permPos]
    tran = EOP_shocks[pars.tranPos]


def BOP_states_tp1_from_EOP_states_t(EOP_states, EOP_shocks, EOP_params):
    BOP_states =


def transition(states, shocks):
    py___code = '


v_tp1_from_EOP_states_and_transition_shocks(

def expect(function, distribution, *args, **kwds):
    # As of 20210620 we can only handle one type of distribution - our own!
    # TODO: choose an external library (scipy? dolo?) and use it instead of ours
    # choose one in which a pmf is a function (hence the 'f') not list of numbers
    if (type(distribution) != DiscreteDistribution) \
       or (distribution.__module__ != 'HARK.distribution'):
        _log.critical('Distribution must be of type HARK.distribution.DiscreteDistribution')
        return
    # OK, it is a HARK.distributions.DiscreteDistribution object
    for key in distribution.parameters['ShkPosn']:
        setattr(distribution, parameters['ShkPosn']=expect_discrete()

# We need to tell the recipient which variable is where

def expect_using_DiscreteDistribution(v_tp1_from_a_t_Expectable, HARK_distribution_DiscreteDistribution, *args):
    shk_pos_then_args=HARK_distribution_DiscreteDistribution.parameters['ShkPosn']


    f_query=np.apply_along_axis(
        func, 0, dstn_array, *shk_pos_then_args
    )

    return f_exp


def v_tp1_from_a_t_Expectable(PermTranDstn, *args, **kwds):
    permPos=IncShkDstn.parameters['ShkPosn']['perm']
    tranPos=IncShkDstn.parameters['ShkPosn']['tran']

    m_tp1=m_tp1_from_a_t
    v_tp1=vFunc_tp1(m_tp1)
    shks_bcst[pars.permPos] ** (1-pars.CRRA - 0.0)
    v_t_from_a_t=_


def calc_expectations_new(dstn, func_wrapped=lambda z: z, *args, **kwds):
    func_unwrapped=func_wrapped(dstn, *args)

    calc_expectation(dstn, func_unwrapped, *args)

    func_wrapped(dstn, func_wrapped, *args, *kwds)


# def calc_expectations_positional(dstn, func_positional=lambda y: y, *args, **kwds):
#     N = dstn.dim()
#     dstn_array = np.column_stack(dstn.X)
#     if N > 1:
#         # numpy is weird about 1-D arrays.
#         dstn_array = dstn_array.T

#     vars = SimpleNamespace(**dstn.parameters['ShkPosn'])

#     breakpoint()

#     f_query = np.apply_along_axis(
#         func_positional, 0, dstn_array, *args
#     )

#     # Compute expectations over the values
#     f_exp = np.dot(
#         f_query,
#         np.vstack(dstn.pmf)
#     )

#     # a hack.
#     if f_exp.size == 1:
#         f_exp = f_exp.flat[0]
#     elif f_exp.shape[0] == f_exp.size:
#         f_exp = f_exp.flatten()

#     return f_exp

def calc_expect(dstn, func_of_perm_tran=lambda z: z, *args):
    perm_bcst_Pos=dstn.parameters['ShkPosn']['perm']
    tran_bcst_Pos=dstn.parameters['ShkPosn']['tran']
#    perm = dstn.X[permPos]
#    tran = dstn.X[tranPos]
    answer=dstn.X[perm_bcst_Pos]**2 + dstn.X[tran_bcst_Pos]**2
    breakpoint()
