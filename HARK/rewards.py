import numpy as np

from HARK.metric import MetricObject

# ==============================================================================
# ============== Define utility functions        ===============================
# ==============================================================================


def CRRAutility(c, rho):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter rho.

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, CRRA = 1.0, 2.0    # Set two values at once with Python syntax
    >>> CRRAutility(c=c, rho=CRRA)
    -1.0
    """

    if np.isscalar(c):
        c = np.asarray(c)
    if rho == 1:
        return np.log(c)

    return c ** (1.0 - rho) / (1.0 - rho)


def CRRAutilityP(c, rho):
    """
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter rho.

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    if rho == 1:
        return 1 / c

    return c**-rho


def CRRAutilityPP(c, rho):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal utility of
    consumption c given risk aversion parameter rho.

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return -rho * c ** (-rho - 1.0)


def CRRAutilityPPP(c, rho):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    utility of consumption c given risk aversion parameter rho.

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return (rho + 1.0) * rho * c ** (-rho - 2.0)


def CRRAutilityPPPP(c, rho):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    marginal utility of consumption c given risk aversion parameter rho.

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return -(rho + 2.0) * (rho + 1.0) * rho * c ** (-rho - 3.0)


def CRRAutility_inv(u, rho):
    """
    Evaluates the inverse of the CRRA utility function (with risk aversion para-
    meter rho) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given utility value
    """

    if np.isscalar(u):
        u = np.asarray(u)
    if rho == 1:
        return np.exp(u)

    return ((1.0 - rho) * u) ** (1 / (1.0 - rho))


def CRRAutilityP_inv(uP, rho):
    """
    Evaluates the inverse of the CRRA marginal utility function (with risk aversion
    parameter rho) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value.
    """

    if np.isscalar(uP):
        uP = np.asarray(uP)
    return uP ** (-1.0 / rho)


def CRRAutility_invP(u, rho):
    """
    Evaluates the derivative of the inverse of the CRRA utility function (with
    risk aversion parameter rho) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given utility value
    """

    if np.isscalar(u):
        u = np.asarray(u)
    if rho == 1:
        return np.exp(u)

    return ((1.0 - rho) * u) ** (rho / (1.0 - rho))


def CRRAutilityP_invP(uP, rho):
    """
    Evaluates the derivative of the inverse of the CRRA marginal utility function
    (with risk aversion parameter rho) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    rho : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value
    """

    if np.isscalar(uP):
        uP = np.asarray(uP)
    return (-1.0 / rho) * uP ** (-1.0 / rho - 1.0)


def StoneGearyCRRAutility(c, rho, shifter, factor=1.0):
    """
    Evaluates Stone-Geary version of a constant relative risk aversion (CRRA)
    utility of consumption c with given risk aversion parameter rho and
    Stone-Geary intercept parameter shifter

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Relative risk aversion
    shifter : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, CRRA, stone_geary = 1.0, 2.0, 0.0
    >>> StoneGearyCRRAutility(c=c, rho=CRRA, shifter=stone_geary)
    -1.0
    """

    if np.isscalar(c):
        c = np.asarray(c)
    if rho == 1:
        return factor * np.log(shifter + c)

    return factor * (shifter + c) ** (1.0 - rho) / (1.0 - rho)


def StoneGearyCRRAutilityP(c, rho, shifter, factor=1.0):
    """
    Marginal utility of Stone-Geary version of a constant relative risk aversion (CRRA)
    utility of consumption c with a given risk aversion parameter rho and
    Stone-Geary intercept parameter shifter

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Relative risk aversion
    shifter : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        marginal utility

    """

    if np.isscalar(c):
        c = np.asarray(c)
    return factor * (shifter + c) ** (-rho)


def StoneGearyCRRAutilityPP(c, rho, shifter, factor=1.0):
    """
    Marginal marginal utility of Stone-Geary version of a CRRA utilty function
    with risk aversion parameter rho and Stone-Geary intercept parameter shifter

    Parameters
    ----------
    c : float
        Consumption value
    rho : float
        Relative risk aversion
    shifter : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        marginal utility

    """

    if np.isscalar(c):
        c = np.asarray(c)
    return factor * (-rho) * (shifter + c) ** (-rho - 1)


def StoneGearyCRRAutility_inv(u, rho, shifter, factor=1.0):
    if np.isscalar(u):
        u = np.asarray(u)

    return (u * (1.0 - rho) / factor) ** (1.0 / (1.0 - rho)) - shifter


def StoneGearyCRRAutilityP_inv(uP, rho, shifter, factor=1.0):
    if np.isscalar(uP):
        uP = np.asarray(uP)

    return (uP / factor) ** (-1.0 / rho) - shifter


def StoneGearyCRRAutility_invP(u, rho, shifter, factor=1.0):
    if np.isscalar(u):
        u = np.asarray(u)

    return (1.0 / (1.0 - rho)) * (u * (1.0 - rho) / factor) ** (1.0 / (1.0 - rho) - 1.0)


def StoneGearyCRRAutilityP_invP(uP, rho, shifter, factor=1.0):
    if np.isscalar(uP):
        uP = np.asarray(uP)

    return (-1.0 / rho) * (uP / factor) ** (-1.0 / rho - 1.0)


def CARAutility(c, alpha):
    """
    Evaluates constant absolute risk aversion (CARA) utility of consumption c
    given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return 1 - np.exp(-alpha * c) / alpha


def CARAutilityP(c, alpha):
    """
    Evaluates constant absolute risk aversion (CARA) marginal utility of
    consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return np.exp(-alpha * c)


def CARAutilityPP(c, alpha):
    """
    Evaluates constant absolute risk aversion (CARA) marginal marginal utility
    of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return -alpha * np.exp(-alpha * c)


def CARAutilityPPP(c, alpha):
    """
    Evaluates constant absolute risk aversion (CARA) marginal marginal marginal
    utility of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal marginal utility
    """

    if np.isscalar(c):
        c = np.asarray(c)
    return alpha**2.0 * np.exp(-alpha * c)


def CARAutility_inv(u, alpha):
    """
    Evaluates inverse of constant absolute risk aversion (CARA) utility function
    at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to u
    """

    if np.isscalar(u):
        u = np.asarray(u)
    return -1.0 / alpha * np.log(alpha * (1 - u))


def CARAutilityP_inv(u, alpha):
    """
    Evaluates the inverse of constant absolute risk aversion (CARA) marginal
    utility function at marginal utility uP given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to uP
    """

    if np.isscalar(u):
        u = np.asarray(u)
    return -1.0 / alpha * np.log(u)


def CARAutility_invP(u, alpha):
    """
    Evaluates the derivative of inverse of constant absolute risk aversion (CARA)
    utility function at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal onsumption value corresponding to u
    """

    if np.isscalar(u):
        u = np.asarray(u)
    return 1.0 / (alpha * (1.0 - u))


def cobb_douglas(x, zeta, factor):
    """
    Evaluates Cobb Douglas utility at quantities of goods consumed `x`
    given elasticity parameters `zeta` and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : np.ndarray
        Elasticity parameters for each good. Must be consistent with `x`.
    factor : float
        Multiplicative efficiency parameter. (e.g. TFP in production function)

    Returns
    -------
    (unnamed) : np.ndarray
        Utility

    """

    # move goods axis to the end
    goods = np.moveaxis(x, 0, -1)

    return factor * np.sum(goods**zeta, axis=-1)


def cobb_douglas_p(x, zeta, factor, arg=0):
    """
    Evaluates the marginal utility of consumption indexed by `arg` good at
    quantities of goods consumed `x` given elasticity parameters `zeta`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : np.ndarray
        Elasticity parameters for each good. Must be consistent with `x`.
    factor : float
        Multiplicative efficiency parameter.
    arg : int
        Index of good to evaluate marginal utility.

    Returns
    -------
    (unnamed) : np.ndarray
        Utility
    """

    return cobb_douglas(x, zeta, factor) * zeta[arg] / x[arg]


def cobb_douglas_pp(x, zeta, factor, args=(0, 1)):
    """
    Evaluates the marginal marginal utility of consumption indexed by `args`
    at quantities of goods consumed `x` given elasticity parameters `zeta`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : np.ndarray
        Elasticity parameters for each good. Must be consistent with `x`.
    factor : float
        Multiplicative efficiency parameter.
    args : tuple(int)
        Indexes of goods to evaluate marginal utility. `args[0]` is the
        index of the first derivative taken, and `args[1]` is the index of
        the second derivative taken.

    Returns
    -------
    (unnamed) : np.ndarray
        Utility
    """

    if args[0] == args[1]:
        coeff = zeta[args[0]] - 1
    else:
        coeff = zeta[args[1]]

    return cobb_douglas_p(x, zeta, factor, args[0]) * coeff / x[args[1]]


def cobb_douglas_pn(x, zeta, factor, args=()):
    """
    Evaluates the nth marginal utility of consumption indexed by `args`
    at quantities of goods consumed `x` given elasticity parameters `zeta`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : np.ndarray
        Elasticity parameters for each good. Must be consistent with `x`.
    factor : float
        Multiplicative efficiency parameter.
    args : tuple(int)
        Indexes of goods to evaluate marginal utility. `args[0]` is the
        index of the first derivative taken, and `args[1]` is the index of
        the second derivative taken. This function works by recursively taking
        derivatives, so `args` can be of any length.

    Returns
    -------
    (unnamed) : np.ndarray
        Utility
    """

    if isinstance(args, int):
        args = (args,)

    if len(args):
        idx = args[-1]  # last index
        counts = dict(zip(*np.unique(args, return_counts=True)))
        coeff = zeta[idx] - counts[idx] + 1
        new_args = tuple(list(args)[:-1])  # remove last element
        return cobb_douglas_pn(x, zeta, factor, new_args) * coeff / x[idx]
    else:
        return cobb_douglas(x, zeta, factor)


def const_elast_subs(x, zeta, subs, factor, power):
    """
    Evaluates Constant Elasticity of Substitution utility at quantities of
    goods consumed `x` given parameters `alpha`, 'subs', 'factor', and 'power'.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : Sequence[float]
        Share parameter for each good. Must be consistent with `x`.
    subs : float
        Substitution parameter.
    factor : float
        Factor productivity parameter. (e.g. TFP in production function)
    power : float
        degree of homogeneity of the utility function

    Returns
    -------
    np.ndarray
        CES utility.
    """

    # move goods axis to the end
    goods = np.moveaxis(x, 0, -1)

    return factor * np.sum(zeta * goods**subs, axis=-1) ** (power / subs)


def const_elast_subs_p(x, zeta, subs, factor, power, arg=0):
    """
    Evaluates the marginal utility of consumption indexed by `arg` good at quantities
    of goods consumed `x` given parameters `alpha`, 'subs', 'factor', and 'power'.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. First axis must index goods.
    zeta : Sequence[float]
        Share parameter for each good. Must be consistent with `x`.
    subs : float
        Substitution parameter.
    factor : float
        Factor productivity parameter. (e.g. TFP in production function)
    power : float
        degree of homogeneity of the utility function

    Returns
    -------
    np.ndarray
        CES marginal utility.
    """

    return (
        const_elast_subs(x, zeta, factor * power / subs, subs, power - subs)
        * zeta[arg]
        * subs
        * x[arg] ** (subs - 1)
    )


class UtilityFunction(MetricObject):
    distance_criteria = ["eval_func"]

    def __init__(self, eval_func, der_func=None, inv_func=None):
        self.eval_func = eval_func
        self.der_func = der_func
        self.inv_func = inv_func

    def __call__(self, *args, **kwargs):
        return self.eval_func(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        if not hasattr(self, "der_func") or self.der_func is None:
            raise NotImplementedError("No derivative function available")
        return self.der_func(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        if not hasattr(self, "inv_func") or self.inv_func is None:
            raise NotImplementedError("No inverse function available")
        return self.inv_func(*args, **kwargs)

    def der(self, *args, **kwargs):
        return self.derivative(*args, **kwargs)

    def inv(self, *args, **kwargs):
        return self.inverse(*args, **kwargs)


def CDutility(c, d, c_share, d_bar):
    return c**c_share * (d + d_bar) ** (1 - c_share)


def CDutilityPc(c, d, c_share, d_bar):
    return c_share * ((d + d_bar) / c) ** (1 - c_share)


def CDutilityPd(c, d, c_share, d_bar):
    return (1 - c_share) * (c / (d + d_bar)) ** c_share


def CDutilityPc_inv(uc, d, c_share, d_bar):
    return (d + d_bar) * (uc / c_share) ** (1 / (1 - c_share))


def CRRACDutility(c, d, c_share, d_bar, CRRA):
    return CDutility(c, d, c_share, d_bar) ** (1 - CRRA) / (1 - CRRA)


def CRRACDutilityPc(c, d, c_share, d_bar, CRRA):
    return c_share / c * CDutility(c, d, c_share, d_bar) ** (1 - CRRA)


def CRRACDutilityPd(c, d, c_share, d_bar, CRRA):
    return (1 - c_share) / (d + d_bar) * CDutility(c, d, c_share, d_bar) ** (1 - CRRA)


def CRRACDutilityPc_inv(uc, d, c_share, d_bar, CRRA):
    return (c_share / uc * (d + d_bar) ** (c_share * CRRA - c_share - CRRA + 1)) ** (
        1 / (c_share * CRRA - c_share + 1)
    )


class UtilityFuncCRRA(UtilityFunction):
    """
    A class for representing a CRRA utility function.

    Parameters
    ----------
    CRRA : float
        The coefficient of constant relative risk aversion.
    """

    distance_criteria = ["CRRA"]

    def __init__(self, CRRA):
        self.CRRA = CRRA

    def __call__(self, c, order=0):
        """
        Evaluate the utility function at a given level of consumption c.

        Parameters
        ----------
        c : float or np.ndarray
            Consumption level(s).
        order : int, optional
            Order of derivative. For example, `order == 1` returns the
            first derivative of utility of consumption, and so on. By default 0.

        Returns
        -------
        float or np.ndarray
            Utility (or its derivative) evaluated at given consumption level(s).
        """
        if order == 0:
            return CRRAutility(c, self.CRRA)
        else:  # order >= 1
            return self.derivative(c, order)

    def derivative(self, c, order=1):
        """
        The derivative of the utility function at a given level of consumption c.

        Parameters
        ----------
        c : float or np.ndarray
            Consumption level(s).
        order : int, optional
            Order of derivative. For example, `order == 1` returns the
            first derivative of utility of consumption, and so on. By default 1.

        Returns
        -------
        float or np.ndarray
            Derivative of CRRA utility evaluated at given consumption level(s).

        Raises
        ------
        ValueError
            Derivative of order higher than 4 is not supported.
        """
        if order == 1:
            return CRRAutilityP(c, self.CRRA)
        elif order == 2:
            return CRRAutilityPP(c, self.CRRA)
        elif order == 3:
            return CRRAutilityPPP(c, self.CRRA)
        elif order == 4:
            return CRRAutilityPPPP(c, self.CRRA)
        else:
            raise ValueError(f"Derivative of order {order} not supported")

    def inverse(self, u, order=(0, 0)):
        """
        The inverse of the utility function at a given level of utility u.

        Parameters
        ----------
        u : float or np.ndarray
            Utility level(s).
        order : tuple, optional
            Order of derivatives. For example, `order == (1,1)` represents
            the first derivative of utility, inverted, and then differentiated
            once. For a simple mnemonic, order refers to the number of `P`s in
            the function `CRRAutility[#1]_inv[#2]`. By default (0, 0),
            which is just the inverse of utility.

        Returns
        -------
        float or np.ndarray
            Inverse of CRRA utility evaluated at given utility level(s).

        Raises
        ------
        ValueError
            Higher order derivatives are not supported.
        """
        if order == (0, 0):
            return CRRAutility_inv(u, self.CRRA)
        elif order == (1, 0):
            return CRRAutilityP_inv(u, self.CRRA)
        elif order == (0, 1):
            return CRRAutility_invP(u, self.CRRA)
        elif order == (1, 1):
            return CRRAutilityP_invP(u, self.CRRA)
        else:
            raise ValueError(f"Inverse of order {order} not supported")

    def derinv(self, u, order=(1, 0)):
        """
        Short alias for inverse with default order = (1,0). See `self.inverse`.
        """
        return self.inverse(u, order)


class UtilityFuncStoneGeary(UtilityFuncCRRA):
    def __init__(self, CRRA, factor=1.0, shifter=0.0):
        self.CRRA = CRRA
        self.factor = factor
        self.shifter = shifter

    def __call__(self, c, order=0):
        if order == 0:
            return StoneGearyCRRAutility(c, self.CRRA, self.shifter, self.factor)
        else:  # order >= 1
            return self.derivative(c, order)

    def derivative(self, c, order=1):
        if order == 1:
            return StoneGearyCRRAutilityP(c, self.CRRA, self.shifter, self.factor)
        elif order == 2:
            return StoneGearyCRRAutilityPP(c, self.CRRA, self.shifter, self.factor)
        else:
            raise ValueError(f"Derivative of order {order} not supported")

    def inverse(self, u, order=(0, 0)):
        if order == (0, 0):
            return StoneGearyCRRAutility_inv(u, self.CRRA, self.shifter, self.factor)
        elif order == (1, 0):
            return StoneGearyCRRAutilityP_inv(u, self.CRRA, self.shifter, self.factor)
        elif order == (0, 1):
            return StoneGearyCRRAutility_invP(u, self.CRRA, self.shifter, self.factor)
        elif order == (1, 1):
            return StoneGearyCRRAutilityP_invP(u, self.CRRA, self.shifter, self.factor)
        else:
            raise ValueError(f"Inverse of order {order} not supported")


class UtilityFuncCobbDouglas(UtilityFunction):
    """
    A class for representing a Cobb-Douglas utility function.

    TODO: Add inverse methods.

    Parameters
    ----------
    c_share : float
        Share parameter for consumption. Must be between 0 and 1.
    d_bar : float
        Intercept parameter for durable consumption. Must be non-negative.
    """

    distance_criteria = ["c_share", "d_bar"]

    def __init__(self, c_share, d_bar=0):
        self.c_share = c_share
        self.d_bar = d_bar

        assert 0 <= c_share <= 1, "Share parameter must be between 0 and 1."

    def __call__(self, c, d):
        """
        Evaluate the utility function at a given level of consumption c.
        """
        return CDutility(c, d, self.c_share, self.d_bar)

    def derivative(self, c, d, axis=0):
        if axis == 0:
            return CDutilityPc(c, d, self.c_share, self.d_bar)
        elif axis == 1:
            return CDutilityPd(c, d, self.c_share, self.d_bar)
        else:
            raise ValueError(f"Axis must be 0 or 1, not {axis}")

    def inverse(self, uc, d):
        return CDutilityPc_inv(uc, d, self.c_share, self.d_bar)


class UtilityFuncCobbDouglasCRRA(UtilityFuncCobbDouglas):
    """
    A class for representing a Cobb-Douglas aggregated CRRA utility function.

    TODO: Add derivative and inverse methods.

    Parameters
    ----------
    c_share : float
        Share parameter for consumption. Must be between 0 and 1.
    d_bar : float
        Intercept parameter for durable consumption. Must be non-negative.
    CRRA: float
        Coefficient of relative risk aversion.
    """

    distance_criteria = ["c_share", "d_bar", "CRRA"]

    def __init__(self, c_share, CRRA, d_bar=0):
        super().__init__(c_share, d_bar)
        self.CRRA = CRRA

    def __call__(self, c, d):
        return CRRACDutility(c, d, self.c_share, self.d_bar, self.CRRA)

    def derivative(self, c, d, axis=0):
        if axis == 0:
            return CRRACDutilityPc(c, d, self.c_share, self.d_bar, self.CRRA)
        elif axis == 1:
            return CRRACDutilityPd(c, d, self.c_share, self.d_bar, self.CRRA)
        else:
            raise ValueError(f"Axis must be 0 or 1, not {axis}")

    def inverse(self, uc, d):
        return CRRACDutilityPc_inv(uc, d, self.c_share, self.d_bar, self.CRRA)


class UtilityFuncConstElastSubs(UtilityFunction):
    """
    A class for representing a constant elasticity of substitution utility function.

    TODO: Add derivative and inverse methods.

    Parameters
    ----------
    shares : Sequence[float]
        Share parameter for each good. Must be consistent with `x`.
    subs : float
        Substitution parameter.
    factor : float
        Factor productivity parameter. (e.g. TFP in production function)
    homogeneity : float
        degree of homogeneity of the utility function
    """

    distance_criteria = ["shares", "subs", "factor", "homogeneity"]

    def __init__(self, shares, subs, homogeneity=1.0, factor=1.0):
        assert subs != 0.0, "Consider using a Cobb-Douglas utility function instead."
        assert subs != 1.0, "Linear utility is not implemented."

        self.shares = np.asarray(shares)
        self.subs = subs
        self.homogeneity = homogeneity
        self.factor = factor

        self.CES = 1 / (1 - subs)
        self.dim = self.shares.size  # number of goods

    def __call__(self, x):
        return const_elast_subs(
            x, self.shares, self.subs, self.factor, self.homogeneity
        )

    def derivative(self, x, arg=0):
        return const_elast_subs_p(
            x, self.shares, self.subs, self.factor, self.homogeneity, arg
        )
