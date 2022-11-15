from typing import Callable, Optional

import numpy as np

from HARK.core import MetricObject

# ==============================================================================
# ============== Define utility functions        ===============================
# ==============================================================================


def CRRAutility(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    """

    if gam == 1:
        return np.log(c)
    else:
        return c ** (1.0 - gam) / (1.0 - gam)


def CRRAutilityP(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal utility
    """

    if gam == 1:
        return 1 / c

    return c**-gam


def CRRAutilityPP(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal utility of
    consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal utility
    """

    return -gam * c ** (-gam - 1.0)


def CRRAutilityPPP(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal utility
    """

    return (gam + 1.0) * gam * c ** (-gam - 2.0)


def CRRAutilityPPPP(c, gam):
    """
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    marginal utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal marginal utility
    """

    return -(gam + 2.0) * (gam + 1.0) * gam * c ** (-gam - 3.0)


def CRRAutility_inv(u, gam):
    """
    Evaluates the inverse of the CRRA utility function (with risk aversion para-
    meter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given utility value
    """
    if gam == 1:
        return np.exp(u)
    else:
        return ((1.0 - gam) * u) ** (1 / (1.0 - gam))


def CRRAutilityP_inv(uP, gam):
    """
    Evaluates the inverse of the CRRA marginal utility function (with risk aversion
    parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value.
    """
    return uP ** (-1.0 / gam)


def CRRAutility_invP(u, gam):
    """
    Evaluates the derivative of the inverse of the CRRA utility function (with
    risk aversion parameter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given utility value
    """
    if gam == 1:
        return np.exp(u)
    else:
        return ((1.0 - gam) * u) ** (gam / (1.0 - gam))


def CRRAutilityP_invP(uP, gam):
    """
    Evaluates the derivative of the inverse of the CRRA marginal utility function
    (with risk aversion parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value
    """
    return (-1.0 / gam) * uP ** (-1.0 / gam - 1.0)


def uFunc_CRRA_stone_geary(c, CRRA, stone_geary):
    """
    Evaluates Stone-Geary version of a constant relative risk aversion (CRRA)
    utility of consumption c wiht given risk aversion parameter CRRA and
    Stone-Geary intercept parameter stone_geary

    Parameters
    ----------
    c : float
        Consumption value
    CRRA : float
        Relative risk aversion
    stone_geary : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, CRRA, stone_geary = 1.0, 2.0, 0.0
    >>> utility(c=c, CRRA=CRRA, stone_geary=stone_geary )
    -1.0
    """
    if CRRA == 1:
        return np.log(stone_geary + c)
    else:
        return (stone_geary + c) ** (1.0 - CRRA) / (1.0 - CRRA)


def uPFunc_CRRA_stone_geary(c, CRRA, stone_geary):
    """
    Marginal utility of Stone-Geary version of a constant relative risk aversion (CRRA)
    utility of consumption c wiht given risk aversion parameter CRRA and
    Stone-Geary intercept parameter stone_geary

    Parameters
    ----------
    c : float
        Consumption value
    CRRA : float
        Relative risk aversion
    stone_geary : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        marginal utility

    """
    return (stone_geary + c) ** (-CRRA)


def uPPFunc_CRRA_stone_geary(c, CRRA, stone_geary):
    """
    Marginal marginal utility of Stone-Geary version of a CRRA utilty function
    with risk aversion parameter CRRA and Stone-Geary intercept parameter stone_geary

    Parameters
    ----------
    c : float
        Consumption value
    CRRA : float
        Relative risk aversion
    stone_geary : float
        Intercept in Stone-Geary utility
    Returns
    -------
    (unnamed) : float
        marginal utility

    """
    return (-CRRA) * (stone_geary + c) ** (-CRRA - 1)


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
    return 1.0 / (alpha * (1.0 - u))


def cobb_douglas(x, alpha, factor):
    """
    Evaluates Cobb Douglas utility at quatitites of goods consumed `x`
    given elasticity parameters `alpha` and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. Last axis must index goods.
    alpha : np.ndarray
        Elasticity parameters for each good. Must be consistent with `x`.
    factor : float
        Multiplicative efficiency parameter. (e.g. TFP in production function)

    Returns
    -------
    (unnamed) : np.ndarray
        Utility

    """

    return factor * np.sum(x**alpha, axis=-1)


def cobb_douglas_p(x, alpha, factor, arg=0):
    """
    Evaluates the marginal utility of consumption indexed by `arg` good at
    quantities of goods consumed `x` given elasticity parameters `alpha`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. Last axis must index goods.
    alpha : np.ndarray
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

    return cobb_douglas(x, alpha, factor) * alpha[arg] / x[..., arg]


def cobb_douglas_pp(x, alpha, factor, args=(0, 1)):
    """
    Evaluates the marginal marginal utility of consumption indexed by `args`
    at quantities of goods consumed `x` given elasticity parameters `alpha`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. Last axis must index goods.
    alpha : np.ndarray
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
        coeff = alpha[args[1]] - 1
    else:
        coeff = alpha[args[1]]

    return cobb_douglas_p(x, alpha, factor, args[0]) * coeff / x[..., args[1]]


def cobb_douglas_pn(x, alpha, factor, args=()):
    """
    Evaluates the nth marginal utility of consumption indexed by `args`
    at quantities of goods consumed `x` given elasticity parameters `alpha`
    and efficiency parameter `factor`.

    Parameters
    ----------
    x : np.ndarray
        Quantities of goods consumed. Last axis must index goods.
    alpha : np.ndarray
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
        counts = dict(zip(*np.unique(args, return_counts=True)))
        idx = args[-1]
        coeff = alpha[idx] - counts[idx] + 1
        new_args = tuple(list(args)[:-1])
        return cobb_douglas_pn(x, alpha, factor, new_args) * coeff / x[..., idx]
    else:
        return cobb_douglas(x, alpha, factor)


def const_elast_subs(x, alpha, subs, factor, power):

    return factor * np.sum(alpha * x**subs, axis=-1) ** (power / subs)


def const_elast_subs_p(x, alpha, subs, factor, power, arg=0):

    return (
        const_elast_subs(x, alpha, factor * power / subs, subs, power - subs)
        * alpha[arg]
        * subs
        * x[..., arg] ** (subs - 1)
    )


class UtilityFunction(MetricObject):

    distance_criteria = ["eval_func", "deriv_func", "inv_func"]

    def __init__(self, eval_func, deriv_func=None, inv_func=None):
        self.eval_func = eval_func
        self.deriv_func = deriv_func
        self.inv_func = inv_func

    def __call__(self, *args):

        return self.eval_func(*args)

    def derivative(self, *args, **kwargs):

        return self.deriv_func(*args, **kwargs)

    def inverse(self, *args, **kwargs):

        return self.inv_func(*args, **kwargs)

    def der(self, *args, **kwargs):

        return self.derivative(*args, **kwargs)

    def inv(self, *args, **kwargs):

        return self.inverse(*args, **kwargs)


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
            raise ValueError("Derivative of order {} not supported".format(order))

    def inverse(self, u, order=(0, 0)):
        """
        The inverse of the utility function at a given level of utility u.

        Parameters
        ----------
        u : float or np.ndarray
            Utility level(s).
        order : tuple, optional
            Order of derivatives. For example, `order == (1,1)` represents
            the first derivative of utility, inversed, and then differenciated
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
            raise ValueError("Inverse of order {} not supported".format(order))

    def der(self, c, order=1):
        """
        Short alias for derivative. See `self.derivative`.
        """
        return self.derivative(c, order)

    def inv(self, c, order=(0, 0)):
        """
        Short alias for inverse. See `self.inverse`.
        """
        return self.inverse(c, order)

    def derinv(self, u, order=(1, 0)):
        """
        Short alias for inverse. See `self.inverse`.
        """
        return self.inverse(u, order)


class UtilityFuncCobbDouglas(UtilityFunction):
    def __init__(self, EOS, factor=1.0):
        self.EOS = np.asarray(EOS)
        self.factor = factor

        assert np.isclose(
            np.sum(self.EOS), 1.0
        ), """The sum of the elasticity of substitution 
        parameteres must be less than or equal to 1."""

        assert factor > 0, "Factor must be positive."

        self.dim = len(self.EOS)  # number of goods

    def __call__(self, x):
        assert self.EOS.size == x.shape[-1], "x must be compatible with EOS"
        cobb_douglas(x, self.EOS, self.factor)

    def derivative(self, x, args=()):
        assert self.EOS.size == x.shape[-1], "x must be compatible with EOS"
        return cobb_douglas_pn(x, self.EOS, self.factor, args)


class UtilityFuncConstElastSubs(UtilityFunction):
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
