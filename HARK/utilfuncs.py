from typing import Callable, Optional

import numpy as np

from HARK.core import MetricObject
from HARK.utilities import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
    CRRAutilityPPP,
    CRRAutilityPPPP,
    cobb_douglas,
    cobb_douglas_pn,
    const_elast_subs,
    const_elast_subs_p,
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
