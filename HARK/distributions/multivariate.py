from itertools import product
from typing import List, Union

import numpy as np
from numpy import linalg
from scipy import special
from scipy.stats._multivariate import multi_rv_frozen, multivariate_normal_frozen

from HARK.distributions.base import Distribution
from HARK.distributions.continuous import Lognormal, Normal
from HARK.distributions.discrete import DiscreteDistribution

# MULTIVARIATE DISTRIBUTIONS


class MultivariateNormal(multivariate_normal_frozen, Distribution):
    """
    A Multivariate Normal distribution.

    Parameters
    ----------
    mu : numpy array
        Mean vector.
    Sigma : 2-d numpy array. Each dimension must have length equal to that of
            mu.
        Variance-covariance matrix.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, mu=[1, 1], Sigma=[[1, 0], [0, 1]], seed=None):
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.M = self.mu.size
        multivariate_normal_frozen.__init__(self, mean=self.mu, cov=self.Sigma)
        Distribution.__init__(self, seed=seed)
        self.infimum = -np.inf * np.ones(self.M)
        self.supremum = np.inf * np.ones(self.M)

    def discretize(self, N, method="hermite", endpoints=False):
        """
        For multivariate normal distributions, the Gauss-Hermite
        quadrature rule is used as the default method for discretization.
        """

        return self._approx(N, method=method, endpoints=endpoints)

    def _approx(self, N, method="hermite", endpoints=False):
        """
        Returns a discrete approximation of this distribution.

        The discretization will have N**M points, where M is the dimension of
        the multivariate normal.

        It uses the fact that:
            - Being positive definite, Sigma can be factorized as Sigma = QVQ',
              with V diagonal. So, letting A=Q*sqrt(V), Sigma = A*A'.
            - If Z is an N-dimensional multivariate standard normal, then
              A*Z ~ N(0,A*A' = Sigma).

        The idea therefore is to construct an equiprobable grid for a standard
        normal and multiply it by matrix A.
        """

        # Start by computing matrix A.
        v, Q = np.linalg.eig(self.Sigma)
        sqrtV = np.diag(np.sqrt(v))
        A = np.matmul(Q, sqrtV)

        # Now find a discretization for a univariate standard normal.

        z_approx = Normal().discretize(N, method=method)

        # Now create the multivariate grid and pmv
        Z = np.array(list(product(*[z_approx.atoms.flatten()] * self.M)))
        pmv = np.prod(np.array(list(product(*[z_approx.pmv] * self.M))), axis=1)

        # Apply mean and standard deviation to the Z grid
        atoms = self.mu[None, ...] + np.matmul(Z, A.T)

        limit = {
            "dist": self,
            "method": method,
            "N": N,
            "endpoints": endpoints,
        }

        # Construct and return discrete distribution
        return DiscreteDistribution(
            pmv,
            atoms.T,
            seed=self.random_seed(),
            limit=limit,
        )


class MultivariateLogNormal(multi_rv_frozen, Distribution):
    """
    A Multivariate Lognormal distribution.

    Parameters
    ----------
    mu : Union[list, numpy.ndarray], optional
        Means of underlying multivariate normal, default [0.0, 0.0].
    Sigma : Union[list, numpy.ndarray], optional
        nxn variance-covariance matrix of underlying multivariate normal, default [[1.0, 0.0], [0.0, 1.0]].
    seed : int, optional
        Seed for random number generator, default 0.
    """

    def __init__(
        self,
        mu: Union[List, np.ndarray] = [0.0, 0.0],
        Sigma: Union[List, np.ndarray] = [[1.0, 0.0], [0.0, 1.0]],
        seed=None,
    ):
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.M = self.mu.size

        if self.Sigma.shape != (self.M, self.M):
            raise AttributeError(f"Sigma must be a {self.M}x{self.M} matrix")

        if np.array_equal(self.Sigma, self.Sigma.T):
            res = np.all(np.linalg.eigvals(self.Sigma) >= 0)
            if not res:
                raise AttributeError("Sigma must be positive semi-definite")
        else:
            raise AttributeError("Sigma must be positive semi-definite")

        multi_rv_frozen.__init__(self)
        Distribution.__init__(self, seed=seed)
        self.dstn = MultivariateNormal(mu=self.mu, Sigma=self.Sigma)

    def mean(self):
        """
        Mean of the distribution.

        Returns
        -------
        np.ndarray
            Mean of the distribution.
        """

        return np.exp(self.mu + 0.5 * np.diag(self.Sigma))

    def _cdf(self, x: Union[list, np.ndarray]):
        """
        Cumulative distribution function of the distribution evaluated at x.

        Parameters
        ----------
        x : np.ndarray
            Point at which to evaluate the CDF.

        Returns
        -------
        float
            CDF evaluated at x.
        """

        x = np.asarray(x)
        if (x.shape != self.M) & (x.shape[1] != self.M):
            raise ValueError(f"x must be and {self.M}-dimensional input")
        return self.dstn.cdf(np.log(x))

    def _pdf(self, x: Union[list, np.ndarray]):
        """
        Probability density function of the distribution evaluated at x.

        Parameters
        ----------
        x : Union[list, np.ndarray]
            Point at which to evaluate the PDF.

        Returns
        -------
        float
            PDF evaluated at x.
        """

        x = np.asarray(x)

        if (x.shape != (self.M,)) & (x.shape[1] != self.M):
            raise ValueError(f"x must be an {self.M}-dimensional input")

        eigenvalues = linalg.eigvals(self.Sigma)

        pseudo_det = np.prod(eigenvalues[eigenvalues > 1e-12])

        inverse_sigma = linalg.pinv(self.Sigma)

        rank_sigma = linalg.matrix_rank(self.Sigma)

        pd = np.multiply(
            (1 / np.prod(x, axis=1, keepdims=True)),
            (2 * np.pi) ** (-rank_sigma / 2)
            * pseudo_det ** (-0.5)
            * np.exp(-(1 / 2) * np.multiply(np.log(x) @ inverse_sigma, np.log(x))),
        )
        return pd

    def _marginal(self, x: Union[np.ndarray, float, list], dim: int):
        """
        Marginal distribution of one of the variables in the bivariate distribution evaluated at x.

        Parameters
        ----------
        x : Union[np.ndarray, float]
            Point at which to evaluate the marginal distribution.
        dim : int
            Which of the random variables to evaluate.

        Returns
        -------
        float
            Marginal distribution evaluated at x.
        """

        x = np.asarray(x)
        x_dim = Lognormal(mu=self.mu[dim], sigma=np.sqrt(self.Sigma[dim, dim]))
        return x_dim.pdf(x)

    def _marginal_cdf(self, x: Union[np.ndarray, float, list], dim: int):
        """
        Cumulative distribution function of one of the variables from the bivariate distribution evaluated at x.

        Parameters
        ----------
        x : Union[np.ndarray, float]
            Point at which to evaluate the marginal CDF.
        dim : int
            Which of the random variables to evaluate.

        Returns
        -------
        float
            Marginal CDF evaluated at x.
        """

        x = np.asarray(x)
        x_dim = Lognormal(mu=self.mu[dim], sigma=np.sqrt(self.Sigma[dim, dim]))
        return x_dim.cdf(x)

    def rvs(self, size: int = 1, random_state=None):
        """
        Random sample from the distribution.

        Parameters
        ----------
        size : int
            Number of data points to generate.
        random_state : optional
            Seed for random number generator.

        Returns
        -------
        np.ndarray
            Random sample from the distribution.
        """
        return np.exp(self.dstn.rvs(size, random_state=random_state))

    def _approx_equiprobable(
        self,
        N: int,
        endpoints: bool = False,
        tail_bound: Union[float, list, tuple] = None,
        decomp: str = "cholesky",
    ):
        """
        Makes a discrete approximation using the equiprobable method to this multi-
        variate lognormal distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation.
        tail_bound : Union[float, list, tuple], optional
            The values of the CDF according to which the distribution is truncated.
            If only a single number is specified, it is the lower tail bound and a
            symmetric upper bound is chosen. Can make one-tailed approximations
            with 0.0 or 1.0 as the lower and upper bound respectively. By default
            the distribution is not truncated.
        endpoints : bool
            If endpoints is True, then atoms at the corner points of the truncated
            region are included. By default, endpoints is False, which is when only
            the interior points are included.
        decomp : str in ["cholesky", "sqrt", "eig"], optional
            The method of decomposing the covariance matrix. Available options are
            the Cholesky decomposition, the positive-definite square root, and the
            eigendecomposition. By default the Cholesky decomposition is used.
            NOTE: The method of decomposition might affect the expectations of the
            discretized distribution along each dimension dfferently.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """

        if endpoints:
            tail_N = 1
        else:
            tail_N = 0

        if decomp not in ["cholesky", "sqrt", "eig"]:
            raise NotImplementedError(
                "Decomposition method must be 'cholesky', 'sqrt' or 'eig'"
            )

        if np.array_equal(self.Sigma, np.diag(np.diag(self.Sigma))):
            ind_atoms = np.empty((self.M, N + 2 * tail_N))

            for i in range(self.M):
                if self.Sigma[i, i] == 0.0:
                    x_atoms = np.repeat(np.exp(self.mu[i]), N + 2 * tail_N)
                    ind_atoms[i] = x_atoms
                else:
                    x_atoms = (
                        Lognormal(mu=self.mu[i], sigma=np.sqrt(self.Sigma[i, i]))
                        ._approx_equiprobable(
                            N, tail_N=tail_N, tail_bound=tail_bound, endpoints=endpoints
                        )
                        .atoms
                    )
                    ind_atoms[i] = x_atoms

            atoms_list = [ind_atoms[i] for i in range(self.M)]
            atoms = np.stack(
                [ar.flatten() for ar in list(np.meshgrid(*atoms_list))], axis=1
            ).T

            interiors = np.empty([self.M, (N + 2 * tail_N) ** (self.M)])

            inners = np.zeros(N + 2 * tail_N)

            if tail_N > 0:
                inners[:tail_N] = [(tail_N - i) for i in range(tail_N)]
                inners[-tail_N:] = [(i + 1) for i in range(tail_N)]

            for i in range(self.M):
                inners_i = [inners for _ in range((N + 2 * tail_N) ** i)]

                interiors[i] = np.repeat(
                    [*inners_i], (N + 2 * tail_N) ** (self.M - (i + 1))
                )

        else:
            if tail_bound is not None:
                if type(tail_bound) is float:
                    tail_bound = [tail_bound, 1 - tail_bound]

                if tail_bound[0] < 0 or tail_bound[1] > 1:
                    raise ValueError("Tail bounds must be between 0 and 1")

                cdf_cuts = np.linspace(tail_bound[0], tail_bound[1], N + 1)
                int_prob = tail_bound[1] - tail_bound[0]

            else:
                cdf_cuts = np.linspace(0, 1, N + 1)
                int_prob = 1.0

            Z = Normal()

            z_cuts = np.empty(2 * tail_N + N + 1)
            if tail_N > 0:
                z_cuts[0:tail_N] = Z.ppf(cdf_cuts[0])
                z_cuts[-tail_N:] = Z.ppf(cdf_cuts[-1])

            z_cuts[tail_N : tail_N + N + 1] = Z.ppf(cdf_cuts)
            z_bins = [(z_cuts[i], z_cuts[i + 1]) for i in range(N + 2 * tail_N)]

            atoms = np.empty([self.M, (N + (2 * tail_N)) ** self.M])

            interiors = np.empty([self.M, (N + 2 * tail_N) ** (self.M)])

            inners = np.zeros(N + 2 * tail_N)

            if tail_N > 0:
                inners[:tail_N] = [(tail_N - i) for i in range(tail_N)]
                inners[-tail_N:] = [(i + 1) for i in range(tail_N)]

            def eval(params, z):
                inds = []
                excl = []

                for j in range(len(z)):
                    if z[j, 0] == z[j, 1]:
                        excl.append(j)
                    elif params[j] != 0.0:
                        inds.append(j)

                dim = len(inds)

                p = np.repeat(params[inds], 2).reshape(dim, 2)

                Z = np.multiply(p, z[inds])

                bounds = ((p**2 - Z) / (np.sqrt(2) * p)).T

                if len(inds) > 0:
                    x_exp = np.prod(
                        -0.5
                        * np.exp(np.square(params[inds]) / 2)
                        * (special.erf(bounds[1]) - special.erf(bounds[0]))
                    )
                else:
                    x_exp = 1

                if len(excl) > 0:
                    x_others = np.prod(np.exp(np.multiply(params[excl], z[excl, 1])))
                else:
                    x_others = 1

                return x_exp * x_others * (N / int_prob) ** dim

            if decomp == "cholesky":
                L = np.linalg.cholesky(self.Sigma)

                for i in range(self.M):
                    mui = self.mu[i]
                    params = L[i, 0 : i + 1]

                    Z_list = [z_bins for _ in range(i + 1)]

                    Z_bins = [np.array(x) for x in list(product(*Z_list))]

                    xi_atoms = []

                    for z_bin in Z_bins:
                        atom = np.exp(mui) * eval(params, z_bin)
                        xi_atoms.append(atom)

                    xi_atoms_arr = np.repeat(
                        xi_atoms, (N + 2 * tail_N) ** (self.M - (i + 1))
                    )

                    inners_i = [inners for _ in range((N + 2 * tail_N) ** i)]

                    interiors[i] = np.repeat(
                        [*inners_i], (N + 2 * tail_N) ** (self.M - (i + 1))
                    )

                    atoms[i] = xi_atoms_arr
            else:
                Λ, Q = np.linalg.eig(self.Sigma)

                A = Q @ np.diag(np.sqrt(Λ))

                if decomp == "sqrt":
                    A = A @ Q.T

                for i in range(self.M):
                    mui = self.mu[i]
                    params = A[i]

                    Z_list = [z_bins for _ in range(self.M)]

                    Z_bins = [np.array(x) for x in list(product(*Z_list))]

                    xi_atoms = []

                    for z_bin in Z_bins:
                        atom = np.exp(mui) * eval(params, z_bin)
                        xi_atoms.append(atom)

                    inners_i = [inners for _ in range((N + 2 * tail_N) ** i)]

                    interiors[i] = np.repeat(
                        [*inners_i], (N + 2 * tail_N) ** (self.M - (i + 1))
                    )

                    atoms[i] = xi_atoms

        max_locs = np.argmax(np.abs(interiors), axis=0)

        max_inds = np.stack([max_locs, np.arange(len(max_locs))], axis=1)

        prob_locs = interiors[max_inds[:, 0], max_inds[:, 1]]

        def prob_assign(x):
            if x == 0:
                return 1 / (N**self.M)
            else:
                return 0.0

        prob_vec = np.vectorize(prob_assign)

        pmv = prob_vec(prob_locs)

        limit = {
            "dist": self,
            "method": "equiprobable",
            "N": N,
            "endpoints": endpoints,
            "tail_bound": tail_bound,
            "decomp": decomp,
        }

        return DiscreteDistribution(
            pmv=pmv,
            atoms=atoms,
            seed=self.random_seed(),
            limit=limit,
        )
