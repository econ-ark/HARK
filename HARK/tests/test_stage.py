"""
This file implements unit tests for abstract Bellman stage code.
"""

from typing import Any, Mapping
import HARK.distribution as distribution
from HARK.utilities import CRRAutility
from HARK.stage import Stage
import unittest

CRRA = 5

R = 1.01
G = 1.02

sigma_psi = 1.05
sigma_theta = 1.15
sigma_eta = 1.1
p_live = 0.98

class testPortfolioConsumptionStages(unittest.TestCase):
    def setUp(self):

        self.consumption_stage = Stage(
            transition = lambda x, k, a : {'a' : x['m'] - a['c']}, 
            reward = lambda x, k, a : CRRAutility(a['c'], CRRA), 
            inputs = ['m'], 
            actions = ['c'],
            outputs = ['a'],
            constraints = [lambda x, k, a: x['m'] - a['c']], # has to be nonnegative to clear
            discount = .96 # lambda x, k, a : .96 * k['psi']^(1 - CRRA) < --- 
        )

        self.allocation_stage = Stage(
            transition = lambda x, k, a : {'a' : x['a'], 'alpha' : a['alpha']}, 
            inputs = ['a'], 
            actions = ['alpha'],
            outputs = ['a', 'alpha'],
            constraints = [
                lambda x, k, a: 1 - a['alpha'], 
                lambda x, k, a: a['alpha']
            ]
        )

        def growth_transition(x, k, a): 
            return {'m' : ((x['alpha'] * k['eta'] + (1 - x['alpha']) * R) 
                           * x['a'] + k['theta']) 
                    / (k['psi'] * G)}

        self.growth_stage = Stage(
            transition = growth_transition,
            inputs = ['a', 'alpha'],
            discount = lambda x, k, a: p_live * k['psi'] ** (1 - CRRA), 
            shocks = {
                'psi' : distribution.Lognormal(0, sigma_psi),
                'theta' : distribution.Lognormal(0, sigma_theta),
                'eta' : distribution.Lognormal(0, sigma_eta),
                # 'live' : distribution.Bernoulli(p_live) ## Not implemented for now
            },
            outputs = ['m'],
        )

    def test_consumption_stage(self):

        def consumption_v_y(y : Mapping[str,Any]):
            return CRRAutility(y['a'], CRRA)

        pi_star, q = self.consumption_stage.optimal_policy(
            {'m' : [9, 11, 20, 300, 4000, 5500]},
            v_y = consumption_v_y
            )

        # q function has proper coords
        assert q.coords['m'][4].data.tolist() == 4000

        # Consumer over half the resources
        assert (pi_star[5] > 2750).data.tolist()

        assert self.consumption_stage.T({'m' : 100}, {}, {'c' : 50})['a'] == 50

        assert self.consumption_stage.T({'m' : 100}, {}, {'c' : 101})['a'] == -1

        assert self.consumption_stage.reward({'m' : 100}, {}, {'c' : 50}) < 0.00001

        assert self.consumption_stage.q({'m' : 100}, {}, {'c' : 50}, v_y = consumption_v_y) < 0.000001

        sol = self.consumption_stage.solve(
            {'m' : [0, 50, 100, 1000]},
            {},
            consumption_v_y
            )

        ## tests for the consumption soluition...
        """
        <xarray.Dataset>
        Dimensions:  (m: 4)
        Coordinates:
        * m        (m) int64 0 50 100 1000
        Data variables:
            v_x      (m) float64 -1.291e+17 -1.254e-06 -7.839e-08 -7.839e-12
            pi*      (m) float64 4.414e-05 25.1 50.2 502.0
            q        (m) float64 -1.291e+17 -1.254e-06 -7.839e-08 -7.839e-12
        """


    def test_allocation_stage(self):

        def allocation_v_y(y : Mapping[str,Any]):
            return CRRAutility(y['alpha'] * y['a'] + 1,CRRA) \
                + CRRAutility((1 - y['alpha']) * y['a'] + 1, CRRA * 0.9) 

        assert self.allocation_stage.T({'a': 100}, {}, {'alpha' : 0.5})['a'] == 100

        assert self.allocation_stage.reward({'a': 100}, {}, {'alpha' : 0.5}) == 0

        # smoke tests
        pi_star, q = self.allocation_stage.optimal_policy(
            {'a' : [9, 11, 20, 300, 4000, 5500]},
            v_y = allocation_v_y
            )

        sol = self.allocation_stage.solve(
            {'a' : [0, 50, 100, 1000]},
            {},
            allocation_v_y
            )

    def test_growth_stage(self):

        assert self.growth_stage.T(
            {'a': 100, 'alpha' : 0.5},
            {'psi' : 1.00, 'theta' : 1.10, 'eta' : 1.05, 'live' : 1},
            {}
        )['m'] == 102.05882352941175

        def growth_v_y(y : Mapping[str,Any]):
            return CRRAutility(y['m'], CRRA)

        pi_star, q = self.growth_stage.optimal_policy(
            {'a' : [300, 600],
            'alpha' : [0, 1.0]
            },
            {'psi' : [1., 1.1], 
            'theta' : [1., 1.1], 
            'eta' : [1., 1.1],
            # 'live' : [0, 1] 
            }, 
            v_y = growth_v_y)

        q

        sol = self.growth_stage.solve(
            {'a' : [0, 500, 1000], 'alpha' : [0, 0.5, 1.0]},
            {
                'psi' : 4, 
                'theta' : 4, 
                'eta' : 4,
            # 'live' : [0, 1] 
            }, growth_v_y)

        