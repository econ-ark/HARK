"""
This HA model combines SIM, a Taylor rule for monetary policy
with a gradual tax adjustment rule and real bonds on the fiscal side
"""

import numpy as np
import sequence_jacobian as sj

"""Simple household block"""

hh = sj.hetblocks.hh_sim.hh
make_grids = sj.hetblocks.hh_sim.make_grids

def income(Y, T, e_grid):
    # post-tax labor income
    y = (Y-T) * e_grid
    return y

household_simple = hh.add_hetinputs([make_grids, income])


"""Inflation, monetary, and fiscal"""

@sj.simple
def nkpc(pi, Y, X, C, kappa_w, vphi, frisch, markup_ss, eis, beta):
    piw = pi + X - X(-1)
    # note: for simplicity, we ignore distortionary effect of taxation here
    piwres = kappa_w * (vphi*(Y/X)**(1/frisch) - 1/markup_ss * X * C**-(1/eis)) + beta * piw(1) - piw
    return piwres, piw


@sj.simple
def monetary_taylor(pi, ishock, rss, phi_pi):
    i = rss + phi_pi * pi + ishock
    r_ante = i - pi(1)
    return i, r_ante


@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r


@sj.solved(unknowns={'B': (-1., 1.)}, targets=['Bres'], solver="brentq")
def fiscal_deficit_Trule(r, G, B, Tss, phi_T, Y):
    T = Tss + phi_T * (B(-1) - B.ss)
    Bres = (1 + r) * B(-1) + G - T - B
    return T, Bres


"""Overall model"""

@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = C + G - Y
    return asset_mkt, goods_mkt


ha = sj.create_model([household_simple, nkpc, monetary_taylor, ex_post_rate, fiscal_deficit_Trule, mkt_clearing],
                     name="Simple HA Model")
