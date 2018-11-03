# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.4
# ---

# # A Gentle Introduction to DCEGM
# This notebook introduces the DCEGM method introduced in [1]. The paper generalizes
# the method of Endogenous Grid Points (EGM) in [2] to mixed choice models with
# both Discrete and Continuous (DC) choices. Usually, the models solved by EGM
# have first order conditions (FOCs) that are necessary and sufficient. When we
# introduce the discrete choice, the FOCs can be shown to only be necessecary.
# The generalization consists for a practical method to weed out solutions to
# the FOCs actual solutions to the optimization problem.

# import some libraries to be used below
import matplotlib.pyplot as plt
import numpy
import dcegm

# # The model
# Let us start by motivating the solution method by a problem. The problem is
# one of optimal consumption and retirement,
#
# $$
# \max_{\{c_t,d_t\}^T_{t=1}} ∑^T_{t=1}\beta^t(log(c_t)-\delta (2-d_t))
# $$
#
# where $d_t=1$ means that the agent is retired, and $d_t=2$ means that the agent
# works. Then, $\delta$ can be interpreted as the disutility of work, and we
# see that we have the limiting utility function of a CRRA with $\gamma \downarrow 1$.
#
# We some the problem under a no-borrowing constraint, so $c_t\leq M_t$ where $M_t$
# is the ressources at the beginning of each period that can be consumed or saved
# for next period. The ressource state $M_t$ is given by
#
# $$
# M_t = R(M_{t-1}-c_{t-1}) + y\cdot d_{t-1}
# $$
#
# where $R$ is a fixed interest factor and $y$ is (possibly stochastic) labour income.
# We follow standard timing such that at the very start of period $t$ last period's
# labour income is transferred to the agent, then the agent makes their choices,
# and just as the period is about to end, interest is paid on whatever ressources
# are left.

# To construct an instance of an agent who solves the problem above, we initialize
# a `RetiringDeaton` object:

model = dcegm.RetiringDeaton(saveCommon = True)

# And then we can solve the problem as usual for `AgentType` objects by using
# the `solve` method

model.solve()

# After having `solve`d the model, the `model` object will hold the usual fields
# such as `solution_terminal`, and a list called `solution` that holds each period
# $t$'s solution in `method.solution[T-1-t]`. Let us start from the end, and look
# at the terminal solution

# +
f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

t=19
plt.subplot(1,2,1)
model.plotC(t, 1)
model.plotC(t, 2)

plt.subplot(1,2,2)
model.plotV(t, 1)
model.plotV(t, 2)

# -

# We immidiately notice two things: the two grids over $M_t$ are different as
# the EGM step produces different endogenous grids depending on the discrete
# grids (even if the exogenous grids over the post-decision states are the same),
# and there are regions where it is optimal to retire (for high $M_t$) and work
# (low $M_t$). The intuition should be straight forward. The value function is
# the upper envelope of the two choice specific value functions, so it's clear
# that it will not be differentiable at that point, and that the resulting
# consumption function has a discontinuity at the threshold value of $M_t$.
# The authors call these primary kinks.
#
# Since we chose to set the `saveCommon` keyword to `True`, the `solveOnePeriod`
# method will also save the consumption function and value function on the common
# grid (`model.mGrid`). We can plot this

# +
f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

plt.subplot(1,2,1)
plt.plot(model.mGrid, model.solution[0].C)
plt.xlabel("M")
plt.ylabel("C(M)")
plt.subplot(1,2,2)
plt.plot(model.mGrid, numpy.divide(-1.0, model.solution[0].V_T))
plt.xlabel("M")
plt.ylabel("V(M)")
plt.ylim((-20, -5))
# -


# The "kink" refers to the nondifferentiability in the value function, and we
# see the effect quite clearly in the consumption function, where it translates
# into a discountinuity. Discontinuities and nondifferentiable points are bad for
# any numeric solution method, and this is exactly why we need DCEGM to solve
# this model quickly.

# Let's go back one period.
# It's important to keep in mind that seen from period $t=18$, we have to take
# into consideration that varying consumption today may change the wealth tomorrow
# in such a way that the optimal decision flips from retirement to work (and the
# other way around). Let's plot it.

# +
f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

t=18
plt.subplot(1,2,1)
model.plotC(t, 1)
model.plotC(t, 2)

plt.subplot(1,2,2)
model.plotV(t, 1)
model.plotV(t, 2)
# -

# This time we see a discontinuity already in the choice specific consumption
# function and kinks in the choice specific value functions for the workers!
# This is *not* the discontinuity from the retirement threshold in period $t=18$,
# but from the "future" discontinuity in $t=19$. We'll first look at the final
# consumption function $C_18(M)$, and then we'll return to these to these.

# +
t = 18
f, axs = plt.subplots(1,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

plt.subplot(1,2,1)
plt.plot(model.mGrid, model.solution[model.T-1-t].C)
plt.xlabel("M")
plt.ylabel("C(M)")
plt.subplot(1,2,2)
plt.plot(model.mGrid, numpy.divide(-1.0, model.solution[model.T-1-t].V_T))
plt.xlabel("M")
plt.ylabel("V(M)")
plt.ylim((-40, -8))
# -

# We once again see a primary kink and discontinuity, but we also see the the
# effect of the retirement behavior at period $t=19$. These are called secondary
# kinks. As is maybe clear by now, each period will introduce a new primary kink,
# will propogate back through the recursion and become secondary kinks in earlier
# periods. Let's finish off by looking at $t=1$

# +
f, axs = plt.subplots(2,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

t=1
plt.subplot(1,2,1)
model.plotC(t, 1)
model.plotC(t, 2)

plt.subplot(1,2,2)
model.plotV(t, 1)
model.plotV(t, 2)
# -

# and

# +
t = 1
f, axs = plt.subplots(1,2,figsize=(10,5))
plt.subplots_adjust(wspace=0.6)

plt.subplot(1,2,1)
plt.plot(model.mGrid, model.solution[model.T-1-t].C)
plt.xlabel("M")
plt.ylabel("C(M)")
plt.subplot(1,2,2)
plt.plot(model.mGrid, numpy.divide(-1.0, model.solution[model.T-1-t].V_T))
plt.xlabel("M")
plt.ylabel("V(M)")
#plt.ylim((-120, -50))
# -

# # References
# [1] Iskhakov, F. , Jørgensen, T. H., Rust, J. and Schjerning, B. (2017), The endogenous grid method for discrete‐continuous dynamic choice models with (or without) taste shocks. Quantitative Economics, 8: 317-365. doi:10.3982/QE643
#
# [2] Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. Economics letters, 91(3), 312-320.

# # Experimental: illustrate how it works

import numpy
from dcegm import rise_and_fall


x = numpy.array([0.1, 0.2, 0.3, 0.25, 0.23, 0.35, 0.5, 0.55, 0.49, 0.48,0.47, 0.6, 0.9])
y = numpy.linspace(0, 10, len(x))
rise, fall = rise_and_fall(x, y)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.scatter(x[rise], y[rise], color="green")
plt.scatter(x[fall], y[fall], color="red")



rise

# +
x = numpy.array([0.1, 0.2, 0.3, 0.27, 0.24, 0.3, 0.5, 0.6, 0.5, 0.4, 0.3, 0.5, 0.7])
y = numpy.array([0.1, 0.2, 0.3, 0.25, 0.23, 0.4, 0.5, 0.55, 0.49, 0.48,0.47, 0.6, 0.9])
rise, fall = rise_and_fall(x, y)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.scatter(x[rise], y[rise], color="green")
plt.scatter(x[fall], y[fall], color="red")

# -




