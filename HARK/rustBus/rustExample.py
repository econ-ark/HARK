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
#     version: 3.6.5
# ---

# ### Setup
# Import the Rust module called dark, as well as plotting
# functionality from matplotlib

import matplotlib.pyplot as plt
import numpy # for argmax
import rust

# ### Specify models
# The `AgentType` called `RustAgent` can be used to setup and solve the
# capital replacement model in [1]. It is possible to specify a model
# quite close to the preferred specification in the original paper, or
# change some parameters to look at how sensitive the model is to various
# specification changes.
#
# A collect of recent papers look at how smoothing can affect the ease of
# solving discrete choice or even mixed discrete and continuous choice models.
# They smooth using an extreme value type I taste shock. This gives rise to the
# the formulas from [1], but with a correction for the scaling parameter that
# determines the amount of smoothing.
#
# Let us try to solve a model with no taste shocks, one with a relatively small
# taste shock, and one with a larger taste shock.
#
# First, we construct three different `RustAgent` instances, and collect them
# in a `tuple` to simplify repeated operations on the objects.

dmodel = rust.RustAgent(sigma = 0.00, DiscFac = 0.95, method = 'VFI')
dmodel_smooth1 = rust.RustAgent(sigma = 0.15, DiscFac = 0.95, method = 'VFI')
dmodel_smooth2 = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'VFI')
models = (dmodel, dmodel_smooth1, dmodel_smooth2)

# We can then iterate over this tuple, and call the `solve()` method that is
# the canonical way to solve models represented by `AgentType`s in HARK.

for model in models:
    model.solve()

# Below, we observe similarly looking value functions (ex ante the shock realizations,
# also called the integrated value function), although the smoothing of the
# non-differentiable point is abvious. The level is also different between specifications.
# Notice, that we extract the model solution from element `0` as this is an infinite
# horizon version of the model, and that the value function (represented on the)
# discrete set of milage) is stored in a field named `V`. As a user, you could have
# obtained this information by looking at the docstring to the `RustSolution` object.

p = plt.figure()
for model in models:
    plt.plot(model.milage, model.solution[0].V, label = "σ = %f" % model.sigma)
plt.legend();

# To look closer at the point of non-differentiability, we look at the policies.
# Again, the docstring will inform you that this is stored in the field `P`.

p = plt.figure()
for model in models:
    plt.plot(model.milage, model.solution[0].P[0], label = "σ = %f" % model.sigma)
plt.legend();

# We clearly see, since the plots should align nicely in the browser, that the
# point of non-differentiability comes from the behavior of the policy around
# a threshold. This threshold represents the exact milage where it is no longer
# optimal to do maintenance on the engine. Instead, the superintendent should
# buy or build a new engine, so essentially regenerate the stochastic process.
# The *conditional choice probabilities* in the smoothed model has the same
# overall look.

# ### Maintenance cost
#
# Of course we can do more than change the scale parameter of the taste shock.
# We could try to analyze the effect of the maintenance cost parameter `c`.

model_lowCost_zeroScale = rust.RustAgent(sigma = 0.0, c = -0.0025, method = 'VFI')
model_highCost_zeroScale = rust.RustAgent(sigma = 0.0, c = -0.0050, method = 'VFI')
costModels = (model_lowCost_zeroScale, model_highCost_zeroScale)

# we solve the model instances

for model in costModels:
    model.solve()

# and then we plot as before

model.c

fig = plt.figure()
for model in costModels:
    plt.plot(model.milage, model.solution[0].V, label = "c = %f" % model.c)
plt.legend();

p = plt.figure()
for model in costModels:
    plt.plot(model.milage, model.solution[0].P[0], label = "c = %f" % model.c)
plt.legend();

# we see that the threshold milage is lower with the higher cost. The obvious
# question is: what is this value of milage and what's the maintenance cost
# right at the threshold?

i_lowCost = numpy.argmax(model_lowCost_zeroScale.solution[0].P < 1)
i_highCost = numpy.argmax(model_highCost_zeroScale.solution[0].P < 1)
threshLow = model_lowCost_zeroScale.milage[i_lowCost-1]
threshHigh = model_highCost_zeroScale.milage[i_highCost-1]
print("The threshold milage with low cost is %f" % threshLow)
print("The threshold milage with high cost is %f" % threshHigh)
threshCostLow = model_lowCost_zeroScale.c*threshLow
threshCostHigh = model_highCost_zeroScale.c*threshHigh
print("The maintenance cost, at threshold, with low cost is %f" % threshCostLow)
print("The maintenance cost, at threshold, with high cost is %f" % threshCostHigh)

# +
# ### Newton's method

dmodel_vfi = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'VFI')
dmodel_newton = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'Newton')

fig = plt.figure()
for model in (dmodel_vfi, dmodel_newton):
    model.solve()
    plt.plot(model.milage, model.solution[0].V, label = "method = %s" % model.method)
plt.legend();
# -

# Notice, that we can only used Newton's method with positive sigma. Let's look
# at the performance using a naïve benchmark

bmodel_vfi = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'VFI')
bmodel_newton = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'Newton')

bmodel_newton.solve()

print('Benchmarking VFI...')
# %timeit bmodel_vfi.solve()
print('Benchmarking Newton...')
# %timeit bmodel_newton.solve()

# We see that Newton's method is significantly faster here. We can of course make Newton's method lose this battle by increasing the dimensinality of the problem dramatically to make solving a linear system dominate the full run time.

bmodel_vfi_big = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'VFI', Nm = 2000)
bmodel_newton_big = rust.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'Newton', Nm = 2000)

print('Benchmarking VFI...')
# %timeit bmodel_vfi_big.solve()
print('Benchmarking Newton...')
# %timeit bmodel_newton_big.solve()

# ### References
# [1] Rust, John. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." Econometrica: Journal of the Econometric Society (1987): 999-1033.
