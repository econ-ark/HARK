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
# import the Rust module called dark, as well as plotting
# functionality from matplotlib

import dark
import matplotlib.pyplot as plt

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

dmodel = dark.RustAgent(sigma = 0.00, DiscFac = 0.95, method = 'VFI')
dmodel_smooth1 = dark.RustAgent(sigma = 0.15, DiscFac = 0.95, method = 'VFI')
dmodel_smooth2 = dark.RustAgent(sigma = 0.55, DiscFac = 0.95, method = 'VFI')
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
# discrete set of states) is stored in a field named `V`. As a user, you could have
# obtained this information by looking at the docstring to the `RustSolution` object.

p = plt.figure()
for model in models:
    plt.plot(model.states, model.solution[0].V)

# To look closer at the point of non-differentiability, we look at the policies.
# Again, the docstring will inform you that this is stored in the field `P`.

p = plt.figure()
for model in models:
    plt.plot(model.states, model.solution[0].P[0])

# We clearly see, since the plots should align nicely in the browser, that the
# point of non-differentiability comes from the behavior of the policy around
# a threshold. This threshold represents the exact milage where it is no longer
# optimal to do maintenance on the engine. Instead, the superintendent should
# buy or build a new engine, so essentially regenerate the stochastic process.

# ### References
# [1] Rust, John. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." Econometrica: Journal of the Econometric Society (1987): 999-1033.
