# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interpolation using Numba `jitclass`
#
# ### The `LinearInterpFast` class

# %%
import numpy as np

from HARK.interpolation import LinearInterp
from HARK.numba import LinearInterpFast

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
x = np.array([0.0, 1.0])
y = np.array([0.0, 2.0])
new_x = np.linspace(0, 1, 100)

# %% [markdown]
# ### Instantiation takes time, the first time
#
# Compare time to instantiation for standard linear interpolator vs. numba implementation.

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time interp = LinearInterp(x,y)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time interpFast = LinearInterpFast(x,y)

# %% [markdown]
# As seen above, instantiating the numba implementation takes longer, but only the first time. See second instantiation below.

# %%
# %time interpFast = LinearInterpFast(x,y)

# %% [markdown]
# ### Calling also takes time, the first time
#
# Compare time for first call below, versus repeated calls thereafter. 

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time interp(new_x)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %time interpFast.eval(new_x)

# %% [markdown]
# After the first call, however, we can see that repeatedly calling the numba implementation is significantly faster. 

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %timeit interp(new_x)

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# %timeit interpFast.eval(new_x)

# %% [markdown]
#    ### Limiting Decay Extrapolation

# %%
interp = LinearInterp(x, y, intercept_limit=0, slope_limit=1, lower_extrap=True)
interpFast = LinearInterpFast(x, y, intercept_limit=0, slope_limit=1, lower_extrap=True)
out_x = np.linspace(-1, 2, 100)
np.max(np.abs(interp(out_x) - interpFast.eval(out_x)))

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### The `CubicInterpFast` class

# %% pycharm={"name": "#%%\n"}
from HARK.interpolation import CubicInterp

from HARK.numba import CubicInterpFast

import matplotlib.pyplot as plt

# %% pycharm={"name": "#%%\n"}
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-(x ** 2) / 9.0)
dydx = 2.0 * x / 9.0 * np.sin(-(x ** 2) / 9.0)

cubic_interp = CubicInterp(x, y, dydx, lower_extrap=True)
cubic_fast = CubicInterpFast(x, y, dydx, lower_extrap=True)

# %% pycharm={"name": "#%%\n"}
xnew = np.linspace(0, 10, num=41, endpoint=True)
xout = np.linspace(-1, 11, num=41, endpoint=True)

plt.plot(x, y, "o", xout, cubic_interp(xout), "-", xout, cubic_fast.eval(xout))
plt.legend(["data", "hark", "fast"], loc="best")
plt.show()

# %% pycharm={"name": "#%%\n"}
np.max(np.abs(cubic_interp(xnew) - cubic_fast.eval(xnew)))

# %% pycharm={"name": "#%%\n"}
cubic_fast.eval(np.array([0.5]))

# %% pycharm={"name": "#%%\n"}
# %timeit cubic_interp(xnew)
# %timeit cubic_interp(xout)

# %%
# %timeit cubic_fast.eval(xnew)
# %timeit cubic_fast.eval(xout)

# %%
