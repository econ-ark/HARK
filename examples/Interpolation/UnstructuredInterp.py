# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from HARK.interpolation import UnstructuredInterp
import matplotlib.pyplot as plt
import numpy as np


# %%
def squared_coords(x, y):
    return x**2 + y**2


# %%
x_grid = np.geomspace(1, 11, 11) - 1
x_mat, y_mat = np.meshgrid(x_grid, x_grid, indexing="ij")
z_mat = squared_coords(x_mat, y_mat)
z_mat[5, 5] = np.nan

# %%
interp = UnstructuredInterp(z_mat, [x_mat, y_mat], method="cubic")

# %%
interp(x_mat[5, 5], y_mat[5, 5])

# %%
x_new, y_new = np.meshgrid(
    np.linspace(0, 10, 11),
    np.linspace(0, 10, 11),
    indexing="ij",
)

# %%
z_unstruc_interp = interp(x_new, y_new)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x_new, y_new, z_unstruc_interp)
plt.show()

# %%
x_rand = np.random.rand(100) * 11
y_rand = np.random.rand(100) * 11
z_rand = squared_coords(x_rand, y_rand)

# %%
rand_interp = UnstructuredInterp(z_rand, [x_rand, y_rand], method="cubic")

# %%
z_rand_interp = rand_interp(x_new, y_new)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x_new, y_new, z_rand_interp)
plt.show()

# %%
