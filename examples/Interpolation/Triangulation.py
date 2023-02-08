# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.9.13 ('hark-gpu')
#     language: python
#     name: python3
# ---

# %%
import matplotlib.tri as tri
import matplotlib.pyplot as plt

import numpy as np


# %%
xlen = 10
ylen = 16
xPoints = np.arange(0, xlen + 1, 1)
yPoints = np.arange(0, ylen + 1, 1)

gridPoints = np.array([[[x, y] for y in yPoints] for x in xPoints])
a = [
    [i + j * (ylen + 1), (i + 1) + j * (ylen + 1), i + (j + 1) * (ylen + 1)]
    for i in range(ylen)
    for j in range(xlen)
]

triang_a = tri.Triangulation(
    gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten(), a
)


plt.triplot(triang_a, "go-")
plt.plot(gridPoints[:, :, 0], gridPoints[:, :, 1], "bo")
plt.title("Triangulation Visualization")


# %%
xlen = 10
ylen = 16
xPoints = np.arange(0, xlen + 1, 1)
yPoints = np.arange(0, ylen + 1, 1)

gridPoints = np.array([[[x, y] for y in yPoints] for x in xPoints])

b = [
    [(i + 1) + (j + 1) * (ylen + 1), (i + 1) + j * (ylen + 1), i + (j + 1) * (ylen + 1)]
    for i in range(ylen)
    for j in range(xlen)
]


triang_b = tri.Triangulation(
    gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten(), b
)

plt.triplot(triang_b, "ro-")
plt.plot(gridPoints[:, :, 0], gridPoints[:, :, 1], "bo")
plt.title("Triangulation Visualization")


# %%
plt.triplot(triang_a, "go-")
plt.triplot(triang_b, "ro-")

# %%
