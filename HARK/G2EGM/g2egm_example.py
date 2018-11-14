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

import time
import mcnab

model = mcnab.MCNAB(T = 20)

t = time.time()
model.solve()
elapsed = time.time() - t
print("Solution took %f seconds" % elapsed)



import matplotlib.pyplot as plt



sol0 = model.solution[18]
plt.scatter(model.grids.M, model.grids.N, c=sol0.Pr_1, s=0.15)

plt.scatter(model.grids.M, model.grids.N, c=sol0.Pr_1, s=0.05)

plt.scatter(sol0.ws.ucon[3], sol0.ws.ucon[4], c=sol0.ws.ucon[2], s=0.3)

plt.scatter(sol0.ws.con[3], sol0.ws.con[4], c=sol0.ws.con[2], s=0.3)

plt.scatter(sol0.ws.dcon[3], sol0.ws.dcon[4], c=sol0.ws.dcon[2], s=0.3)

plt.scatter(sol0.ws.acon[3], sol0.ws.acon[4], c=sol0.ws.acon[2], s=0.3)



sol = model.solution[4]
plt.scatter(model.grids.mMesh, model.grids.nMesh, c=sol.Pr_1, s=0.3)


plt.scatter(model.grids.mMesh, model.grids.nMesh, c=sol.Pr_1, s=0.3)
plt.scatter(model.grids.mMesh, model.grids.nMesh, c=sol.ws.acon[5], s=0.3, alpha=0.1)


