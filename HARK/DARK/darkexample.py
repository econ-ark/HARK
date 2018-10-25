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

import dark
import matplotlib.pyplot as plt

dmodel = dark.RustAgent(sigma = 0.00, DiscFac = 0.95,c = -0.005, method = 'VFI', tolerance = 0.000001, Nm=400)
dmodel2 = dark.RustAgent(sigma = 0.05, DiscFac = 0.95,c = -0.005, method = 'VFI', tolerance = 0.000001, Nm=400)

dmodel.solve()
dmodel2.solve()

dmodel2.solution_terminal.V

p = plt.figure()
plt.plot(dmodel.states, dmodel.solution[0].V)
plt.plot(dmodel2.states, dmodel2.solution[0].V)

p = plt.figure()
plt.plot(dmodel.states, dmodel.solution[0].P[0])
plt.plot(dmodel2.states, dmodel2.solution[0].P[0])


