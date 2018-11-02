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

import matplotlib.pyplot as plt
import numpy
import dcegm

model = dcegm.RetiringDeaton()

model.updateLast()

tersol = model.solution_terminal

model.solution_terminal

# +
#plt.plot(tersol.M, tersol.C)

# +
#plt.plot(tersol.M, tersol.V_T)

# +
#plt.plot(tersol.M, -1.0/tersol.V_T)
# -

model.solve()



# +
t=1
plt.plot(model.solution[t].rs.M, numpy.divide(-1.0,model.solution[t].rs.V_T), label = "retired")

plt.plot(model.solution[t].ws.M, numpy.divide(-1.0,model.solution[t].ws.V_T), label = "working")
plt.legend()
plt.ylim((-30, 0))
plt.title("Choice specific value functions at t=%i" % (model.T-t-1))
# -

t=0
plt.plot(model.solution[t].ws.M, model.solution[t].ws.C, label = "working (t=19)")
t=1
plt.plot(model.solution[t].ws.M, model.solution[t].ws.C, label = "working (t=18)")
t=4
plt.plot(model.solution[t].ws.M, model.solution[t].ws.C, label = "working (t=15)")
plt.legend()
plt.title("Consumption functions for workers")
plt.xlim((0,200))
plt.ylim((0,150))
plt.xlabel("m")
plt.ylabel("c(m)")

# +
t=18
plt.plot(model.solution[t].rs.M, model.solution[t].rs.C, label = "retired")

plt.plot(model.solution[t].ws.M, model.solution[t].ws.C, label = "working")
plt.legend()
plt.title("Consumption functions at period t = %i" % (model.T-t+1))
plt.xlim((0,500))
plt.xlabel("m")
plt.ylabel("c(m)")
# -

plt.plot(model.solution[t].ws.M, numpy.divide(-1.0, model.solution[t].ws.V_T))

# +
t=2
plt.plot(model.solution[t].rs.M, model.solution[t].rs.C, label = "retired")

plt.plot(model.solution[t].ws.M, model.solution[t].ws.C, label = "working")
plt.legend()
# -

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

