This folder contains Matlab scripts used to solve the individual's consumption
and saving problem.  Below is a list of files in this directory.

- README.txt                  : The text file you are currently reading.
- CheckForBigRImpatience.m    : Checks whether the consumer is return impatient,
				i.e. whether scriptPR < 1.  Called whenever the
				parameter values are changed.
- CheckForGammaImpatience.m   : Checks whether the consumer is growth impatient,
				i.e. whether scriptPG < 1.  Called whenever the
				parameter values are changed.
- FindStableArm.m             : Finds a set of Euler points on the stable arm
				of the phase diagram, and interpolates them to
				construct the consumption function.  Also makes
				the extrapolated portion of the consumption func-
				tion, as well as the value function.
- globalizeTBSvars.m          : Declares a large number of variables to be global
				rather than local so that functions can use them
                                without having to pass them explicitly.  Called
				by a large number of functions.
- initializeParams.m          : Sets the base values for the parameters.
- resetParams.m               : Resets the parameters to their base values.
- setValues.m                 : Uses the current parameter values to generate
				many other useful variable values.  This should
				be called every time parameter values are changed
- ShowParams.m                : Displays various parameters and other values.
