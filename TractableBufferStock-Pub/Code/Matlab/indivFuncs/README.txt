This folder contains Matlab functions used in solving the individual's consumption
and saving problem.  A list of files in this directory is below.  Each file contains
additional documentation and help.

- README.txt                  : The text file you are currently reading.
- BackShoot.m                 : Generates a list of Euler points by repeatedly
				iterating backwards to generate the level of con-
				sumption and assets in the previous period.
- cE.m                        : The current consumption function, which calls two
				other functions depending on the inputted value:
                                one for the interpolated function, and one for the
				extrapolation.
- cEExtrap.m                  : The extrapolated portion of the consumption func-
				tion.  Called by cE.m.  The difference between per-
				fect foresight employed consumption and extrapola-
				ted precautionary savings.
- cEInterp.m                  : The interpolated portion of the consumption func-
                                tion.  Called by cE.m.  Uses InterpValue.m with the
				quintic coefficients for the consumption function.
- cEPF.m                      : The employed consumption function under perfect
				foresight.
- cETaylorNearTarget          : The fourth degree Taylor expansion of the consump-
				tion function near the steady state.
- CRRA.m                      : Constant relative risk aversion utility function,
				using coefficient rho.
- CRRAp.m                     : The first derivative of CRRA utility function.
- CRRApp.m                    : The second derivative of CRRA utility function.
- CRRAppp.m                   : The third derivative of CRRA utility function.
- CRRApppp.m                  : The fourth derivative of CRRA utility function.
- CRRAppppp.m                 : The fifth derivative of CRRA utility function.
- cUPF.m                      : The unemployed consumpton function under perfect
				foresight.
- D.m                         : The derivative of the inputed function at the in-
				inputed level.
- EulerPointsStartingFromSSPlus.m  : Generates a list of Euler points starting from
				a level of assets that is [input] above the steady
				state.
- generateInterpMatrix.m      : Using an nx4 matrix as its input, representing
				points in [x f(x) f'(x) f''(x)] format, generates
				a 6x(n-1) matrix of quintic coefficients represen-
				ting the interpolated function for the given points
- Identity.m                  : Simply returns the inputed value.
- InterpValue.m               : Takes three inputs: the value at which to evaluate,
				the set of interpolation points, and the set of
				quintic coefficients (from generateInterpMatrix),
				returning the value of the interpolated function at
				this value.
- kappaEFind.m                : Finds the MPC at steady state using an implicit
				formula derived by Mathematica.
- kappaEPFind.m               : Finds the MMPC at steady state using an implicit
				formula derived by Mathematica.
- kappaEPPFind.m              : Finds the MMMPC at steady state using an implicit
				formula derived by Mathematica.
- kappaEPPPFind.m             : Finds the MMMMPC at steady state using an implicit
				formula derived by Mathematica.
- kappaEPtFromtp1.m           : Finds the MMPC in the previous period based on sev-
				eral inputs.
- kappaEtFromtp1.m            : Finds the MPC in the previous period based on sev-
				eral inputs.
- kappaLim0Find.m             : Finds the limiting MPC as assets approach zero.
- LogscriptCtp1OscriptCt.m    : The log of expected consumption growth if the consum-
				er stays employed (takes level of assets as input).
- naturalEFuncLim0.m          : Unknown purpose, possibly not used?
- naturalFuncLim0.m           : Used by kappaLim0Find to find the limiting MPC.
- PhiGammaSolve.m             : A four input, four output function used to solve for
                                the parameters to the exponential extrapolation of
				the precautionary saving function.
- psavE.m                     : Precautionary saving, which calls two functions de-
				pending on the level of assets inputed: an interpo-
				lation and an extrapolation.
- psaveEExtrap.m              : The extrapolated portion of the precautionary saving
				function; the sum of two exponential functions.
- psaveEInterp.m              : The interpolated portion of the precautionary saving
				function; the difference between consumption under
				perfect foresight vs under uncertainty while employed
- quinticCoeffs.m             : Takes two points in [x f(x) f'(x) f''(x)] form and
				returns the coefficients of the quintic polynomial
				that fits this data.  Called by generateInterpMatrix
- scriptcEDelEqZero.m         : The locus where the change in consumption equals zero
- scriptcEtFromtp1.m          : Finds the level of consumption in the previous period
				based on the current period's consumption and assets.
- scriptCtp1OscriptCt.m       : Expected growth in consumption if the consumer stays
				employed (takes level of assets as input).
- scriptmEDelEqZero.m         : The locus where the change in assets equals zero.
- scriptmEtFromtp1.m          : Finds the level of assets in the previous period based
				on this period's consumption and assets.
- scriptmEtp1Fromt.m          : Finds the level of assets in the next period based on
				this period's assets and consumption.
- SimAddAnotherPoint.m        : Takes an input of an nx2 matrix representing the path
				of assets and consumption over time, and returns the
				next point on this path, based on the consumption
				function and scriptmEtp1Fromt
- SimGeneratePath.m           : Generates a path of assets and consumption over time,
				starting from the inputed level and iterating for the
				inputed number of generations.
- sortLowerArm.m              : Sorts the Euler points of the lower arm so that they
				are in the correct order (rather than reversed).
- vE.m                        : The value function, representing the value to the con-
				sumer at having the inputed level of assets at the be-
				ginning of the period (while employed and uncertain)
- vEInterp.m                  : The interpolated portion of the value function, called
				by vE.
- vEP.m                       : The first derivative of the value function.
- vEPF.m                      : The value function for an employed consumer under per-
				fect foresight.
- vUPF.m                      : The value function for an unemployed consumer under
				perfect foresight.
