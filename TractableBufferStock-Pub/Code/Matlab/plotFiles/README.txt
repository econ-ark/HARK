This folder contains Matlab script and functions that create various plots for the
precautionary saving problem, both for the individual and the small open economy.
A list of files in this directory is below.

- README.txt                  : The text file you are currently reading.
- BufferFigPlotA.m            : Produces a plot showing how expected consumption
				growth changes with the level of assets.
- BufferFigPlotB.m            : Produces a plot showing how expected consumption
				growth changes with the level of assets, and how
				this changes with a change in the interest rate.
- ConsFuncPlot.m              : Produces a plot showing the consumption function
				under perfect foresight and under uncertainty.
- cPathAfterThetaDropPlot.m   : Shows the path of consumption over time after a
				drop in the time preference rate.
- mcPathPlot.m                : Shows the path of consumption and assets over time
				after a drop in the time preference rate, using
				the phase diagram.
- MPCPathAfterThetaDropPlot.m : Shows the path of the MPC over time after a drop
				in the time preference rate.
- PhaseDiagPlot.m             : Produces a plot showing the phase diagram for the
				individual's consumption problem, with delC and
				delM equals zero loci.
- plot_arrow.m                : A function that adds an arrow to the current plot;
				found through the Matlab Central File Exchange at
				http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=3345&objectType=FILE
				(slightly modified to allow for larger heads when
				the plot range is oddly scaled)
- plotMyFunc.m                : Returns 201x2 matrix of [x f(x)] values for the
				given function between the inputted min and max.
				Used to plot functions. 

- SOENoStakescPathPlot.m      : Shows the path of aggregate consumption over time
				in a small open economy with no stakes.
- SOEStakescPathAfterThetaDropPlot : Shows the path of aggregate consumption over
				time ins a small open economy with stakes, after
				a drop in the time preference rate.
- TBStargetPlot.m             : Produces a plot showing the target level of assets
				with sustainable consumption.

All of the .m files produce a pdf of the same name, saved to the figures directory
(except plot_arrow.m and plotMyFunc.m).
