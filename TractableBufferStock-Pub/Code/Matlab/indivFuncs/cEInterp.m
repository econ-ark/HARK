% This is the interpolated part of the consumption function, between 0 and
% the highest Euler point.  It takes a single input of the level of assets
% and returns the level of consumption.

function scriptc = cEInterp(scriptm)
globalizeTBSvars;
scriptc = InterpValue(scriptm, EulerPoints, consumptionCoeffs);
