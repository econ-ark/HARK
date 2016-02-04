% This is the interpolated part of the value function, between scriptmBot
% and scriptmTop.

function scriptv = vEInterp(scriptm)
globalizeTBSvars;
scriptv = InterpValue(scriptm, vEPoints, vECoeffs);
