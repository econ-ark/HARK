% This function is the fourth degree Taylor expansion of the consumption function
% around the target level of assets.  It takes as an input the distance
% from the target level of assets and returns the Taylor approximation at
% this point.

function x = cETaylorNearTarget(filldelta)
globalizeTBSvars;
x = scriptcE + filldelta * kappaE + (1/2) * (filldelta^2) * kappaEP + (1/6) * (filldelta^3) * kappaEPP + (1/24) * (filldelta^4) * kappaEPPP;
