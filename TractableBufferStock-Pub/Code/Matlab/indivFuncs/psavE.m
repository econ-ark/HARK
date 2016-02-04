% Precautionary saving is the difference between consumption of the perfect
% foresight consumer and the consumer facing uncertainty.  The function is
% split between the interpolation (between 0 and the highest Euler point)
% and the extrapolation (above the highest Euler point).  It takes a level
% of assets as an input and returns the precautionary saving at this level.

function x = psavE(scriptm)
globalizeTBSvars;
if scriptm < scriptmTop
    x = psavEInterp(scriptm);
else
    x = psavEExtrap(scriptm);
end
