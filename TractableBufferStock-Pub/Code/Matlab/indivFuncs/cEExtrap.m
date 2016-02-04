% The extrapolation of the consumption function above the highest Euler point is defined as
% the difference between perfect foresight consumption (a linear function)
% and the exponential extrapolation of precautionary saving.  It takes a
% single input of the level of assets and returns the consumption function.

function x = cEExtrap(scriptm)
x = cEPF(scriptm) - psavEExtrap(scriptm);
