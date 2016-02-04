% This function is used in the psavE function, and represents the fact that
% precautionary saving is the difference between consumption with perfect
% foresight and consumption under uncertainty.

function x = psavEInterp(scriptm)
globalizeTBSvars;
x = cEPF(scriptm) - cEInterp(scriptm);
