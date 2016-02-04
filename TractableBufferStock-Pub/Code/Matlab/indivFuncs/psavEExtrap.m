% This function takes a level of assets (above the highest Euler point) and
% returns the extrapolated value of precautionary saving using the
% parameters that were solved for in FindStableArm.

function x = psavEExtrap(scriptm)
globalizeTBSvars;
x = ephi0*exp(phi1*(scriptmTop-scriptm)) - egamma0*exp(gamma1*(scriptmTop-scriptm));
