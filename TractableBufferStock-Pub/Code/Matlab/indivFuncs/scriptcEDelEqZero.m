% This function is the linear equation describing the locus where the the
% change in consumption between periods would be zero.  It takes a level of
% assets and returns the "sustainable consumption" level. 

function x = scriptcEDelEqZero(scriptm)
globalizeTBSvars;
x = scriptm * ((scriptR * kappa * Pi)/(1 + scriptR * kappa * Pi));
