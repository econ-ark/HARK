% This is the consumption function for an unemployed consumer, which takes
% as an input the level of assets and returns the consumption of the
% unemployed consumer.  Note that an unemployed consumer has perfect
% foresight, as he knows that he will never be employed again.

function x = cUPF(scriptm)
globalizeTBSvars;
x = kappa * scriptm;
