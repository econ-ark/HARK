% This is the linear function that describes the locus where the change in
% assets between periods would be zero.  Given a level of assets, it
% returns the level of consumption that results in the same level of assets
% next period.

function x = scriptmEDelEqZero(scriptm)
globalizeTBSvars;
x = (biggamma / bigR) + (1-biggamma/bigR)*scriptm;
