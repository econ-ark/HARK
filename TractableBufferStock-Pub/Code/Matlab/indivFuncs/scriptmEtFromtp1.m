% This function returns the previous period's monetary assets based on this
% period's assets and consumption.

function x = scriptmEtFromtp1(scriptmEtp1, scriptcEtp1)
globalizeTBSvars;
x = (biggamma / bigR)*(scriptmEtp1 - 1) + scriptcEtFromtp1(scriptmEtp1, scriptcEtp1);
