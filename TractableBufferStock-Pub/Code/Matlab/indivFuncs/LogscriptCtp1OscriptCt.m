% This function takes as an input a level of assets and returns the
% log of expected consumption growth from this period to next if the
% consumer stays employed.

function x = LogscriptCtp1OscriptCt(scriptm)
globalizeTBSvars;
x = log(((bigR * mybeta)^(1/rho) * (1 + mho * (cE((scriptm - cE(scriptm))*scriptR + 1)/(kappa * (scriptm - cE(scriptm))*scriptR)-1))^(1/rho)));
