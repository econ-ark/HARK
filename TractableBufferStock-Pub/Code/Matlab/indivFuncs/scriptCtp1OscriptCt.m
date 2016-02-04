% Given a level of consumption, this function returns the approximate ratio
% of consumption this period to expected employed consumption next period.

function x = scriptCtp1OscriptCt(scriptm)
globalizeTBSvars;
x = ((bigR * mybeta)^(1/rho) * (1 + mho * (cE((scriptm - cE(scriptm))*scriptR + 1)/(kappa * (scriptm - cE(scriptm))*scriptR)-1))^(1/rho));
