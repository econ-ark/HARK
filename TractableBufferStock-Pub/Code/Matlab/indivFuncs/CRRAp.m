% This function returns the first derivative of constant relative risk
% aversion utility at the specified level of consumption.  The variable rho
% must be defined elsewhere in the system.

function u = CRRAp(scriptc)
globalizeTBSvars;
u = scriptc.^(-rho);
