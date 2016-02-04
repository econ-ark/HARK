% This function returns the fourth derivative of constant relative risk
% aversion utility at the specified level of consumption.  The variable rho
% must be defined elsewhere in the system.

function u = CRRApppp(scriptc)
globalizeTBSvars;
u = -rho * (-rho-1) * (-rho-2) * scriptc.^(-rho-3);
