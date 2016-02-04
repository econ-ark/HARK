% This function returns the second derivative of constant relative risk
% aversion utility at the specified level of consumption.  The variable rho
% must be defined elsewhere in the system.

function u = CRRApp(scriptc)
globalizeTBSvars;
u = -rho * scriptc.^(-rho-1);
