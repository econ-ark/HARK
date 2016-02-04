% This function returns the third derivative of constant relative risk
% aversion utility at the specified level of consumption.  The variable rho
% must be defined elsewhere in the system.

function u = CRRAppppp(scriptc)
globalizeTBSvars;
u = -rho * (-rho-1) * (-rho-2) * (-rho-3) * scriptc.^(-rho-4);
