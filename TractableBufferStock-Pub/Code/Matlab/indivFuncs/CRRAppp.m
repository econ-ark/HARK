% This function returns the third derivative of constant relative risk
% aversion utility at the specified level of consumption.  The variable rho
% must be defined elsewhere in the system.

function u = CRRAppp(scriptc)
globalizeTBSvars;
u = -rho * (-rho-1) * scriptc.^(-rho-2);
