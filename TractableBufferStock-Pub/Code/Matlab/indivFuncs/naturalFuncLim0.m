% This function is used in order to find the limiting MPC as assets
% approach zero.  See kappaLim0Find.

function x = naturalFuncLim0(kEt)
globalizeTBSvars;
x = Beth * scriptR * mho * (kappa * scriptR * ((1-kEt)/kEt))^(-rho-1) * kappa;
