% Documentation needed, unknown purpose.

function x = naturalEFuncLim0(kappaEt)
globalizeTBSvars;
x = Beth * scriptR * mho * (kappa * scriptR * ((1-kappaEt)/kappaEt))^(-rho-1) * kappa;
