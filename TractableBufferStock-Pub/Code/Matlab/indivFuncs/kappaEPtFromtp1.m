% This function returns the second derivative of consumption in the
% previous period based on several inputs

function x = kappaEPtFromtp1(scriptmEtp1, scriptcEtp1, kappaEtp1, scriptmEt, scriptcEt, kappaEt, kappaEPtp1)
globalizeTBSvars;
scriptcUtp1 = (scriptmEt - scriptcEt) * scriptR * kappa;
y = (Beth * scriptR^2 * (1-kappaEt)^2 * (mho * kappa^2 * CRRAppp(scriptcUtp1) + (1-mho) * kappaEtp1^2 * CRRAppp(scriptcEtp1) + (1-mho) * CRRApp(scriptcEtp1) * kappaEPtp1) - (kappaEt)^2 * CRRAppp(scriptcEt));
z = (CRRApp(scriptcEt) + Beth * scriptR * (mho * kappa * CRRApp(scriptcUtp1) + (1-mho) * kappaEtp1 * CRRApp(scriptcEtp1)));
x = y/z;
