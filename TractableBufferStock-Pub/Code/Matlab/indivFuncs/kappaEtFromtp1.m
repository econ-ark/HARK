% % This function returns the first derivative of consumption in the
% previous period based on several inputs

function x = kappaEtFromtp1(scriptmEtp1, scriptcEtp1, kappaEtp1, scriptmEt, scriptcEt)
globalizeTBSvars;
scriptcUtp1 = kappa * (scriptmEt - scriptcEt) * scriptR;
natural = Beth * scriptR * (1 / CRRApp(scriptcEt)) * ((1-mho) * CRRApp(scriptcEtp1) * kappaEtp1 + mho * CRRApp(scriptcUtp1) *kappa);
x = natural / (natural + 1);
