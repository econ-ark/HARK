% This function simulates the path of consumption and assets given a
% starting level of assets and the number of periods to simulate over.

function mcPath = SimGeneratePath(scriptmInitial,PeriodsToGo)
globalizeTBSvars;
scriptcInitial = cE(scriptmInitial);
mcPath = [scriptmEBase scriptcEBase ; scriptmInitial scriptcInitial];
for j = 1:PeriodsToGo
    mcPath = [mcPath ; SimAddAnotherPoint(mcPath)];
end
