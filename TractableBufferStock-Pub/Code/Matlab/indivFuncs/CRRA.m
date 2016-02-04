% Returns the value of constant relative risk aversion utility for the
% specified consumption level.  This function requires that rho be defined
% elsewhere in the system.

function u = CRRA(scriptc)
globalizeTBSvars;
if rho == 1
    u = log(scriptc);
else
    u = (scriptc.^(1-rho))/(1-rho);
end
