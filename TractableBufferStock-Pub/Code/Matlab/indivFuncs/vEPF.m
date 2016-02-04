% This is the value function for an employed consumer under perfect
% foresight.  Given a level of assets, it returns the consumer's value of
% holding those assets.

function x = vEPF(scriptm)
globalizeTBSvars
temp = CRRA(scriptcEFuncInf(scriptm)) * littleV;
if rho==1
    temp = temp + log(bigR * mybeta) * (mybeta/((mybeta-1)^2));
end
x = temp;
