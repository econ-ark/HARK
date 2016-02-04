% This is the value function, which is in three pieces: between scriptmBot
% and scriptmTop, below scriptmBot, and above scriptmTop.  Outside the
% bounds, the function performs a numeric integration.

function v = vE(scriptm)
globalizeTBSvars;
if scriptm < scriptmBot
    areasum = 0;
    current = scriptm;
    iterator = (scriptmBot - scriptm)/300;
    while current < scriptmBot
        areasum = areasum + CRRAp(cEInterp(current + iterator/2))*iterator;
        current = current + iterator;
    end
    v = scriptvBot - areasum;
elseif scriptm > scriptmTop
    areasum = 0;
    current = scriptmTop;
    iterator = (scriptm - scriptmTop)/300;
    while current < scriptm
        areasum = areasum + CRRAp(cEExtrap(current + iterator/2))*iterator;
        current = current + iterator;
    end
    v = scriptvTop + areasum;
else
    v = vEInterp(scriptm);
end
