% This function takes no inputs and returns the limiting employed & uncertain
% MPC as assets approach zero by performing a binary search.

function kappaESeek = kappaLim0Find()
globalizeTBSvars;
gothisdeep = 100;
counter = 0;
searchAt = 0.5;
found = 0;
while (counter <= gothisdeep && found == 0)
    if searchAt < naturalFuncLim0(searchAt)/(1 + naturalFuncLim0(searchAt))
        searchAt = searchAt - 2^(-counter)*(lambda-0.5);
    end
    if searchAt > naturalFuncLim0(searchAt)/(1 + naturalFuncLim0(searchAt))
        searchAt = searchAt + 2^(-counter)*(lambda-0.5);
    end
    if searchAt == naturalFuncLim0(searchAt)/(1 + naturalFuncLim0(searchAt))
        found = 1;
    end
    counter = counter + 1;
end
kappaESeek = searchAt;
