% This function takes no inputs and does a binary search for the MPC at
% steady state.  The equation used was derived explicitly by Mathematica
% and then translated into Matlab code.  The user is invited to verify the
% solution by hand or with other computer methods.

function kappaESeek = kappaEFind()
globalizeTBSvars;
gothisdeep = 100; % A limit on how many iterations to attempt.  This is not reached in practice, as we hit the machine precision limit
counter = 0;
searchAt = 0;
found = 0;
% The binary search for a solution.
while (counter <= gothisdeep && found == 0)
    scriptcEP = searchAt;
    LHS = scriptcEP * CRRApp(scriptcE);
    RHS = Beth * ((1 - mho) * (1 - scriptcEP)* scriptcEP * scriptR * CRRApp(scriptcE) + kappa * mho * (1 - scriptcEP)* scriptR * CRRApp(scriptcU));
    if LHS > RHS
        searchAt = searchAt + 2^(-counter);
    end
    if LHS < RHS
        searchAt = searchAt - 2^(-counter);
    end
    if LHS == RHS
        found = 1;
    end
    counter = counter + 1;
end
kappaESeek = scriptcEP;
