% This function takes no inputs and does a binary search for the second derivative of consumption at
% steady state.    The equation used was derived explicitly by Mathematica
% and then translated into Matlab code.  The user is invited to verify the
% solution by hand or with other computer methods.


function kappaEPSeek = kappaEPFind()
globalizeTBSvars;
gothisdeep = 100; % A limit on the number of iterations to search for; not reached in practice.
counter = 0;
searchAt = 0;
found = 0;
scriptcEP = kappaE;
% The binary search for a solution to the equation
while (counter <= gothisdeep && found == 0)
    scriptcEPP = searchAt;
    LHS = scriptcEPP * CRRApp(scriptcE) + scriptcEP^2 * CRRAppp(scriptcE);
    RHS = Beth * (-(1 - mho) * scriptcEP * scriptcEPP * scriptR * CRRApp(scriptcE) + (1 - mho) * (1 - scriptcEP)^2 * scriptcEPP * scriptR^2 * CRRApp(scriptcE) - kappa * mho * scriptcEPP * scriptR * CRRApp(scriptcU) + (1 - mho) * (1 - scriptcEP)^2 * scriptcEP^2 * scriptR^2 * CRRAppp(scriptcE) + kappa^2 * mho * (1 - scriptcEP)^2 * scriptR^2 * CRRAppp(scriptcU));
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
kappaEPSeek = scriptcEPP;
