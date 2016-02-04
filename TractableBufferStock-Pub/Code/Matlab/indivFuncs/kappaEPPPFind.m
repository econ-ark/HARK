% This function takes no inputs and does a binary search for the fourth derivative of consumption at
% steady state.    The equation used was derived explicitly by Mathematica
% and then translated into Matlab code.  The user is invited to verify the
% solution by hand or with other computer methods.


function kappaEPPPSeek = kappaEPPPFind()
globalizeTBSvars;
gothisdeep = 100; % A limit on how many iterations to attempt.  Not reached in practice.
counter = 0;
searchAt = 0;
found = 0;
scriptcEP = kappaE;
scriptcEPP = kappaEP;
scriptcEPPP = kappaEPP;
% The binary search for a solution
while (counter <= gothisdeep && found == 0)
    scriptcEPPPP = searchAt;
    LHS = scriptcEPPPP * CRRApp(scriptcE) + 3 * scriptcEPP^2 * CRRAppp(scriptcE) + 4 * scriptcEP * scriptcEPPP * CRRAppp(scriptcE) + 6 * scriptcEP^2 * scriptcEPP * CRRApppp(scriptcE) + scriptcEP^4 * CRRAppppp(scriptcE);
    RHS = Beth * (-(1 - mho) * scriptcEP*  scriptcEPPPP * scriptR * CRRApp(scriptcE) + 3 * (1 - mho) * scriptcEPP^3 * scriptR^2 * CRRApp(scriptcE) - 4 * (1 - mho) * (1 - scriptcEP) * scriptcEPP * scriptcEPPP * scriptR^2 * CRRApp(scriptcE) - 6 * (1 - mho) * (1 - scriptcEP)^2 * scriptcEPP * scriptcEPPP * scriptR^3 * CRRApp(scriptcE) + (1 - mho) * (1 - scriptcEP)^4 * scriptcEPPPP * scriptR^4 * CRRApp(scriptcE) - kappa * mho * scriptcEPPPP * scriptR * CRRApp(scriptcU) + 3 * (1 - mho) * scriptcEP^2 * scriptcEPP^2 * scriptR^2 * CRRAppp(scriptcE) - 4 * (1 - mho) * (1 - scriptcEP) * scriptcEP^2 * scriptcEPPP * scriptR^2 * CRRAppp(scriptcE) - 18 * (1 - mho) * (1 - scriptcEP)^2 * scriptcEP * scriptcEPP^2 * scriptR^3 * CRRAppp(scriptcE) + 3 * (1 - mho) * (1 - scriptcEP)^4 * scriptcEPP^2 * scriptR^4 * CRRAppp(scriptcE) + 4 * (1 - mho) * (1 - scriptcEP)^4 * scriptcEP * scriptcEPPP * scriptR^4 * CRRAppp(scriptcE) + 3 * kappa^2 * mho * scriptcEPP^2 * scriptR^2 * CRRAppp(scriptcU) - 4 * kappa^2 * mho * (1 - scriptcEP) * scriptcEPPP * scriptR^2 * CRRAppp(scriptcU) - 6 * (1 - mho) * (1 - scriptcEP)^2 * scriptcEP^3 * scriptcEPP * scriptR^3 * CRRApppp(scriptcE) + 6 * (1 - mho) * (1 - scriptcEP)^4 * scriptcEP^2 * scriptcEPP * scriptR^4 * CRRApppp(scriptcE) - 6 * kappa^3 * mho * (1 - scriptcEP)^2 * scriptcEPP * scriptR^3 * CRRApppp(scriptcU) + (1 - mho) * (1 - scriptcEP)^4 * scriptcEP^4 * scriptR^4 * CRRAppppp(scriptcE) + kappa^4 * mho * (1 - scriptcEP)^4 * scriptR^4 * CRRAppppp(scriptcU));
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
kappaEPPPSeek = scriptcEPPPP;
