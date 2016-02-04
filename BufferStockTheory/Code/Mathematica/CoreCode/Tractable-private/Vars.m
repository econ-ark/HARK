(* ::Package:: *)

If[NameQ["ParamsAreSet"]==True,If[ParamsAreSet==True,Print["This file should be executed before parameter values are defined."];Abort[]]]


(* Variables *)

(* \[Lambda] is the MPS for perfect foresight problem (like unemployed consumers); *)
(* \[Kappa] is the MPC for perfect foresight problem (like unemployed consumers)  *)
\[Lambda]=((R)^-1) ((R) \[Beta])^(1/\[Rho]);
\[Kappa]=1-\[Lambda];

(* \[CapitalGamma] is the growth factor conditional on remaining employed *)
\[CapitalGamma]=\[GothicCapitalG]/(1-\[Mho]);

(* \[ScriptCapitalR] is the return factor for the normalized problem *)
(* (R) is the return factor in levels (in parens to allow search-and-replace) *)
\[ScriptCapitalR]=(R)/\[CapitalGamma];

(* \[ScriptCapitalP] is the growth patience factor *)
\[ScriptCapitalP]Growth=((R) \[Beta])^(1/\[Rho])/\[CapitalGamma];
\[ScriptCapitalP]Return=((R) \[Beta])^(1/\[Rho])/(R);
\[WeierstrassP]\[Gamma]=Log[\[ScriptCapitalP]Growth];
\[WeierstrassP]rtntn=Log[\[ScriptCapitalP]Return];
\[CurlyPi]=(1+(\[ScriptCapitalP]Growth^-\[Rho]-1)/\[Mho])^(1/\[Rho]);
\[GothicH]=(1/(1-\[GothicCapitalG]/(R)));  (* Human wealth, as a ratio to permanent labor income *)
\[Zeta]=\[ScriptCapitalR] \[Kappa] \[CurlyPi]; (* A useful constant *)
\[Beta] = 1/(1+\[CurlyTheta]); (* Time preference factor *)
(R)= 1+(r);
\[GothicCapitalG]= 1+\[GothicG];

(* Target values of \[ScriptM]E,\[ScriptC]E,\[ScriptA]E,\[ScriptB]E,\[ScriptB]U,\[ScriptC]U *)
\[ScriptM]E=1+((R)/(\[CapitalGamma]+\[Zeta] \[CapitalGamma]-(R)));
\[ScriptC]E=(1-\[ScriptCapitalR]^-1)\[ScriptM]E+\[ScriptCapitalR]^-1;
\[ScriptA]E=\[ScriptM]E-\[ScriptC]E;
\[ScriptB]E=\[ScriptA]E \[ScriptCapitalR];
\[ScriptY]E=\[ScriptA]E (\[ScriptCapitalR]-1) + 1;
\[ScriptX]E=Chop[\[ScriptY]E-\[ScriptC]E];
\[ScriptB]U=\[ScriptB]E;
\[ScriptM]U=(\[ScriptM]E-\[ScriptC]E)\[ScriptCapitalR];
\[ScriptC]U=\[ScriptM]U \[Kappa];
\[GothicV] =1/(1-\[Beta] (((R) \[Beta])^((1/\[Rho])-1)));
\[ScriptV]U=u[\[ScriptC]U] \[GothicV];
\[ScriptV]E=(u[\[ScriptC]E] + \[Beta] (\[CapitalGamma]^(1-\[Rho])) \[Mho] vUPF[\[ScriptA]E \[ScriptCapitalR]])/(1-\[Beta] (\[CapitalGamma]^(1-\[Rho]) (1-\[Mho])));
                                                                                                                                                   
(* Combined discount factor for normalized problem *)                                                                                              
\[Bet] = \[ScriptCapitalR] \[Beta] \[CapitalGamma]^(1-\[Rho]);

