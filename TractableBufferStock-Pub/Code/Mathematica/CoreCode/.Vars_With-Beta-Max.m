(* ::Package:: *)

If[NameQ["ParamsAreSet"]==True,If[ParamsAreSet==True,Print["This file should be executed before parameter values are defined."];Abort[]]]


(* Variables *)

(* \[Lambda] is the MPS for perfect foresight problem (like unemployed consumers); *)
(* \[Kappa] is the MPC for perfect foresight problem (like unemployed consumers)  *)
\[Lambda]=((R)^-1) ((R) \[Beta])^(1/\[Rho])(1-PDies); (* Here we incorporate death rate into ctDiscrete framework.*)
\[Kappa]=1-\[Lambda];
\[Lambda]InfHor=((R)^-1) ((R) \[Beta])^(1/\[Rho]); 
\[Kappa]InfHor=1-\[Lambda]InfHor;

(* \[CapitalGamma] is the growth factor conditional on remaining employed *)
\[CapitalGamma]=\[GothicCapitalG]/(1-\[Mho]);
\[Gamma]=Log[\[CapitalGamma]];

(* SoiTax is the social insurance tax on the employed*)
SoiTax = \[Mho]*Severance;

(* \[ScriptCapitalR] is the return factor for the normalized problem *)
(* (R) is the return factor in levels (in parens to allow search-and-replace) *)
\[ScriptCapitalR]=(R)/\[CapitalGamma];
\[CapitalThorn]\[CapitalGamma] = ((R) \[Beta])^(1/\[Rho])/\[CapitalGamma];
(* \[CapitalThorn] is the growth patience factor *)
\[CapitalThorn]=((R) \[Beta])^(1/\[Rho]);
\[CapitalThorn]\[CapitalGamma]=\[CapitalThorn]/\[CapitalGamma];
\[CapitalThorn]\[GothicCapitalG]=\[CapitalThorn]/\[GothicCapitalG];
\[CapitalThorn]\[CapitalGamma]=\[CapitalThorn]/\[CapitalGamma];
\[CapitalThorn]Rtn=\[CapitalThorn]Rtn=\[CapitalThorn]/(R);
\[WeierstrassP]\[Gamma]=Log[\[CapitalThorn]\[CapitalGamma]];
\[WeierstrassP]rtn=Log[\[CapitalThorn]Rtn];
\[CapitalPi]=((\[CapitalThorn]\[CapitalGamma]^-\[Rho]-(1-\[Mho]))/\[Mho])^(1/\[Rho]);
\[CurlyPi]=(\[CapitalThorn]\[CapitalGamma]^-\[Rho]-1)/\[Mho]; (* It is easy to show that \[CapitalPi]=(1+\[CurlyPi])^(1/\[Rho]) *)
\[CapitalPi]Alt=(1+(\[CapitalThorn]\[CapitalGamma]^(-\[Rho])-1)/\[Mho])^(1/\[Rho]); (* Alternative writing of \[CapitalPi] useful for some purposes *)
\[GothicH]=(1/(1-\[GothicCapitalG]/(R)))*(1-SoiTax);  (* Human wealth, as a ratio to permanent labor income *)
\[Zeta]=\[ScriptCapitalR] \[Kappa] \[CapitalPi]Alt; (* A useful constant *)
\[Aleph]=(\[CapitalThorn]\[CapitalGamma]^-\[Rho] - 1)/\[Mho]; (* Another useful constant *)
\[Beta] = 1/(1+\[CurlyTheta]); (* Time preference factor *)
(R)= 1+(r);
\[GothicCapitalG]= 1+\[GothicG];
\[Mu]=\[ScriptCapitalR] \[Kappa] \[CapitalPi] + 1;

\[Beta]Max = (\[GothicCapitalG]^\[Rho]/(R))/((1-\[Mho])^(\[Rho]+1)); (* Equation DiscountMaxWhenFHWCFails *)
\[CurlyTheta]Min = \[Beta]Max^-1-1;

(* Target values of \[ScriptM]E,\[ScriptC]E,\[ScriptA]E,\[ScriptB]E,\[ScriptB]U,\[ScriptC]U *)
\[ScriptM]E=((1-SoiTax)*(\[Zeta]+1)-\[Zeta]*Severance)/(\[Zeta]-(1-\[Tau])\[ScriptCapitalR]+1); (*without social insurance, \[ScriptM]E=1+((R)/(\[CapitalGamma]+\[Zeta] \[CapitalGamma]-(R)))*)
\[ScriptM]ENoSoi=1+((R)/(\[CapitalGamma]+\[Zeta] \[CapitalGamma]-(R)));
\[ScriptC]E=(1-\[Tau]-\[ScriptCapitalR]^-1)\[ScriptM]E+(1-SoiTax)\[ScriptCapitalR]^-1;
\[ScriptA]E=(1-\[Tau])\[ScriptM]E-\[ScriptC]E;
\[ScriptB]E=\[ScriptA]E \[ScriptCapitalR];
\[ScriptY]E=\[ScriptA]E (\[ScriptCapitalR]-1) + 1-SoiTax;
\[ScriptX]E=Chop[\[ScriptY]E-\[ScriptC]E];
\[ScriptB]U=\[ScriptB]E;
\[ScriptM]U=((1-\[Tau])\[ScriptM]E-\[ScriptC]E)\[ScriptCapitalR]+Severance;
\[ScriptC]U=\[ScriptM]U \[Kappa];
\[GothicV] =1/(1-\[Beta]*(1-PDies) (((R) \[Beta](1-PDies))^((1/\[Rho])-1))); (* Here we incorporate death rate into ctDiscrete framework.*)
\[ScriptV]U=u[\[ScriptC]U] \[GothicV];
\[ScriptV]E=(u[\[ScriptC]E] + \[Beta] (\[CapitalGamma]^(1-\[Rho])) \[Mho] vUPF[\[ScriptA]E \[ScriptCapitalR]])/(1-\[Beta] (\[CapitalGamma]^(1-\[Rho]) (1-\[Mho])));
                                                                                                                                                   
(* Combined discount factor for normalized problem *)                                                                                              
\[Bet] = \[ScriptCapitalR] \[Beta] \[CapitalGamma]^(1-\[Rho]);



(* ::Input:: *)
(**)
