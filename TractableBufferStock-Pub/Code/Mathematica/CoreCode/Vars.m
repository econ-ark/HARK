(* ::Package:: *)

If[NameQ["ParamsAreSet"]==True,If[ParamsAreSet==True,Print["This file should be executed before parameter values are defined."];Abort[]]]


(* Variables *)

(* \[Lambda] is the MPS for perfect foresight problem (like unemployed consumers); *)
(* \[Kappa] is the MPC for perfect foresight problem (like unemployed consumers)  *)
\[Lambda]=((R)^-1) ((R) \[Beta])^(1/\[Rho])(1-PDies); (* Here we incorporate death rate into ctDiscrete framework.*)
\[Kappa]=1-\[Lambda];
\[Lambda]InfHor=((R)^-1) ((R) \[Beta])^(1/\[Rho]); (* Version without death *)
\[Kappa]InfHor=1-\[Lambda]InfHor; (* Version without death *)

(* \[CapitalGamma] is the growth factor conditional on remaining employed *)
\[CapitalGamma]=\[GothicCapitalG]/(1-\[Mho]);
\[Gamma]=Log[\[CapitalGamma]];

(* SoiTax is the social insurance tax on the employed in the Carroll and Jeanne paper *)
SoiTax = \[Mho]*Severance;

(* \[ScriptCapitalR] is the return factor for the normalized problem *)
(* (R) is the return factor in levels *)
(* Written in parens wherever possible to someday allow search-and-replace *)
(* Probably not with \[GothicCapitalR] because that's used as portfolio-weighted return in other notebooks *)
\[ScriptCapitalR]=(R)/\[CapitalGamma];
\[CapitalThorn]\[CapitalGamma] = ((R) \[Beta])^(1/\[Rho])/\[CapitalGamma];
(* Absolute patience factor     *) \[CapitalThorn]=((R) \[Beta])^(1/\[Rho]);
(* Return   patience factor     *) \[CapitalThorn]Rtn=\[CapitalThorn]/(R);
(* Growth   patience factor-TBS *) \[CapitalThorn]\[CapitalGamma]=\[CapitalThorn]/\[CapitalGamma];
(* Growth   patience factor-PF  *) \[CapitalThorn]\[GothicCapitalG]=\[CapitalThorn]/\[GothicCapitalG];
(* \[CapitalThorn]\[Mho]=\[CapitalThorn] (1-\[Mho])^(1/\[Rho])/\[CapitalGamma]; *)
(* Absolute patience rate       *) \[WeierstrassP]=Log[\[CapitalThorn]];
(* Return   patience rate       *) \[WeierstrassP]rtn=Log[\[CapitalThorn]Rtn];
(* Growth   patience rate-TBS   *) \[WeierstrassP]\[Gamma]=Log[\[CapitalThorn]\[CapitalGamma]];
(* \[WeierstrassP]\[Mho]=Log[\[CapitalThorn]\[Mho]]; *)

(* Consumption growth augmentation factor *) \[CapitalPi]=((\[CapitalThorn]\[CapitalGamma]^-\[Rho]-(1-\[Mho]))/\[Mho])^(1/\[Rho]);
(* Growth impatience contribution to \[CapitalPi]    *) \[CurlyPi]=(\[CapitalThorn]\[CapitalGamma]^-\[Rho]-1)/\[Mho]; (* It is easy to show that \[CapitalPi]=(1+\[CurlyPi])^(1/\[Rho]) *)
(* Alternative \[CapitalPi] useful for some purposes *) \[CapitalPi]Alt=(1+\[CurlyPi])^(1/\[Rho]);
(* Human wealth ratio to permanent income *) \[GothicH]=(1/(1-\[GothicCapitalG]/(R)))*(1-SoiTax);  
(* A useful object *) \[Zeta]=\[ScriptCapitalR] \[Kappa] \[CapitalPi]Alt; 
(* A useful object *) \[Aleph]=(\[CapitalThorn]\[CapitalGamma]^-\[Rho] - 1)/\[Mho]; 
(* A useful object *) \[Mu]=\[ScriptCapitalR] \[Kappa] \[CapitalPi] + 1;
(* Time preference factor *) \[Beta] = 1/(1+\[CurlyTheta]); 
(* Interest factor        *) (R)= 1+(r);
(* Wage growth factor     *) \[GothicCapitalG]= 1+\[GothicG];



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

(* Useful boundaries or limits *)
\[Beta]MaxMax=\[Beta]MaxWhereRICHoldsExactly=(R)^(\[Rho]-1);
\[CurlyTheta]MinMin=\[CurlyTheta]MinWhereRICHoldsExactly=(\[Beta]MaxWhereRICHoldsExactly^-1)-1; 
\[Beta]MaxWhereGIC\[CapitalGamma]HoldsExactly=((R)^-1) \[CapitalGamma]^\[Rho]; 
\[CurlyTheta]MinWhereGIC\[CapitalGamma]HoldsExactly=(\[Beta]MaxWhereGIC\[CapitalGamma]HoldsExactly^-1)-1;
\[Beta]MaxWhereGIC\[GothicCapitalG]HoldsExactly=((R)^-1) \[GothicCapitalG]^\[Rho]; 
\[CurlyTheta]MinWhereGIC\[GothicCapitalG]HoldsExactly=(\[Beta]MaxWhereGIC\[GothicCapitalG]HoldsExactly^-1)-1;
\[Beta]MaxWhereGICTBSHoldsExactly=(((1-\[Mho])(R))^-1) \[CapitalGamma]^\[Rho]; 
\[CurlyTheta]MinWhereGICTBSHoldsExactly=(\[Beta]MaxWhereGICTBSHoldsExactly^-1)-1;

(* Tests of various conditions, performed using ambient parameter values *)
RIC := \[CapitalThorn]Rtn < 1;
GIC\[CapitalGamma]:=\[CapitalThorn]\[CapitalGamma]<1;
GIC\[GothicCapitalG]:=\[CapitalThorn]\[GothicCapitalG]<1;
GICTBS:=(\[CapitalThorn]\[CapitalGamma](1-\[Mho]))^(1/\[Rho]) < 1;
FHWC\[CapitalGamma] := \[CapitalThorn]\[CapitalGamma] < 1;
FHWC\[GothicCapitalG] := \[CapitalThorn]\[GothicCapitalG] < 1;

(* Values of objects that are useful but always derived and never directly set *) 
RBase=1+rBase;
\[Beta]Base=1/(1+\[CurlyTheta]Base);
\[CapitalThorn]Base=(RBase \[Beta]Base)^(1/\[Rho]Base);
\[GothicCapitalG]Base=1+\[GothicG]Base;
\[CapitalGamma]Base=\[GothicCapitalG]Base/(1-\[Mho]Base);


(* ::Input:: *)
(**)
