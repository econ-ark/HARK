(* ::Package:: *)

If[NameQ["ParamsAreSet"]==True,If[ParamsAreSet==True,Print["This file should be executed before parameter values are defined."];Abort[]]]


(* Functions *)
(* Defines the functions used in solving the model *)
(* Expressions like eq:natural in the code refer to the corresponding equation in TractableBufferStock.tex *)
(* (The archive containing TractableBufferStock.tex along with this software can be downloaded from *)
(* http://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/ctDiscrete.zip *)

CheckFor\[CapitalGamma]Impatience :=If[\[CapitalThorn]\[CapitalGamma]>= 1. && (R < \[CapitalGamma]),Print["Aborting: Employed Consumer Not Growth Impatient."];Abort[]];
CheckForRImpatience :=If[\[CapitalThorn]Rtn>= 1.,Print["Aborting: Employed Consumer Not Return Impatient."];Abort[]];

(* Define utility function from highest to lowest derivative to avoid Mathematica difficulties *)
Clear[u];
u'''[\[ScriptC]_] := -\[Rho](-\[Rho]-1) \[ScriptC]^(-\[Rho]-2);
u''[\[ScriptC]_] := -\[Rho] \[ScriptC]^(-\[Rho]-1);
u'[\[ScriptC]_] := \[ScriptC]^-\[Rho];
u[\[ScriptC]_] := If[ Chop[\[Rho]-1] != 0, (\[ScriptC]^(1-\[Rho]))/(1-\[Rho]), Log[\[ScriptC]] ];

(* The consumption function is defined differently below and above the limit \[ScriptM]Top *)
(* Below, it is the result of interpolation among a set of gridpoints *)
(* Above, it is the result of extrapolation of a function defined at \[ScriptM]Top *)

(* Precautionary saving for Employed consumer is the difference between consumption of the perfect foresight consumer and the consumer facing uncertainty *)
psavEInterp[\[ScriptM]_] := cEPF[\[ScriptM]]-cEInterp[\[ScriptM]];
(* Extrapolated precautionary saving is defined below *)
cEExtrap[\[ScriptM]_]    := cEPF[\[ScriptM]]-psavEExtrap[\[ScriptM]];

(* Mathematica's built-in differentiation operator ' does not work well with conditionals, so define the derivatives explicitly *)
psavE'''[\[ScriptM]_] := psavEInterp'''[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
psavE'''[\[ScriptM]_] := psavEExtrap'''[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
psavE''[\[ScriptM]_] := psavEInterp''[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
psavE''[\[ScriptM]_] := psavEExtrap''[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
psavE'[\[ScriptM]_] := psavEInterp'[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
psavE'[\[ScriptM]_] := psavEExtrap'[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
psavE[\[ScriptM]_] := psavEInterp[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
psavE[\[ScriptM]_] := psavEExtrap[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;

(* Consumption function for Employed consumer *)
cE'''[\[ScriptM]_] := cEInterp'''[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
cE'''[\[ScriptM]_] := -psavE'''[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
cE''[\[ScriptM]_] := cEInterp''[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
cE''[\[ScriptM]_] := -psavE''[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
cE'[\[ScriptM]_] := cEInterp'[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
cE'[\[ScriptM]_] := \[Kappa]-psavE'[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
cE[\[ScriptM]_] := cEInterp[\[ScriptM]] /; \[ScriptM] < \[ScriptM]Top;
cE[\[ScriptM]_] := cEPF[\[ScriptM]]-psavE[\[ScriptM]] /; \[ScriptM] >= \[ScriptM]Top;
SetAttributes[cE,Listable]; (* Allows function to be applied to a list and return a list *)

(* Find limiting MPC as \[ScriptM] approaches zero, using eq:natural and nearby results *)
\[Natural]EFuncLim0[\[Kappa]Et_] := \[Bet] \[ScriptCapitalR] \[Mho] (\[Kappa] \[ScriptCapitalR] ((1-\[Kappa]Et)/\[Kappa]Et))^(-\[Rho]-1) \[Kappa];
(* The first version of the function below is easier to understand because it corresponds directly to eq:MPCat0, but under some parameter values FindRoot tries \[Kappa]E = 1 which blows up *)
(* The second version (which is the operative one) deals with this problem by searching for a value which is exponentiated to generate \[Kappa]E -- this can never reach 1 *)
\[Kappa]ELim0Find := \[Kappa]ESeek /.  FindRoot[ \[Kappa]ESeek == \[Natural]EFuncLim0[\[Kappa]ESeek]/(1 + \[Natural]EFuncLim0[\[Kappa]ESeek]), {\[Kappa]ESeek, 0.9, 0.99999999999999}];
\[Kappa]ELim0Find := Block[{\[Kappa]ELim0Soln},
  Off[FindRoot::cvmit]; (* Turn off warning about insufficient accuracy *)
  \[Kappa]ELim0Soln=Exp[-1/\[Kappa]EInvSeek/.FindRoot[Exp[-1/\[Kappa]EInvSeek]==\[Natural]EFuncLim0[Exp[-1/\[Kappa]EInvSeek]]/(1+\[Natural]EFuncLim0[Exp[-1/\[Kappa]EInvSeek]]),{\[Kappa]EInvSeek,0.5,100}]];
  On[FindRoot::cvmit]; (* Turn warning back on *)
  Return[\[Kappa]ELim0Soln]
];

(* Perfect foresight solutions for employed consumer *)
cEPF[\[ScriptM]_] := (\[ScriptM]-(1-SoiTax)+\[GothicH])\[Kappa]; (* eq:normC in PerfForesightCRRA.tex *)
(* The If statement below implements the limit as \[Rho] approaches zero (the logarithmic case) *)
vEPF[\[ScriptM]_] := u[cEPF[\[ScriptM]]] \[GothicV]+If[ Chop[\[Rho]-1] == 0, Log[ (R) \[Beta] ] (\[Beta]/((\[Beta]-1)^2)),0];
(* Perfect foresight solutions for unemployed consumer *)
cUPF[\[ScriptM]_]  := \[Kappa] \[ScriptM];
(* The If statement below implements the limit as \[Rho] approaches zero (the logarithmic case) *)
vUPF[\[ScriptM]_]  := u[ \[Kappa] \[ScriptM] ] \[GothicV]+If[ Chop[\[Rho]-1] == 0, Log[ (R) \[Beta] ] (\[Beta]/((\[Beta]-1)^2)),0];

(* Stable loci for phase diagram *)
\[ScriptC]EDelEqZero[\[ScriptM]_] := ((\[ScriptCapitalR] \[Kappa] \[CapitalPi])/(1+\[ScriptCapitalR] \[Kappa] \[CapitalPi]))\[ScriptM]+(Severance \[Kappa] \[CapitalPi])/(1+\[ScriptCapitalR] \[Kappa] \[CapitalPi]); (* eq:DceEq0 in TractableBufferStock.tex *)
\[ScriptM]EDelEqZero[\[ScriptM]_] := (\[CapitalGamma]/(R))*(1-SoiTax)+(1-\[CapitalGamma]/(R))\[ScriptM]; (* eq:mDelEqZero *)

(* Dynamics (forward and backward) *)
\[ScriptM]Etp1Fromt[\[ScriptM]Et_,\[ScriptC]Et_] := ((1-\[Tau])\[ScriptM]Et-\[ScriptC]Et)\[ScriptCapitalR]+1-SoiTax; (* Dynamic budget constraint *)
\[ScriptC]EtFromtp1[\[ScriptM]Etp1_,\[ScriptC]Etp1_] := \[CapitalGamma] ((R) \[Beta])^(-1/\[Rho]) \[ScriptC]Etp1  (1+\[Mho] ((\[ScriptC]Etp1/(\[Kappa] ((1-\[Tau])\[ScriptM]Etp1-(1-SoiTax)+Severance)))^\[Rho]-1))^(-1/\[Rho]); (* Reverse shooting equation for \[ScriptC] from eq:cReverse *)
\[ScriptM]EtFromtp1[\[ScriptM]Etp1_,\[ScriptC]Etp1_] := (\[CapitalGamma]/(R))(\[ScriptM]Etp1-1+SoiTax)+ \[ScriptC]EtFromtp1[\[ScriptM]Etp1,\[ScriptC]Etp1]/(1-\[Tau]); (* eq:mReverse *)
\[Kappa]EtFromtp1[\[ScriptM]Etp1_,\[ScriptC]Etp1_,\[Kappa]Etp1_,\[ScriptM]Et_,\[ScriptC]Et_] := Block[{\[ScriptC]Utp1 = \[Kappa] (((1-\[Tau])\[ScriptM]Et-\[ScriptC]Et) \[ScriptCapitalR]+Severance),\[Natural]},
  \[Natural] =  \[Bet] \[ScriptCapitalR] (1/u''[\[ScriptC]Et]) ((1-\[Mho]) u''[\[ScriptC]Etp1] \[Kappa]Etp1 + \[Mho] u''[\[ScriptC]Utp1] \[Kappa]);
  Return[\[Natural]/(1+\[Natural])]
]; (* eq:naturalSolved *)

(* Second derivative of consumption function *)
\[Kappa]EPtFromtp1[\[ScriptM]Etp1_,\[ScriptC]Etp1_,\[Kappa]Etp1_,\[ScriptM]Et_,\[ScriptC]Et_,\[Kappa]Et_,\[Kappa]EPtp1_] := Block[{\[ScriptC]Utp1=(((1-\[Tau])\[ScriptM]Et-\[ScriptC]Et)\[ScriptCapitalR]+Severance) \[Kappa]},
   (\[Bet] \[ScriptCapitalR]^2 (1-\[Kappa]Et)^2 (\[Mho] \[Kappa]^2 u'''[\[ScriptC]Utp1]+(1-\[Mho]) \[Kappa]Etp1^2 u'''[\[ScriptC]Etp1]+(1-\[Mho]) u''[\[ScriptC]Etp1] \[Kappa]EPtp1 )-(\[Kappa]Et)^2 u'''[\[ScriptC]Et])/
   (u''[\[ScriptC]Et]+\[Bet] \[ScriptCapitalR] (\[Mho] \[Kappa] u''[\[ScriptC]Utp1]+(1-\[Mho]) \[Kappa]Etp1 u''[\[ScriptC]Etp1]))
]; (* eq:kappaPReverse *)

(* %% *) (* Second order Taylor expansion around the target *)
(* %% *) Clear[cETaylorNearTarget];
(* %% *) cETaylorNearTarget''[\[FilledUpTriangle]_]  :=                     \[Kappa]EP;
(* %% *) cETaylorNearTarget'[\[FilledUpTriangle]_]   :=      \[Kappa]E +         \[FilledUpTriangle] \[Kappa]EP;
(* %% *) cETaylorNearTarget[\[FilledUpTriangle]_]    := \[ScriptC]E+\[FilledUpTriangle] \[Kappa]E +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EP;
(* %% *) 
(* %% *) (* Third order Taylor expansion around the target *)
(* %% *) Clear[cETaylorNearTarget];
(* %% *) cETaylorNearTarget'''[\[FilledUpTriangle]_] :=                         (1/1) (\[FilledUpTriangle]^0) \[Kappa]EPP;
(* %% *) cETaylorNearTarget''[\[FilledUpTriangle]_]  :=                     \[Kappa]EP+(1/1) (\[FilledUpTriangle]^1) \[Kappa]EPP;
(* %% *) cETaylorNearTarget'[\[FilledUpTriangle]_]   :=      \[Kappa]E +         \[FilledUpTriangle] \[Kappa]EP+(1/2) (\[FilledUpTriangle]^2) \[Kappa]EPP;
(* %% *) cETaylorNearTarget[\[FilledUpTriangle]_]    := \[ScriptC]E+\[FilledUpTriangle] \[Kappa]E +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EP+(1/6) (\[FilledUpTriangle]^3) \[Kappa]EPP;
(* %% *) 
(* %% *) (* Fourth order Taylor expansion around the target *)
(* %% *) Clear[cETaylorNearTarget];
(* %% *) cETaylorNearTarget''''[\[FilledUpTriangle]_]:=                                                 (\[FilledUpTriangle]^0)\[Kappa]EPPP;
(* %% *) cETaylorNearTarget'''[\[FilledUpTriangle]_] :=                                                 (\[FilledUpTriangle]^1)\[Kappa]EPPP;
(* %% *) cETaylorNearTarget''[\[FilledUpTriangle]_]  :=                     \[Kappa]EP+(1/1) (\[FilledUpTriangle]^1) \[Kappa]EPP +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EPPP;
(* %% *) cETaylorNearTarget'[\[FilledUpTriangle]_]   :=      \[Kappa]E +         \[FilledUpTriangle] \[Kappa]EP+(1/2) (\[FilledUpTriangle]^2) \[Kappa]EPP +(1/6) (\[FilledUpTriangle]^3)\[Kappa]EPPP;
(* %% *) cETaylorNearTarget[\[FilledUpTriangle]_]    := \[ScriptC]E+\[FilledUpTriangle] \[Kappa]E +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EP+(1/6) (\[FilledUpTriangle]^3) \[Kappa]EPP +(1/24)(\[FilledUpTriangle]^4)\[Kappa]EPPP; 

(* Fifth order Taylor expansion around the target is necessary for exactly matching the polynomial at c,c',c'' at adjacent points *)
Clear[cETaylorNearTarget];
cETaylorNearTarget'''''[\[FilledUpTriangle]_] :=                                                            +  (1/1)(\[FilledUpTriangle]^0)\[Kappa]EPPPP;
cETaylorNearTarget''''[\[FilledUpTriangle]_]  :=                                                 (\[FilledUpTriangle]^0)\[Kappa]EPPP +  (2/2)(\[FilledUpTriangle]^1)\[Kappa]EPPPP;
cETaylorNearTarget'''[\[FilledUpTriangle]_]   :=                                                 (\[FilledUpTriangle]^1)\[Kappa]EPPP +  (1/2)(\[FilledUpTriangle]^2)\[Kappa]EPPPP;
cETaylorNearTarget''[\[FilledUpTriangle]_]    :=                     \[Kappa]EP+(1/1) (\[FilledUpTriangle]^1) \[Kappa]EPP +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EPPP +  (1/6)(\[FilledUpTriangle]^3)\[Kappa]EPPPP;
cETaylorNearTarget'[\[FilledUpTriangle]_]     :=      \[Kappa]E +         \[FilledUpTriangle] \[Kappa]EP+(1/2) (\[FilledUpTriangle]^2) \[Kappa]EPP +(1/6) (\[FilledUpTriangle]^3)\[Kappa]EPPP + (1/24)(\[FilledUpTriangle]^4)\[Kappa]EPPPP;
cETaylorNearTarget[\[FilledUpTriangle]_]      := \[ScriptC]E+\[FilledUpTriangle] \[Kappa]E +(1/2) (\[FilledUpTriangle]^2)\[Kappa]EP+(1/6) (\[FilledUpTriangle]^3) \[Kappa]EPP +(1/24)(\[FilledUpTriangle]^4)\[Kappa]EPPP +(1/120)(\[FilledUpTriangle]^5)\[Kappa]EPPPP;

vE'[\[ScriptM]_] := u'[cE[\[ScriptM]]]; (* Envelope result *)
vE[\[ScriptM]_] := \[ScriptV]Bot + NIntegrate[u'[cEInterp[\[ScriptM]\[Bullet]]],{\[ScriptM]\[Bullet],\[ScriptM]Vec[[1]], \[ScriptM]}] /; \[ScriptM] < \[ScriptM]Vec[[1]]; (* If \[ScriptM] is below the bottom, use consumption function to construct \[ScriptV] *)
vE[\[ScriptM]_] := vEInterp[\[ScriptM]] /; \[ScriptM]Vec[[-1]] >=  \[ScriptM] >=  \[ScriptM]Vec[[1]]; (* If we are in interior of interpolating points, use interpolating approximation *)
vE[\[ScriptM]_] := \[ScriptV]Top + NIntegrate[u'[cEInterp[\[ScriptM]\[Bullet]]],{\[ScriptM]\[Bullet],\[ScriptM]Vec[[-1]], \[ScriptM]}] /; \[ScriptM] > \[ScriptM]Vec[[-1]]; (* If \[ScriptM] is above the top, use consumption function to construct \[ScriptV] *)


(* BackShoot iterates the reverse Euler equation until it reaches a point outside some predefined boundaries *)

(* 
The only subtlety here is why our test is not whether \[ScriptM]Prev < \[ScriptM]MinBound rather than \[ScriptM]Prev < \[ScriptM]MaxPermitted.
The answer is that inaccuracies in the high-order approximation to the consumption function around the target 
lead to ever-larger errors in the consumption function as it approaches \[ScriptM]MinBound.  We therefore compute the analytical
structure of the consumption function at \[ScriptM]MinBound and connect to a point that is sufficiently far away not to be 
much distorted by those inaccuracies.
*)
BackShoot[InitialPoints_] := Block[{\[ScriptM]Prev,\[ScriptC]Prev,\[Kappa]Prev,\[ScriptV]Prev,\[Kappa]PPrev
,\[ScriptM]MaxPermitted=4*\[ScriptM]E,\[ScriptM]MinBound=1-Severance/\[ScriptCapitalR],\[ScriptM]Buffer=\[ScriptM]E-\[ScriptM]MinBound,Counter = 0,PointsList =InitialPoints
},
 {\[ScriptM]Prev,\[ScriptC]Prev,\[Kappa]Prev,\[ScriptV]Prev,\[Kappa]PPrev} = Take[InitialPoints[[-1]],5];
 \[ScriptM]MinPermitted=\[ScriptM]MinBound+(\[ScriptM]Buffer/10); 
 If[VerboseOutput != False, Print["Solving ..."]];
 While[
  \[ScriptM]Prev > \[ScriptM]MinPermitted && \[ScriptM]Prev <= \[ScriptM]MaxPermitted, 
    AppendTo[
        PointsList
          ,{\[ScriptM]Now,\[ScriptC]Now} = {\[ScriptM]EtFromtp1[\[ScriptM]Prev,\[ScriptC]Prev],\[ScriptC]EtFromtp1[\[ScriptM]Prev,\[ScriptC]Prev]};
          \[Kappa]Now = \[Kappa]EtFromtp1[\[ScriptM]Prev,\[ScriptC]Prev,\[Kappa]Prev,\[ScriptM]Now,\[ScriptC]Now];
          \[Kappa]PNow = \[Kappa]EPtFromtp1[\[ScriptM]Prev,\[ScriptC]Prev,\[Kappa]Prev,\[ScriptM]Now,\[ScriptC]Now,\[Kappa]Now,\[Kappa]PPrev];
          \[ScriptV]Now = u[\[ScriptC]Now] + \[Beta] (\[CapitalGamma]^(1-\[Rho])) ((1-\[Mho]) \[ScriptV]Prev + \[Mho] vUPF[((1-\[Tau])\[ScriptM]Now-\[ScriptC]Now)\[ScriptCapitalR]+Severance ]);
          {\[ScriptM]Prev,\[ScriptC]Prev,\[Kappa]Prev,\[ScriptV]Prev,\[Kappa]PPrev} = {\[ScriptM]Now,\[ScriptC]Now,\[Kappa]Now,\[ScriptV]Now,\[Kappa]PNow}
    ];
    If[\[ScriptM]Prev <= \[ScriptM]MinPermitted && VerboseOutput != False
      ,Print["  Below \[ScriptM]MinPermitted after ",Counter," backwards Euler iterations."];
       Print["Last 2 Points:",MatrixForm[PointsList[[-2;;-1]]]]];
    If[\[ScriptM]Prev > \[ScriptM]MaxPermitted && VerboseOutput != False
      ,Print["  Above \[ScriptM]MaxPermitted after ",Counter," backwards Euler iterations."];
       Print["Last 2 Points:",MatrixForm[PointsList[[-2;;-1]]]]];
  Counter++];
  Return[Most[PointsList]]
]; (* Returns all points but the last one (that's what Most[] does), which is the one that exceeded the permitted boundaries *)

(* Iterate starting from the steady state plus \[FilledUpTriangle] *)
EulerPointsStartingFromSSPlus[\[FilledUpTriangle]_] := Block[{},
  \[ScriptM]Start = \[ScriptM]E + \[FilledUpTriangle];
  \[Kappa]Start   = \[Kappa]E   + \[Kappa]EP \[FilledUpTriangle] + \[Kappa]EPP (\[FilledUpTriangle]^2)/2 + \[Kappa]EPPP (\[FilledUpTriangle]^3)/6;
  \[Kappa]PStart  =        \[Kappa]EP   + \[Kappa]EPP \[FilledUpTriangle]       + \[Kappa]EPPP (\[FilledUpTriangle]^2)/2;
  \[ScriptC]Start = cETaylorNearTarget[\[FilledUpTriangle]];
  \[ScriptV]Start = \[ScriptV]E + NIntegrate[u'[cETaylorNearTarget[\[Bullet]]],{\[Bullet],0,\[FilledUpTriangle]}];
  StartPoint = {\[ScriptM]Start,\[ScriptC]Start,\[Kappa]Start,\[ScriptV]Start,\[Kappa]PStart};
  BackShoot[{StartPoint}]
];

FindStableArm := Block[{},
  CheckFor\[CapitalGamma]Impatience;
  CheckForRImpatience;
  \[Kappa]EMax = \[Kappa]ELim0Find;
(* Digest the \[ScriptC]E derivatives into numbers; must do this before evaluating EulerPointsStartingFromSSPlus *)
  \[Kappa]E   = Select[\[ScriptC]EPAnalytical, 0 <= # <= 1 &][[1]];
  If[Not[MachineNumberQ[\[Kappa]E]],Print["\[Kappa]E = ",\[Kappa]E," is not a number in FindStableArm."];Abort[]];
(* Higher derivatives are recursive functions of lower ones, so make the necessary substitutions to obtain them *)
  \[Kappa]EP   =   \[ScriptC]EPPAnalytical                                                        /. \[ScriptC]EP -> \[Kappa]E;
  \[Kappa]EPP  =  (\[ScriptC]EPPPAnalytical                                       /. \[ScriptC]EPP -> \[Kappa]EP) /. \[ScriptC]EP -> \[Kappa]E;
  \[Kappa]EPPP = ((\[ScriptC]EPPPPAnalytical                     /. \[ScriptC]EPPP -> \[Kappa]EPP)/. \[ScriptC]EPP -> \[Kappa]EP) /. \[ScriptC]EP -> \[Kappa]E;
  \[Kappa]EPPPP=(((\[ScriptC]EPPPPPAnalytical /. \[ScriptC]EPPPP -> \[Kappa]EPPP)/. \[ScriptC]EPPP -> \[Kappa]EPP)/. \[ScriptC]EPP -> \[Kappa]EP) /. \[ScriptC]EP -> \[Kappa]E;

  StableArmBelowSS = Sort[EulerPointsStartingFromSSPlus[-\[CurlyEpsilon]]];
(* %% *)  D\[ScriptM]First=StableArmBelowSS[[-1,1]]-StableArmBelowSS[[-2,1]];    (* Size of gap between 1st Euler point and 2nd *)
(* %% *)  IterLength=Length[StableArmBelowSS];
(* %% *)  PrecisionAugmentationFactor = Floor[1+0.1/(\[Mho]+0.1)]; (* Smaller \[Mho] yields kinkier consumption function; increase precision accordingly *)
(* %% *)  If[VerboseOutput != False  && PrecisionAugmentationFactor>1, Print["Solving again from different starting points to augment precision."]];
(* %% *)  Do[StableArmBelowSS = Join[
(* %% *)    StableArmBelowSS
(* %% *)    ,Take[EPList=EulerPointsStartingFromSSPlus[-\[CurlyEpsilon]  -D\[ScriptM]First*(i/PrecisionAugmentationFactor)],Min[Length[EPList],IterLength]]]
(* %% *)                   ,{i,PrecisionAugmentationFactor-1}]; 
(* %% *)  StableArmBelowSS = Sort[StableArmBelowSS];
  StableArmAboveSS = EulerPointsStartingFromSSPlus[\[CurlyEpsilon]];  
  {\[ScriptM]Bot,\[ScriptC]Bot,\[Kappa]Bot,\[ScriptV]Bot,\[Kappa]PBot} = Take[StableArmBelowSS[[ 1]],5];
  {\[ScriptM]Top,\[ScriptC]Top,\[Kappa]Top,\[ScriptV]Top,\[Kappa]PTop} = Take[StableArmAboveSS[[-1]],5];
  ETarget = {\[ScriptM]E,\[ScriptC]E,\[Kappa]E,\[ScriptV]E,\[Kappa]EP};
  StableArmPoints = Sort[Union[Join[StableArmBelowSS,{ETarget},StableArmAboveSS]]];
  {\[ScriptM]Vec,\[ScriptC]Vec,\[Kappa]Vec,\[ScriptV]Vec,\[Kappa]PVec} = Take[Transpose[StableArmPoints],5];
uPVec   = u'[\[ScriptC]Vec];
uPPVec  = u''[\[ScriptC]Vec];
uPPPVec = u'''[\[ScriptC]Vec];
\[ScriptV]PVec   = uPVec;
\[ScriptV]PPVec  = uPPVec \[Kappa]Vec;

(* This is the value such that, when integrated, the interpolating function matches the last descending Euler point *)
\[Kappa]EPAtZero = 2 (\[ScriptC]Vec[[1]]-((1-\[Tau])\[ScriptM]Vec[[1]]+Severance/\[ScriptCapitalR])\[Kappa]EMax)/(((1-\[Tau])\[ScriptM]Vec[[1]]+Severance/\[ScriptCapitalR])^2); 
cEPoints = Join[{{{-Severance/\[ScriptCapitalR]},0.,\[Kappa]EMax,\[Kappa]EPAtZero}},Transpose[{Partition[\[ScriptM]Vec,1],\[ScriptC]Vec,\[Kappa]Vec,\[Kappa]PVec}]];
cEInterp = Interpolation[cEPoints];
vEInterp = Interpolation[Transpose[{Partition[\[ScriptM]Vec,1],\[ScriptV]Vec,u'[\[ScriptC]Vec],u''[\[ScriptC]Vec] \[Kappa]Vec}]];
(* Caclulate the derivatives of the precautionary saving function necessary for constructing the extrapolation *)

s0=psavEInterp[\[ScriptM]Top];
s1=psavEInterp'[\[ScriptM]Top];
s2=psavEInterp''[\[ScriptM]Top];
s3=psavEInterp'''[\[ScriptM]Top];

(* Now use the automatically derived analytical formulas to produce numerical values for the parameters for the extrapolation *)

\[Phi]0 = \[Phi]Analytical0;
\[Phi]1 = \[Phi]Analytical1;
\[Gamma]0 = \[Gamma]Analytical0;
\[Gamma]1 = \[Gamma]Analytical1;

(* Define the precautionary saving function extrapolation -- defined here because only now do the coefficients have numerical values *)
Clear[psavEExtrap];
psavEExtrap[\[ScriptM]_] := Chop[Exp[\[Phi]0 - \[Phi]1 (\[ScriptM]-\[ScriptM]Top)] + Exp[\[Gamma]0 - \[Gamma]1 (\[ScriptM]-\[ScriptM]Top)] //N]; (* Chop removes any tiny imaginary part *)

]; (* End FindStableArm *)

SimAddAnotherPoint[\[ScriptM]\[ScriptC]Path_]:= Block[{},
  \[ScriptM]NextVal = \[ScriptM]Etp1Fromt[\[ScriptM]\[ScriptC]Path[[-1,1]],\[ScriptM]\[ScriptC]Path[[-1,2]]];
  \[ScriptC]NextVal = cE[\[ScriptM]NextVal];
  {\[ScriptM]NextVal,\[ScriptC]NextVal}];

SimAddAnotherPointUsing[\[ScriptC]Func_,\[ScriptM]\[ScriptC]Path_]:= Block[{},
  \[ScriptM]NextVal = \[ScriptM]Etp1Fromt[\[ScriptM]\[ScriptC]Path[[-1,1]],\[ScriptM]\[ScriptC]Path[[-1,2]]];
  \[ScriptC]NextVal = \[ScriptC]Func[\[ScriptM]NextVal];
  {\[ScriptM]NextVal,\[ScriptC]NextVal}];

SimGeneratePathFrom\[ScriptM]Start[\[ScriptM]Initial_,PeriodsToGo_] := Block[{},
  \[ScriptC]Initial = cE[\[ScriptM]Initial];
  \[ScriptM]\[ScriptC]Path = {{\[ScriptM]Initial,\[ScriptC]Initial}};
  Do[AppendTo[\[ScriptM]\[ScriptC]Path,SimAddAnotherPoint[\[ScriptM]\[ScriptC]Path]],{PeriodsToGo}];
];

SimGeneratePath[\[ScriptM]Initial_,PeriodsToGo_] := Block[{},
  \[ScriptC]Initial = cE[\[ScriptM]Initial];
  \[ScriptM]\[ScriptC]Path = {{\[ScriptM]EBase,\[ScriptC]EBase},{\[ScriptM]Initial,\[ScriptC]Initial}};
  Do[AppendTo[\[ScriptM]\[ScriptC]Path,SimAddAnotherPoint[\[ScriptM]\[ScriptC]Path]],{PeriodsToGo}];
];

SimGeneratePathUsing[\[ScriptC]Func_,\[ScriptM]Initial_,PeriodsToGo_] := Block[{},
  \[ScriptC]Initial = \[ScriptC]Func[\[ScriptM]Initial];
  \[ScriptM]\[ScriptC]Path = {{\[ScriptM]EBase,\[ScriptC]EBase},{\[ScriptM]Initial,\[ScriptC]Initial}};
  Do[AppendTo[\[ScriptM]\[ScriptC]Path,SimAddAnotherPointUsing[\[ScriptC]Func,\[ScriptM]\[ScriptC]Path]],{PeriodsToGo}];
];

(* Miscellaneous other useful functions *)

\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]_] := (((R) \[Beta])^(1/\[Rho]) (1+\[Mho] ((cEInterp[((1-\[Tau])\[ScriptM]-cEInterp[\[ScriptM]])\[ScriptCapitalR]+1-SoiTax]/(\[Kappa] ((\[ScriptM]-cEInterp[\[ScriptM]])\[ScriptCapitalR]+Severance)))^\[Rho]-1))^(1/\[Rho]));

ShowParams:=MatrixForm[
  Map[
    {#,ToExpression[#]}&
      ,{"\[Mho]","\[Rho]","\[Kappa]E","\[Kappa]","\[CapitalGamma]","\[ScriptCapitalR]","\[CapitalThorn]\[CapitalGamma]","\[CapitalThorn]Rtn","\[WeierstrassP]\[Gamma]","\[WeierstrassP]rtn","\[CapitalPi]","\[GothicH]","\[Beta] ","(R)","\[GothicCapitalG]","(r)","\[CurlyTheta]","\[GothicG]","\[ScriptM]E","\[ScriptC]E","\[Kappa]E","\[CurlyEpsilon]"}]];

SteadyStateVals= { \[ScriptB]E, \[ScriptM]E, \[ScriptC]E, \[ScriptA]E, \[Kappa]E, \[Kappa]EP, \[ScriptV]E};
