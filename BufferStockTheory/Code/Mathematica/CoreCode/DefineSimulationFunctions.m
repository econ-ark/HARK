(* ::Package:: *)

(*
Set up the routines needed for simulating the model
*)

Simulate[NumOfPeople_,NumOfPeriods_,bInitial_] := Block[{},

NumOfPointsInCDF = 200;  (* Number of points to use for the approximating CDF *)
ListOfPointsForApproximatingCDF = Table[Round[i], {i, 1, NumOfPeople, (NumOfPeople-1)/(NumOfPointsInCDF-1)}];
CDFTable = Table[i/(NumOfPeople-1) //N, {i,0, NumOfPeople-1}]; (* 0 to 1 *)

(* Variables organized by the moment in a period when they are realized *)
MidStateVars     = {"mt","pRatt","pLevt","\[Theta]t","\[Psi]t","CDFbtFunc","CDFmtFunc","CDFatFunc","CovpRattAndmt","cLevt","ct"};
EndStateVars     = {"at"};
MidStatsToCalc   = {"bt","CDFmt","CDFbt","CDFat","\[Kappa]t","cLevt"};
EndStatsToCalc   = {"CDFat"};
VarsToTakeMeanOf = {"ct","at","mt","bt","\[Theta]t","\[Psi]t","pRatt","pLevt","\[Kappa]t","cLevt"};
VarsToMake       = Union[MidStateVars,EndStateVars,MidStatsToCalc,EndStatsToCalc,VarsToTakeMeanOf];

(* Create empty variables to hold the results *)
Do[
  ToExpression[VarsToMake[[i]]<>"List = {0};"];
  ToExpression[VarsToMake[[i]]<>"Mean = {0};"];  
,{i,Length[VarsToMake]}];

(* Covariance of permanent income and resources *)
CovpRattAndat = CovpRattAndmt = CovpLevtAndat = CovpLevtAndmt = {0};

NumOfZeroGuys  = IntegerPart[\[WeierstrassP] NumOfPeople];(* Number of HHs receiving zero income in any particular year *)

If[VerboseOutput==True,Print["Constructing transitory shock distribution"]];

MakeShocks = True;

(* If we have already constructed the tran shock distribution for a previous sim *)
If[(\[WeierstrassP]>0 && Length[\[Theta]GridZero]  == NumOfPeople  && Abs[Variance[Log[Select[\[Theta]Grid,#>0 &]]]-\[Sigma]Tran^2]<0.001) 
   ,(* then use the existing data rather than constructing again from scratc*) 
   {\[Theta]Grid,\[Theta]Dist} = {\[Theta]GridZero,\[Theta]DistZero};
   MakeShocks=False;
   If[VerboseOutput==True,Print["Tran shock dist already in memory; not reconstructing."]];
   ];

If[MakeShocks==True,
   {\[Theta]Grid,\[Theta]Dist} = DiscreteApproxToMeanOneLogNormal[\[Sigma]Tran,NumOfPeople-NumOfZeroGuys];
   
   If[\[WeierstrassP]>0,
     (* then modify the shocks and probabilities to incorporate the zero-income risk *)
     \[Theta]Grid = \[Theta]Grid/\[WeierstrassP]Cancel;
     \[Theta]Grid = Join[\[Theta]Grid,Table[0,{NumOfZeroGuys}]];
     \[Theta]Dist = \[Theta]Dist \[WeierstrassP]Cancel;
     \[Theta]Dist = Join[\[Theta]Dist,Table[\[WeierstrassP]/NumOfZeroGuys,{NumOfZeroGuys}]];
     {\[Theta]GridZero,\[Theta]DistZero} = {\[Theta]Grid,\[Theta]Dist};
    ];
];

If[VerboseOutput==True,Print["Constructing permanent shock distribution"]];

(* If we have already constructed the distribution for a previous sim *)
MakeShocks = True;
If[(Length[\[Psi]Grid] == NumOfPeople) && Abs[Variance[Log[\[Psi]Grid]]-\[Sigma]Perm^2]<0.001
    ,(* then *) 
   MakeShocks=False;
   If[VerboseOutput==True,Print["Perm shock dist already in memory; not reconstructing."]];
   ];

If[MakeShocks == True,{\[Psi]Grid,\[Psi]Dist} = DiscreteApproxToMeanOneLogNormal[\[Sigma]Perm,NumOfPeople]];

Print["Finished constructing shocks."];

\[Theta]tList[[1]] = \[Theta]Grid[[RandomSample[Range[NumOfPeople]]]];
\[Psi]tList[[1]] = \[Psi]Grid[[RandomSample[Range[NumOfPeople]]]];

btList[[1]]    = Table[bInitial,{NumOfPeople}];
CDFbtList[[1]] = btList[[1]];
mtList[[1]]    = btList[[1]]+\[Theta]tList[[1]];
CDFmtList[[1]] = Sort[mtList[[1]]];
ctList[[1]]    = Map[\[ScriptC][#]&,mtList[[1]]];
\[Kappa]tList[[1]]    = Map[\[Kappa][#]&,mtList[[1]]];
atList[[1]]    = Chop[mtList[[1]]-ctList[[1]]];
CDFatList[[1]] = Sort[atList[[1]]];

pRattList[[1]] = \[Psi]tList[[1]];
pLevtList[[1]] = pRattList[[1]];

cLevtList[[1]] = pLevtList[[1]] ctList[[1]];

btMean[[1]]      = bInitial;
mtMean[[1]]      = Mean[mtList[[1]]];
ctMean[[1]]      = Mean[ctList[[1]]];
\[Kappa]tMean[[1]]      = Mean[\[Kappa]tList[[1]]];
atMean[[1]]      = Mean[atList[[1]]];
pRattMean[[1]]   = Mean[\[Psi]tList[[1]]];
pLevtMean[[1]]   = Mean[\[Psi]tList[[1]]];
cLevtMean[[1]]   = Mean[cLevtList[[1]]];

CovpRattAndat[[1]] = CovpRattAndmt[[1]] = 0;
CovpLevtAndat[[1]] = CovpLevtAndmt[[1]] = 0;

\[Theta]tMean[[1]] = 1;
\[Psi]tMean[[1]] = 1;

Print["Simulating."];
KeepSimulating[NumOfPeriods-1];

]; (* End Simulate *)


KeepSimulating[NumOfRemainingPeriods_] := Block[{},

NumOfPeople = Length[\[Theta]Grid];

LoopOverPeriods = NumOfRemainingPeriods;

While[LoopOverPeriods > 0,

CurrentPeriod = Length[btList]+1;

If[VerboseOutput==True,Print["Simulating Period ",CurrentPeriod]];

AppendTo[\[Theta]tList,\[Theta]Grid[[RandomSample[Range[NumOfPeople]]]]];
AppendTo[\[Psi]tList,\[Psi]Grid[[RandomSample[Range[NumOfPeople]]]]];
\[ScriptCapitalR]tList=((R)/(\[CapitalGamma] \[Psi]tList[[CurrentPeriod]]));
AppendTo[btList   ,\[ScriptCapitalR]tList atList[[CurrentPeriod-1]]];
AppendTo[CDFbtList,Sort[btList[[CurrentPeriod]]]];
AppendTo[mtList   ,btList[[-1]]+\[Theta]tList[[CurrentPeriod]]];
AppendTo[CDFmtList,Sort[mtList[[CurrentPeriod]]]];
AppendTo[ctList   ,\[ScriptC][mtList[[CurrentPeriod]]]];
AppendTo[\[Kappa]tList   ,\[Kappa][mtList[[CurrentPeriod]]]];
AppendTo[atList   ,mtList[[CurrentPeriod]]-ctList[[CurrentPeriod]]];
AppendTo[CDFatList,Sort[atList[[CurrentPeriod]]]];

AppendTo[pRattList,pRattList[[CurrentPeriod-1]] \[Psi]tList[[CurrentPeriod]]];
AppendTo[pLevtList,pLevtList[[CurrentPeriod-1]] \[Psi]tList[[CurrentPeriod]] \[CapitalGamma]];

AppendTo[cLevtList    ,ctList[[-1]] pLevtList[[-1]]];

AppendTo[CDFbtFuncList,Interpolation[Transpose[{CDFTable,CDFbtList[[-1]]}][[ListOfPointsForApproximatingCDF]],InterpolationOrder->1]];
AppendTo[CDFmtFuncList,Interpolation[Transpose[{CDFTable,CDFmtList[[-1]]}][[ListOfPointsForApproximatingCDF]],InterpolationOrder->1]];
AppendTo[CDFatFuncList,Interpolation[Transpose[{CDFTable,CDFatList[[-1]]}][[ListOfPointsForApproximatingCDF]],InterpolationOrder->1]];

AppendTo[CovpRattAndat,Covariance[pRattList[[-1]],atList[[-1]]]];
AppendTo[CovpRattAndmt,Covariance[pRattList[[-1]],mtList[[-1]]]];
AppendTo[CovpLevtAndat,Covariance[pLevtList[[-1]],atList[[-1]]]];
AppendTo[CovpLevtAndmt,Covariance[pLevtList[[-1]],mtList[[-1]]]];

(* Take the mean of the desired variables by constructing and then executing a command *)
Do[
  ToExpression[
    "AppendTo["<>VarsToTakeMeanOf[[i]]<>"Mean,Mean["<>VarsToTakeMeanOf[[i]]<>"List[["<>ToString[CurrentPeriod]<>"]]]];"
  ]
,{i,Length[VarsToTakeMeanOf]}
];

If[LowMem == True && LoopOverPeriods< NumOfRemainingPeriods-3,
  btList[[-3]] = CDFbtList[[-3]] = CDFatList[[-3]] = \[Kappa]tList[[-3]] = 
  atList[[-3]] = CDFatList[[-3]] = pRattList[[-3]] = pLevtList[[-3]] = \[Theta]tList[[-3]] = 
  \[Psi]tList[[-3]] = ctList[[-3]] = 0
  ];

LoopOverPeriods--;

] (* End While LoopOverPeriods *)
]; (* End KeepSimulating *)

