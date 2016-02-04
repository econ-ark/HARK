(* ::Package:: *)

MakeEmptyReport := Block[{},
  DescribeCol = {};
  Map[ToExpression[#<>"Col={}"]&,MPCPermTableScalars[[All,MmaNamePos]]];
  Map[ToExpression[#<>"Col={}"]&,MPCPermTableMeans[[All,MmaNamePos]]];
  Map[ToExpression[#<>"Col={}"]&,MeansVsTargetsScalars[[All,MmaNamePos]]];
(*  Map[ToExpression[#<>"Col={}"]&,MeansVsTargetsMeans[[All,MmaNamePos]]];*)
];

AddCurrentResultsToReport := Block[{},
  AppendTo[DescribeCol,ThisExperimentName];
  Map[ToExpression["AppendTo["<>#<>"Col,"<>#<>"Mean[[-1]]]"]&,MPCPermTableMeans[[All,MmaNamePos]]];
  Map[ToExpression["AppendTo["<>#<>"Col,"<>#<>"]"] &,MPCPermTableScalars[[All,MmaNamePos]]];
(*  Map[ToExpression["AppendTo["<>#<>"Col,"<>#<>"Mean[[-1]]]"]&,MeansVsTargetsMeans[[All,MmaNamePos]]];*)
  Map[ToExpression["AppendTo["<>#<>"Col,"<>#<>"]"] &,MeansVsTargetsScalars[[All,MmaNamePos]]];
];

MmaMPCPermTable := Block[{},
  LtxColNames = MPCPermTableFields[[All,LtxNamePos]];
  MmaColNames = Map[#<>"Col"&,MPCPermTableFields[[All,MmaNamePos]]];
  MatrixForm[Transpose[ToExpression[MmaColNames]],TableHeadings->{DescribeCol,MmaColNames}]
];

MmaMeansVsTargetsTable := Block[{},
  LtxColNames = MeansVsTargetsFields[[All,LtxNamePos]];
  MmaColNames = Map[#<>"Col"&,MeansVsTargetsFields[[All,MmaNamePos]]];
  MatrixForm[Transpose[ToExpression[MmaColNames]],TableHeadings->{DescribeCol,MmaColNames}]
];

LtxMPCPermTable := Block[{},
  Ampersands = Table["&",{Length[DescribeCol]}];
  EndOfLines = Table["\\\\", {Length[DescribeCol]}];
  OutFile = OpenWrite["/Volumes/Data/Work/MPCPerm/public/tables/MPCPermTable.out",FormatType->OutputForm,PageWidth->Infinity];
  MmaTableColNames = Flatten[Join[{{"DescribeCol","Ampersands"}},Table[{MmaColNames[[i]],"Ampersands"},{i,Length[MmaColNames]}]]];
  MmaTableColNames[[-1]]="EndOfLines";
  MmaTableRows = Transpose[ToExpression[MmaTableColNames]];
(*
  MmaTableLabels = Flatten[Join[{{"","&"}},Table[{LtxColNames[[i]],"&"},{i,Length[MmaColNames]}]]];
  MmaTableLabels[[-1]] = "\\\\";
  PrependTo[MmaTableRows,MmaTableLabels];
*)
  Write[OutFile,
   PaddedForm[
    TableForm[
      MmaTableRows,TableHeadings -> {None, None}
    ]
  , {4, 3}]
  ];
  Close[OutFile];
(*  FilePrint["/Volumes/Data/Work/MPCPerm/public/tables/MPCPermTable.out"];*)
];

LtxMeansVsTargetsTable := Block[{},
  Ampersands = Table["&",{Length[DescribeCol]}];
  EndOfLines = Table["\\\\", {Length[DescribeCol]}];
  OutFile = OpenWrite["/Volumes/Data/Work/MPCPerm/public/tables/MPCPermMeansVsTargets.out",FormatType->OutputForm,PageWidth->Infinity];
  MmaTableColNames = Flatten[Join[{{"DescribeCol","Ampersands"}},Table[{MmaColNames[[i]],"Ampersands"},{i,Length[MmaColNames]}]]];
  MmaTableColNames[[-1]]="EndOfLines";
  MmaTableRows = Transpose[ToExpression[MmaTableColNames]];
(*
  MmaTableLabels = Flatten[Join[{{"","&"}},Table[{LtxColNames[[i]],"&"},{i,Length[MmaColNames]}]]];
  MmaTableLabels[[-1]] = "\\\\";
  PrependTo[MmaTableRows,MmaTableLabels];
*)
  Write[OutFile,
   PaddedForm[
    TableForm[
      MmaTableRows,TableHeadings -> {None, None}
    ]
  , {4, 3}]
  ];
  Close[OutFile];
(*  FilePrint["/Volumes/Data/Work/MPCPerm/public/tables/MPCPermMeansVsTargets.out"];*)
];


SolveFinHoriz[mTargetTolerance_] := Block[{},

(* Solve for successively finer approximations to the shock distribution and grids for the end-of-period assets level *)
(* This is much faster than solving for dense grids and fine approximations all the way from the start, *)
(* and nearly as accurate *)

ConstructLastPeriodToConsumeEverything; (* Need to set up finite horizon solution so \[DoubledPi]MPCPerm[40] will reflect finite horizon MPCP *)

PermGridLength=TranGridLengthSetup=3;SetupShocks;
SolveToToleranceOf[mTargetTolerance];

PermGridLength=TranGridLengthSetup=7;SetupShocks;
SolveToToleranceOf[mTargetTolerance];

PermGridLength=TranGridLengthSetup=21;SetupShocks;
SolveToToleranceOf[mTargetTolerance];

\[DoubledPi]\[Digamma]40YearsLeft = \[DoubledPi]\[Digamma]Fin[40];

ModelIsSolved=True;

];


AllReports := Block[{},Print[MmaMPCPermTable];LtxMPCPermTable;Print[MmaMeansVsTargetsTable];LtxMeansVsTargetsTable];

SolveAndSim[MultiplesOfMinSimSize_,ExperimentName_,mTargetTolerance_,QuitIfTrendSmallerThan_]:=Block[{},
  Print[ExperimentName];
  SolveFinHoriz[mTargetTolerance];
  Print["Finished solving for policy function for "<>ExperimentName];
  MinSimSize=Round[1/\[WeierstrassP]];(* Pop size that would include exactly one zero income person *)
  Simulate[MinSimSize MultiplesOfMinSimSize,100,bStartVal=bTarget];
  (* Judge if there is a trend in the mean over the past 10 years, sim has not settled down; continue simming *)
  TrendConfidenceInterval = {1,1};
  While[(TrendConfidenceInterval[[1]]>0 || TrendConfidenceInterval[[2]] < 0) && Not[(Abs[TrendConfidenceInterval[[1]]] < QuitIfTrendSmallerThan && Abs[TrendConfidenceInterval[[2]]] < QuitIfTrendSmallerThan)], 
    KeepSimulating[10];
    atMeanLast10=Take[atMean,-10];
    FittedTrend=LinearModelFit[Transpose[{Table[i,{i,10}],atMeanLast10}],x,x];
    TrendConfidenceInterval = FittedTrend["ParameterConfidenceIntervals"][[-1]]
  ];
  AddCurrentResultsToReport
];
 

SetupReport := Block[{},
(* Mathematica name in the first position, LaTeX name in the second *)
{MmaNamePos,LtxNamePos}={1,2};

MPCPermTableScalars = {
   {"\[CapitalThorn]\[CapitalGamma]","$\\Thorn_{\\grave{\\Gamma}}$"}
  ,{"\[DoubledPi]\[Digamma]Inf","$\\pi_{\\infty}$"}
  ,{"\[DoubledPi]\[Digamma]40YearsLeft","$\\pi_{T-40}$"}
(*,{"\[CapitalThorn]Rtn","$\\Thorn_R$"}*)
(*,{"\[Sigma]Perm","$\\sigma_{\\psi}$"}*)
(*,{"\[Sigma]Tran","$\\sigma_{\\theta}$"}*)
(*,{"\[CapitalGamma]","$\\Gamma$"}*)
(*,{"aTarget","$\\check{a}$"}*)
};

MPCPermTableMeans = {
   {"ct","$\\bar{c}$"}
  ,{"at","$\\bar{a}$"}
  ,{"\[Kappa]t","$\\bar{\\kappa}$"}
  ,{"\[DoubledPi]t","$\\bar{\\pi}$"}
};

MeansVsTargetsScalars = {
 (*  {"\[CapitalThorn]\[CapitalGamma]","$\\Thorn_{\\grave{\\Gamma}}$"}*)
   {"aTarget","$\\pi_{\\infty}$"}
  ,{"\[Kappa]Target","$\\pi_{T-40}$"}
  ,{"\[DoubledPi]Target","$\\pi_{T-40}$"}
};

MeansVsTargetsMeans = {
   {"at","$\\bar{a}$"}
  ,{"\[Kappa]t","$\\bar{\\kappa}$"}
  ,{"\[DoubledPi]t","$\\bar{\\pi}$"}
};

MPCPermTableFields = {
   {"\[CapitalThorn]\[CapitalGamma]"   ,"$\\Thorn_{\\grave{\\Gamma}}$"}
  ,{"ct","$\\bar{c}$"}
  ,{"at","$\\bar{a}$"}
  ,{"\[Kappa]t","$\\bar{\\kappa}$"}
  ,{"\[DoubledPi]t","$\\bar{\\pi}$"}
  ,{"\[DoubledPi]\[Digamma]Inf","$\\pi_{\\infty}$"}
  ,{"\[DoubledPi]\[Digamma]40YearsLeft","$\\pi_{T-40}$"}
};

MeansVsTargetsFields = {
   {"\[CapitalThorn]\[CapitalGamma]"   ,"$\\Thorn_{\\grave{\\Gamma}}$"}
  ,{"at"     ,"$\\bar{a}$"}
  ,{"aTarget","$\\check{a}$"}
  ,{"\[Kappa]t"     ,"$\\bar{\\kappa}$"}
  ,{"\[Kappa]Target","$\\check{\\kappa}$"}
  ,{"\[DoubledPi]t"     ,"$\\bar{\\pi}$"}
  ,{"\[DoubledPi]Target","$\\check{\\pi}$"}
};

MakeEmptyReport;
];

