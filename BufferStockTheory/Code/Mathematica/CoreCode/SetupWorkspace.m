(* ::Package:: *)

(*
This file contains various background routines and default settings for plot formats etc
*)

(* %%% *)<<private/ConcaveKnotMake.m;

If[StringTake[$System,3] == "Mac",
  (* then *) ParentDirectoryString = "..";SubDirectoryString = "/";ThisDirectoryString = ".",
  (* else *) ParentDirectoryString = "..";SubDirectoryString = "\\";ThisDirectoryString = "."];

(* Define functions to construct equiprobable discrete approximation to lognormal *)
(*
The key result on which the approximation rests is the solution to the integral 
that calculates the expectation of the value of a lognormally distributed variable z
in the interval from zMin to zMax.  The solution to this can be verified analytically 
by executing the Mathematica command

Integrate[z PDF[LogNormalDistribution[\[Mu],\[Sigma]],z],{z,zMin,zMax},Assumptions->{zMax-zMin>0&&zMax>0&&zMin>0}]

and that solution
-1/2 E^(\[Mu]+\[Sigma]^2/2) (Erf[(\[Mu]+\[Sigma]^2-Log[zMax])/(Sqrt[2] \[Sigma])]-Erf[(\[Mu]+\[Sigma]^2-Log[zMin])/(Sqrt[2] \[Sigma])])

is directly incorporated into the definition of the function below
*)

ClearAll[DiscreteApproxToMeanOneLogNormal,DiscreteApproxToMeanOneLogNormalWithEdges];

DiscreteApproxToMeanOneLogNormalWithEdges[StdDev_,NumOfPoints_] := Block[{\[Mu],\[Sigma]},
   \[Sigma]=StdDev;
   \[Mu]=-(1/2) \[Sigma]^2;  (* This is the value necessary to make the mean in levels = 1 *)
   \[Sharp]Inner = Table[Quantile[LogNormalDistribution[\[Mu],\[Sigma]],(i/NumOfPoints)],{i,NumOfPoints-1}];
   \[Sharp]Outer = Flatten[{{0}, \[Sharp]Inner,{Infinity}}];
   CDFOuter    = Table[CDF[LogNormalDistribution[\[Mu],\[Sigma]],\[Sharp]Outer[[i]]],{i,1,Length[\[Sharp]Outer]}];
   CDFInner    = Most[Rest[CDFOuter]]; (* Removes first and last elements *)
   MeanPointsProb = Table[CDFOuter[[i]]-CDFOuter[[i-1]],{i,2,Length[\[Sharp]Outer]}];
   MeanPointsVals = Table[
     {zMin,zMax}= {\[Sharp]Outer[[i-1]], \[Sharp]Outer[[i]]};
      -(1/2) E^(\[Mu]+\[Sigma]^2/2) (Erf[(\[Mu]+\[Sigma]^2-Log[zMax])/(Sqrt[2] \[Sigma])]-Erf[(\[Mu]+\[Sigma]^2-Log[zMin])/(Sqrt[2] \[Sigma])]) //N
     ,{i,2,Length[\[Sharp]Outer]}]/MeanPointsProb;
   Return[{MeanPointsVals,MeanPointsProb,\[Sharp]Outer,CDFOuter,\[Sharp]Inner,CDFInner}]
];

DiscreteApproxToMeanOneLogNormal[StdDev_,NumOfPoints_] := Take[DiscreteApproxToMeanOneLogNormalWithEdges[StdDev,NumOfPoints],2];
 
FullPageSize  = {72. 8.5  , 72. 8.5/GoldenRatio};
HalfPageSize  = {72. 6.5  , 72. 6.5/GoldenRatio};
ThirdPageSize = {72. 4.5  , 72. 4.5/GoldenRatio};
LandscapeSize = {72. 11.0, 72. 7.5} // N;

SetOptions[Plot          , PlotStyle  ->{Black,Thickness[Medium]}, BaseStyle -> {FontSize -> 14}, ImageSize -> HalfPageSize];
SetOptions[ListPlot      , BaseStyle -> {FontSize -> 14}, PlotStyle->{Black}, ImageSize -> HalfPageSize];
SetOptions[ParametricPlot, ImageSize -> HalfPageSize, BaseStyle -> {FontSize -> 14}];
SetOptions[ListLinePlot  , ImageSize -> HalfPageSize, BaseStyle -> {FontSize -> 14}];
SetOptions[Plot3D        , ImageSize -> HalfPageSize, BaseStyle -> {FontSize -> 14}];

(* Put figures into standardized location directories called ../Figures *)

ExportFigsToDir[FigName_,DirNameForFigs_] := Block[{},
  Print[ToExpression[FigName]];
  Print["Exporting:"<>DirNameForFigs <> SubDirectoryString <> FigName];
  Export[DirNameForFigs <> SubDirectoryString <> FigName <> ".eps", ToExpression[FigName], "EPS"];
  Export[DirNameForFigs <> SubDirectoryString <> FigName <> ".pdf", ToExpression[FigName], "PDF"];
  Export[DirNameForFigs <> SubDirectoryString <> FigName <> ".png", ToExpression[FigName], "PNG",ImageSize->FullPageSize];
(*  Export[DirNameForFigs <> SubDirectoryString          <> FigName <> ".svg", ToExpression[FigName], "SVG"]; *)  (* SVG version causes a crash on Windows based systems ! *)
  If[OpenFigsUsingShell != False, Run["open "<>DirNameForFigs <> SubDirectoryString <> FigName <> ".pdf"]];
];

ExportFigs[FigName_] := ExportFigsToDir[FigName,"../../../../Figures"];  (* Figures directories are assumed to be four levels up *)

AddBracesTo[\[Bullet]_] := Transpose[{\[Bullet]}]; (* Interpolation[] requires first argument to have braces; this puts braces around its argument list *)
