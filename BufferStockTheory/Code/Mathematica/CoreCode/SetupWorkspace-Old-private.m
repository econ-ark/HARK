(* ::Package:: *)

(*
This file contains various background routines and default settings for plot formats etc
*)

(* %%% *)<<private/ConcaveKnotMake.m;

If[StringTake[$System,3] == "Mac",
  (* then *) ParentDirectoryString = "..";SubDirectoryString = "/";ThisDirectoryString = ".",
  (* else *) ParentDirectoryString = "..";SubDirectoryString = "\\";ThisDirectoryString = "."];

ClearAll[DiscreteApproxToMeanOneLogNormal];
DiscreteApproxToMeanOneLogNormal[StdDev_,NumOfPoints_] := Block[{z,\[Sigma]=StdDev,\[Mu],EdgePoints,BoundPoints,ProbOfMeanPoints,MeanPoints},
  (* If they have given a standard deviation of zeros, just return a list all of whose elements are 1 *)
  If[Chop[StdDev] == 0,Return[{Table[1,{NumOfPoints}],Table[1/NumOfPoints,{NumOfPoints}]}]] //N;
  \[Mu]=LevelAdjustingParameter = -(1/2) (StdDev)^2;  (* This parameter takes on the value necessary to make the mean in levels = 1 *)
  EdgePoints  = Table[Quantile[LogNormalDistribution[LevelAdjustingParameter,StdDev],(i/NumOfPoints)],{i,NumOfPoints-1}] //N;
  EdgePoints  = Flatten[{{0},EdgePoints,{Infinity}}];
  BoundPoints = Transpose[{Take[EdgePoints,Length[EdgePoints]-1],Take[EdgePoints,{2,Length[EdgePoints]}]}];
  ProbOfMeanPoints = Table[CDF[LogNormalDistribution[LevelAdjustingParameter,StdDev],EdgePoints[[i  ]]]
                          -CDF[LogNormalDistribution[LevelAdjustingParameter,StdDev],EdgePoints[[i-1]]]
                          ,{i,2,Length[EdgePoints]}];
(* %%% *) (* Formula below from conversation with Michael Reiter at 2009 SED; but it's no faster -- probably it's what Mma does itself *)
(* %%% *)(*MeanPoints = Map[E^(\[Mu]+\[Sigma]^2/2) Sqrt[\[Pi]/2] \[Sigma] (Erf[(\[Mu]+\[Sigma]^2-Log[#[[1]]])/(Sqrt[2] \[Sigma])]-Erf[(\[Mu]+\[Sigma]^2-Log[#[[2]]])/(Sqrt[2] \[Sigma])]) &,BoundPoints] / ProbOfMeanPoints //N;*)
  MeanPoints = Table[NIntegrate[z PDF[LogNormalDistribution[LevelAdjustingParameter,StdDev],z],{z,EdgePoints[[i-1]],EdgePoints[[i]]}]
                     ,{i,2,Length[EdgePoints]}] / ProbOfMeanPoints;
Return[{MeanPoints,ProbOfMeanPoints}]
];
   
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
