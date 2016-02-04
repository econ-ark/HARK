(* ::Package:: *)

(*
Generate figure showing convergence of consumption rules as period T recedes
*)



Print["Solving for ",LifeLength=100," periods."];
Do[SolveAnotherPeriod,{LifeLength}];

mPlotLimit = 9.5;
mMin = 0;

Print["Constructing TableOfcFuncs."];
TableOfcFuncs = 
  Table[
   Plot[ 
    \[ScriptC][mRat,TimeToT]
    ,{mRat,mMin,mPlotLimit}
    ,ImageSize->HalfPageSize
    ,DisplayFunction->Identity
    ,PlotRange->{{mMin,mPlotLimit+2.5},{0.,\[ScriptC][mPlotLimit+2.5,0]}}
   ] 
  ,{TimeToT,0,LifeLength}
];

