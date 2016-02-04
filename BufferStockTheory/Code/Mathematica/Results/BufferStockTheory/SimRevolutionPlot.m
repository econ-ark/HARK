(* ::Package:: *)

cLevtPlotData = Table[{i,cLevtMean[[i+1]]/cLevtMean[[i]]},{i,1,Length[cLevtMean]-1}];
cLevGrowPlot  = ListPlot[cLevtPlotData, PlotRange -> {Automatic, {1.0, 1.16}}, AxesLabel -> {"Time", "\!\(C\_\(t + 1\)/C\_t\)"},DisplayFunction->Identity,ImageSize->HalfPageSize];

SimRevolution = Show[
   cLevGrowPlot
  ,Graphics[Line[{{0,1.03},{60,1.03}}]]
  , Graphics[Text["Revolution \[LowerRightArrow]",{40,1.035},{1,-1}]]
  , AxesOrigin -> {0, 1}
  , Ticks->{Automatic,{{1.0,"1.00"},{1.03,"\[CapitalGamma]"},{1.05,"1.05"},{1.10, "1.10"},{1.15,"1.15"}}}
  ,DisplayFunction->$DisplayFunction
];

If[SaveFigs == True
   ,ExportFigs["SimRevolution"]
   ,Print[Show[SimRevolution]]
];

