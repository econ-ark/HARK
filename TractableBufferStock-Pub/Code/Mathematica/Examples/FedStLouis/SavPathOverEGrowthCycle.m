(* ::Package:: *)

(*
This file calculate the saving path of optimistic period (2002-2007) with one-time increase of growth expectation and an abrupt reversal of the expectation at 2008.
*)



Target\[ScriptB] = -2.6;
Initial\[ScriptB] = -2;
\[GothicG]=\[GothicG]Base;
Initial\[GothicG] = \[GothicG];
Initial\[ScriptCapitalR] = \[ScriptCapitalR];
PeriodGrowthChange = 6; (* Starting from 2001, but only changes in six periods: 2002-2007*)
PeriodGrowthReversal = 5;
\[ScriptM]Path = Table[0,{i,1+PeriodGrowthChange+PeriodGrowthReversal}];
\[ScriptC]Path = Table[0,{i,1+PeriodGrowthChange+PeriodGrowthReversal}];
\[CurlyTheta] = \[CurlyTheta]Impatient;
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;

(* Calculate the expected growth expansion parameter. *)
AbruptChange = 0;
GrowthChange\[ScriptB] = Initial\[ScriptB];
While[GrowthChange\[ScriptB] > Target\[ScriptB],
\[GothicG]  = Initial\[GothicG];
AbruptChange = AbruptChange + 0.0003001; (* The increment is slightly different from 0.001 to avoid some subtle issues when \[GothicG]=r. *)
\[GothicG] = \[GothicG] + AbruptChange;
FindStableArm;
Do[\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*Initial\[ScriptCapitalR]+1-SoiTax; (*Notice that the budget constraint is still under influence of ``actual'' growth rate. *)
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]]
,{i,PeriodGrowthChange}];
GrowthChange\[ScriptB] = \[ScriptM]Path[[PeriodGrowthChange+1]]-(1-SoiTax)
];

(* Calculate consumption and wealth path after change of growth expectation. *)
\[GothicG] = Initial\[GothicG];
FindStableArm;
Do[\[ScriptM]Path[[PeriodGrowthChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodGrowthChange+i]] - \[ScriptC]Path[[PeriodGrowthChange+i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodGrowthChange+i+1]] = cE[\[ScriptM]Path[[PeriodGrowthChange+i+1]]];
,{i,PeriodGrowthReversal}];

(* We get the path for impatient people. *)
\[ScriptC]PathImpatient = \[ScriptC]Path;
\[ScriptM]PathImpatient = \[ScriptM]Path;
PrependTo[\[ScriptC]PathImpatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathImpatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathImpatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathImpatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathImpatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathImpatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathImpatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathImpatient,\[ScriptM]Path[[1]]];

(* Now given AbruptChange, we obtain path for patient people. *)
\[CurlyTheta] = \[CurlyTheta]Patient;
\[GothicG] = Initial\[GothicG];
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;
\[GothicG] = \[GothicG] + AbruptChange;
FindStableArm;
Do[\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]];
,{i,PeriodGrowthChange}];

\[GothicG] = Initial\[GothicG];
FindStableArm;
Do[\[ScriptM]Path[[PeriodGrowthChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodGrowthChange+i]] - \[ScriptC]Path[[PeriodGrowthChange+i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodGrowthChange+i+1]] = cE[\[ScriptM]Path[[PeriodGrowthChange+i+1]]]
,{i,PeriodGrowthReversal}];

\[ScriptC]PathPatient = \[ScriptC]Path;
\[ScriptM]PathPatient = \[ScriptM]Path;
PrependTo[\[ScriptC]PathPatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathPatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathPatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathPatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathPatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathPatient,\[ScriptM]Path[[1]]];
PrependTo[\[ScriptC]PathPatient,\[ScriptC]Path[[1]]];
PrependTo[\[ScriptM]PathPatient,\[ScriptM]Path[[1]]];

(* Calculate the saving dynamics of patient group, impatient group and overall. *)
savRatePathPatient=((\[ScriptM]PathPatient-(1-SoiTax))(R-1)+(1-SoiTax)-\[ScriptC]PathPatient)/((\[ScriptM]PathPatient-(1-SoiTax))(R-1)+(1-SoiTax));
savRatePathImpatient=((\[ScriptM]PathImpatient-(1-SoiTax))(R-1)+(1-SoiTax)-\[ScriptC]PathImpatient)/((\[ScriptM]PathImpatient-(1-SoiTax))(R-1)+(1-SoiTax));
savRatePathAll = (((\[ScriptM]PathPatient-(1-SoiTax))(R-1)+(1-SoiTax)-\[ScriptC]PathPatient)+((\[ScriptM]PathImpatient-(1-SoiTax))(R-1)+(1-SoiTax)-\[ScriptC]PathImpatient))/(((\[ScriptM]PathPatient-(1-SoiTax))(R-1)+(1-SoiTax))+((\[ScriptM]PathImpatient-(1-SoiTax))(R-1)+(1-SoiTax)));
timePath = Table[i,{i,1997,2012}];
Needs["PlotLegends`"];

SavPathOverEGrowthCyclePat = ListPlot[{Transpose[{timePath,savRatePathPatient}]}
,PlotStyle->{Black},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEGrowthCycleImp = ListPlot[{Transpose[{timePath,savRatePathImpatient}]}
,PlotStyle->{Red},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEGrowthCycleAll = ListLinePlot[Transpose[{timePath,savRatePathAll}]
,PlotStyle->{Green},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEGrowthCycle= ShowLegend[Show[SavPathOverEGrowthCyclePat,SavPathOverEGrowthCycleImp],{
{{Graphics[{Red,Point[{{0,0},{1,0}}]}],"Less patient group"}
,{Graphics[{Black,Point[{{0,0},{1,0}}]}],"More patient group"}
(*,{Graphics[{Gray,Point[{{0,0},{1,0}}]}],"Overall"}*)
},
LegendPosition->{0.5,-0.6},LegendSize->{0.8,0.25}}];



Export[FigsDir<>"/SavPathOverEGrowthCycle.pdf",SavPathOverEGrowthCycle];
Export[FigsDir<>"/SavPathOverEGrowthCyclePat.pdf",SavPathOverEGrowthCyclePat];
Export[FigsDir<>"/SavPathOverEGrowthCycleImp.pdf",SavPathOverEGrowthCycleImp];
