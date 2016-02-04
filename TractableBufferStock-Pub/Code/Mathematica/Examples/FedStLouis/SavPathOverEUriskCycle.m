(* ::Package:: *)

(*
This file calculate the saving path during the period when there is an one-time decrease of unemployment expectation at 2002 and an abrupt reversal at 2008.
*)



Target\[ScriptB] = -2.6;
Initial\[ScriptB] = -2;
\[Mho]=\[Mho]Base;
Initial\[Mho] = \[Mho];
Initial\[ScriptCapitalR] = \[ScriptCapitalR];
PeriodUrateChange = 6; (* Starting from 2001, but only changes in six periods: 2002-2007*)
PeriodUrateReversal = 5;
\[ScriptM]Path = Table[0,{i,1+PeriodUrateChange+PeriodUrateReversal}];
\[ScriptC]Path = Table[0,{i,1+PeriodUrateChange+PeriodUrateReversal}];
\[CurlyTheta] = \[CurlyTheta]Impatient;
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;
\[GothicG]Base=\[GothicG];

(* Calculate the expected umployment rate shrinkage parameter. *)
AbruptChange = 0;
GrowthChange\[ScriptB] = Initial\[ScriptB];
While[GrowthChange\[ScriptB] > Target\[ScriptB],
\[Mho]  = Initial\[Mho];
AbruptChange = AbruptChange + 0.0001; 
\[Mho] = Initial\[Mho] - AbruptChange;
\[GothicG] = \[GothicG]Base + AbruptChange;
FindStableArm;
Do[\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*Initial\[ScriptCapitalR]+1-SoiTax; (*Notice that the budget constraint is still under influence of ``actual'' growth rate. *)
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]]
,{i,PeriodUrateChange}];
GrowthChange\[ScriptB] = \[ScriptM]Path[[PeriodUrateChange+1]]-(1-SoiTax)
];

(* Calculate consumption and wealth path after change of unemployment rate expectation. *)
\[Mho] = Initial\[Mho];
FindStableArm;
Do[\[ScriptM]Path[[PeriodUrateChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodUrateChange+i]] - \[ScriptC]Path[[PeriodUrateChange+i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodUrateChange+i+1]] = cE[\[ScriptM]Path[[PeriodUrateChange+i+1]]];
,{i,PeriodUrateReversal}];

(* Get the pre-experiment path for impatient people. *)
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
\[Mho] = Initial\[Mho];
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;
\[Mho] = Initial\[Mho]-AbruptChange; 
\[GothicG] = \[GothicG]Base + AbruptChange; (* Notice that we increase growth expectation by the same amount in order to keep growth rate of labor income of employed consumers unchanged. *)
FindStableArm;
Do[\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]];
,{i,PeriodUrateChange}];


\[Mho] = Initial\[Mho];
\[GothicG] = \[GothicG]Base;
FindStableArm;
Do[\[ScriptM]Path[[PeriodUrateChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodUrateChange+i]] - \[ScriptC]Path[[PeriodUrateChange+i]])*Initial\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodUrateChange+i+1]] = cE[\[ScriptM]Path[[PeriodUrateChange+i+1]]]
,{i,PeriodUrateReversal}];

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

SavPathOverEUriskCyclePat = ListPlot[{Transpose[{timePath,savRatePathPatient}]
}
,PlotStyle->{Black},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEUriskCycleImp = ListPlot[{
Transpose[{timePath,savRatePathImpatient}]}
,PlotStyle->{Red},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEUriskCycleAll = ListLinePlot[{Transpose[{timePath,savRatePathAll}]}
,PlotStyle->{Black},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverEUriskCyclePatImp = Show[SavPathOverEUriskCycleImp,SavPathOverEUriskCyclePat];
SavPathOverEUriskCycle= ShowLegend[Show[SavPathOverEUriskCyclePatImp],{
{{Graphics[{Red,Point[{{0,0},{1,0}}]}],"Less patient group"}
,{Graphics[{Black,Point[{{0,0},{1,0}}]}],"More patient group"}
(*,{Graphics[{Gray,Point[{{0,0},{1,0}}]}],"Overall"}*)
},
LegendPosition->{0.5,-0.6},LegendSize->{0.8,0.25}}];


Export[FigsDir<>"/SavPathOverEUriskCycle.pdf",SavPathOverEUriskCycle];
Export[FigsDir<>"/SavPathOverEUriskCyclePat.pdf",SavPathOverEUriskCyclePat];
Export[FigsDir<>"/SavPathOverEUriskCycleImp.pdf",SavPathOverEUriskCycleImp];
