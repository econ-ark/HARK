(* ::Package:: *)

(*
This file calculate the saving path over the course of a credit boom (2001-2007) followed by an abrupt reversal (2008-2012)
*)


Target\[ScriptB] = -2.6;
Initial\[ScriptB] = -2;
InitialSev = Severance;
PeriodSevChange = 6; (* Starting from 2001, but only changes in six periods: 2002-2007*)
PeriodSevReversal = 5;
\[ScriptM]Path = Table[0,{i,1+PeriodSevChange+PeriodSevReversal}];
\[ScriptC]Path = Table[0,{i,1+PeriodSevChange+PeriodSevReversal}];
\[CurlyTheta] = \[CurlyTheta]Impatient;
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;

(* Calculate the credit expansion parameter: SmoothChange, and save consumption and wealth path. *)
SmoothChange = 0;
CreditBoom\[ScriptB] = Initial\[ScriptB];
While[CreditBoom\[ScriptB] > Target\[ScriptB],
Severance  = InitialSev;
SmoothChange = SmoothChange + 0.01;
Do[Severance = Severance + SmoothChange;
FindStableArm;
\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]]
,{i,PeriodSevChange}];
CreditBoom\[ScriptB] = \[ScriptM]Path[[PeriodSevChange+1]]-(1-SoiTax)
];

(* Calculate consumption and wealth path after credit reversal. *)
Severance = InitialSev;
FindStableArm;
Do[\[ScriptM]Path[[PeriodSevChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodSevChange+i]] - \[ScriptC]Path[[PeriodSevChange+i]])*\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodSevChange+i+1]] = cE[\[ScriptM]Path[[PeriodSevChange+i+1]]];
,{i,PeriodSevReversal}];

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

(* Now given SmoothChange, we obtain path for patient people. *)
\[CurlyTheta] = \[CurlyTheta]Patient;
Severance = InitialSev;
\[ScriptM]Path[[1]] = \[ScriptM]E;
\[ScriptC]Path[[1]] = \[ScriptC]E;
Do[Severance = Severance + SmoothChange;
FindStableArm;
\[ScriptM]Path[[i+1]] = (1-\[Tau])(\[ScriptM]Path[[i]] - \[ScriptC]Path[[i]])*\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[i+1]] = cE[\[ScriptM]Path[[i+1]]];
,{i,PeriodSevChange}];

Severance = InitialSev;
FindStableArm;
Do[\[ScriptM]Path[[PeriodSevChange+i+1]] = (1-\[Tau])(\[ScriptM]Path[[PeriodSevChange+i]] - \[ScriptC]Path[[PeriodSevChange+i]])*\[ScriptCapitalR]+1-SoiTax;
\[ScriptC]Path[[PeriodSevChange+i+1]] = cE[\[ScriptM]Path[[PeriodSevChange+i+1]]]
,{i,PeriodSevReversal}];

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

SavPathOverECreditCycleSlowPat = ListPlot[{Transpose[{timePath,savRatePathPatient}]
(*,Transpose[{timePath,savRatePathAll}]*)}
,PlotStyle->{Black
},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverECreditCycleSlowImp = ListPlot[{Transpose[{timePath,savRatePathImpatient}]
},PlotStyle->{Red},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverECreditCycleSlowAll = ListLinePlot[{
Transpose[{timePath,savRatePathAll}]},PlotStyle->{Red},
AxesLabel->{"Time","Saving Rate"},
Ticks->{{{2001,"2001"},{2007,"2007"}},{-0.1,0.1}},
PlotRange->{Automatic,PlotRangeVertical},
AxesOrigin->{2000,0}
];

SavPathOverECreditCycleSlow = ShowLegend[Show[SavPathOverECreditCycleSlowImp,SavPathOverECreditCycleSlowPat],{
{{Graphics[{Red,Point[{{0,0},{1,0}}]}],"Less patient group"}
,{Graphics[{Black,Point[{{0,0},{1,0}}]}],"More patient group"}
(*,{Graphics[{Gray,Point[{{0,0},{1,0}}]}],"The Whole Economy"}*)
},
LegendPosition->{0.5,-0.6},LegendSize->{0.8,0.25}}];



Export[FigsDir<>"/SavPathOverECreditCycleSlow.pdf",SavPathOverECreditCycleSlow];
Export[FigsDir<>"/SavPathOverECreditCycleSlowPat.pdf",SavPathOverECreditCycleSlowPat];
Export[FigsDir<>"/SavPathOverECreditCycleSlowImp.pdf",SavPathOverECreditCycleSlowImp];
