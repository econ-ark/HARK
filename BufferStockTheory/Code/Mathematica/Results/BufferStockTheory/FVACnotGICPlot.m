(* ::Package:: *)


CDToHomeDir;<<cFuncsConvergeSolve.m;
{\[ScriptM]Min,\[ScriptM]Max}={0.,5.};
FVACnotGICFuncs=Plot[{\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[m],\[ScriptC][m]}
,{m,\[ScriptM]Min,\[ScriptM]Max}
,PlotStyle->{{Black,Thickness[Medium]},{Black,Thickness[Medium]}}];
FVACnotGIC=Show[FVACnotGICFuncs
,Graphics[Text["\!\(\*SubscriptBox[\"\[DoubleStruckCapitalE]\", \"t\"]\)[\!\(\*SubscriptBox[\"\[CapitalDelta]\[ScriptM]\", 
RowBox[{\"t\", \"+\", \"1\"}]]\)]=0 \[LongRightArrow]    ",{\[ScriptM]Max/2,\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[\[ScriptM]Max/2]},{1,0}]]
,Graphics[Text["  \[LongLeftArrow] c(\!\(\*SubscriptBox[\"\[ScriptM]\", \"t\"]\))",{\[ScriptM]Max/6,\[ScriptC][\[ScriptM]Max/6]},{-1,0}]]
,Ticks->None
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
];
ExportFigs["FVACnotGIC"];
Show[FVACnotGIC]
