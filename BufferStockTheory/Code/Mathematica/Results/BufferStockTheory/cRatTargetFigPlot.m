(* ::Package:: *)

FindTargets[DesiredAccuracy=5];
{cPlotMin,cPlotMax}=\[ScriptC][{\[ScriptM]Min,\[ScriptM]Max}={0.,3.}];
cRatTargetFuncs=Plot[{\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[m],\[ScriptC][m]}
,{m,\[ScriptM]Min,\[ScriptM]Max}
,PlotStyle->{{Black,Thickness[Medium]},{Black,Thickness[Medium]}}
,PlotRange->All];
cRatTargetFig=Show[cRatTargetFuncs
,Graphics[Text[" \[UpperLeftArrow] \!\(\*SubscriptBox[\"\[DoubleStruckCapitalE]\", \"t\"]\)[\!\(\*SubscriptBox[\"\[CapitalDelta]\[ScriptM]\", 
RowBox[{\"t\", \"+\", \"1\"}]]\)]=0",{3 \[ScriptM]Max/4,\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[3 \[ScriptM]Max/4]},{-1,1}]]
,Graphics[Text["  \[LongLeftArrow] c(\!\(\*SubscriptBox[\"\[ScriptM]\", \"t\"]\))",{2 \[ScriptM]Max/3,\[ScriptC][2 \[ScriptM]Max/3]},{-1,0}]]
    ,Graphics[{Dashing[{.01}],Black,Line[{{mTarget,cPlotMin},{mTarget,cPlotMax}}]}]
,Ticks->{{{mTarget,Text[Style["\!\(\*OverscriptBox[\"m\", \"\[Vee]\"]\)",Italic]]},{mTarget,Text[Style["",Italic]]}},None}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
];
If[SaveFigs == True
  ,ExportFigs["cRatTargetFig"];
  ,Print[Show[cRatTargetFig]]
];
