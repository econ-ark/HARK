(* ::Package:: *)

{\[ScriptM]Max,\[ScriptM]MaxMax}={1.5,15} \[ScriptM]E;
\[ScriptC]LowerPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,\[ScriptM]Max},PlotStyle->StableArmStyle];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],0,\[ScriptM]MaxMax},PlotStyle->{Black,Dashing[{.02}],Thickness[Medium]}];
Degree45 = Plot[\[ScriptM],{\[ScriptM],0,cE[\[ScriptM]MaxMax]},PlotStyle->{Black,Dashing[{.01}],Thickness[Medium]}];
cFuncPlotBase=cFuncPlot=Plot[cE[\[ScriptM]],{\[ScriptM],0,\[ScriptM]MaxMax}];
TractableBufferStockcFunc=Show[cFuncPlot,cEPFPlot,Degree45
,Graphics[Text[" \[LongLeftArrow] 45 Degree Line ",{0.8` cE[\[ScriptM]MaxMax],0.8` cE[\[ScriptM]MaxMax]},{-1,0}]]
,Graphics[Text["  \[LongLeftArrow] Consumption Function \!\(\*
StyleBox[\"c\",\nFontWeight->\"Plain\"]\)(\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\))",{\[ScriptM]E/3,cE[\[ScriptM]E/3]},{-1,0}]]
,Graphics[Text[OverBar[Style["\!\(\*
StyleBox[\"c\",\nFontWeight->\"Plain\"]\)",Plain]],{(\[ScriptM]MaxMax 3)/4-10.3,cEPF[(\[ScriptM]MaxMax 3)/4]+0.01},{-1,0}]]
,Graphics[Text[Style["\!\(\*
StyleBox[\"(\",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[SubsuperscriptBox[\"\[ScriptM]\", \"\[ScriptT]\", \"e\"],\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\")\",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\"\[LongRightArrow]\",\nFontWeight->\"Plain\"]\)",Plain],{(\[ScriptM]MaxMax 3)/4-9,cEPF[(\[ScriptM]MaxMax 3)/4]},{-1,0}]]
,Graphics[Text[Style["Perfect Foresight Cons Function",Plain],{(\[ScriptM]MaxMax 3)/4-11,cEPF[(\[ScriptM]MaxMax 3)/4]},{1,0}]]
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \(\[ScriptT]\), \(e\)]\)"}
,Ticks->None];
