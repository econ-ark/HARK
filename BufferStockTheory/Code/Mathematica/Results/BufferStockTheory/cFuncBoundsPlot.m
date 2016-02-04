(* ::Package:: *)

mPlotMin=0;
mPlotMax=25;
cPlot = Plot[\[ScriptC][mPlot],{mPlot,0,mPlotMax},DisplayFunction->Identity,PlotRange->All];
cBelowPlot = Plot[ mPlot \[Kappa]MinInf,{mPlot,mPlotMin,mPlotMax},PlotStyle->{Thickness[0.005],Black},DisplayFunction->Identity,PlotRange->All];
cAbovePlot = Plot[ mPlot \[Kappa]MaxInf,{mPlot,mPlotMin,mPlotMax},PlotStyle->{Dashing[{0.005}],Black},DisplayFunction->Identity,PlotRange->All];
cMaxPlot = Plot[\[ScriptC]TopBoundInf[mPlot],{mPlot,mPlotMin,mPlotMax},PlotStyle->{Thickness[0.005],Black},DisplayFunction->Identity,PlotRange->All];
\[ScriptC]\[Digamma]InfPlot = Plot[\[ScriptC]\[Digamma]Inf[mPlot],{mPlot,mPlotMin,mPlotMax},PlotStyle->{Dashing[{0.005}],Black},DisplayFunction->Identity,PlotRange->All];
cFuncBounds = Show[cPlot,cBelowPlot,cAbovePlot,\[ScriptC]\[Digamma]InfPlot,cMaxPlot,cBelowPlot
  ,Graphics[
    Text[Style["\!\(\*OverscriptBox[\"c\", \"_\"]\)(\[ScriptM])=(\[ScriptM]-1+\[ScriptH])\!\(\*StyleBox[\"\[Kappa]\",\nFontVariations->{\"Underline\"->True}]\)",CharacterEncoding->"WindowsANSI"]
                 ,{0.09 mPlotMax,0.92 \[ScriptC]\[Digamma]Inf[0.020 mPlotMax]},{-1, 0}]]
  ,Graphics[
    Text["\[UpperLeftArrow]"     ,{0.06 mPlotMax,0.92 \[ScriptC]\[Digamma]Inf[0.020 mPlotMax]},{-1,-1}
     ]
   ]
,Graphics[Text[Style["\[DownArrow]",{0.02 mPlotMax,1.10 \[ScriptC]\[Digamma]Inf[0.020 mPlotMax]},CharacterEncoding->"WindowsANSI"],{-1,1}]]
,Graphics[Text[Style[" \!\(\*
StyleBox[\"\[LongLeftArrow]\",\nFontSize->14]\) c(\[ScriptM])",CharacterEncoding->"WindowsANSI"],{0.1 mPlotMax,\[ScriptC][0.1 mPlotMax]},{-1,0}]]
,Graphics[Text[Style[
"\[LongLeftArrow] Upper Bound = Min[\!\(\*OverscriptBox[OverscriptBox[\"c\", \"_\"], \"_\"]\)(\[ScriptM]),\!\(\*OverscriptBox[\"c\", \"_\"]\)(\[ScriptM])]",CharacterEncoding->"WindowsANSI"],{0.35 mPlotMax,0.98\[ScriptC]\[Digamma]Inf[0.35 mPlotMax]},{-1,0}]]
,Graphics[Text[Style[
" \[LongLeftArrow] \!\(\*OverscriptBox[OverscriptBox[\"c\", \"_\"], \"_\"]\)(\[ScriptM])=\!\(\*OverscriptBox[\"\[Kappa]\", \"_\"]\)\[ScriptM] = (1-\!\(\*SuperscriptBox[\"\[WeierstrassP]\", 
RowBox[{\"1\", \"/\", \"\[Rho]\"}]]\)\!\(\*SubscriptBox[\"\[CapitalThorn]\", \"R\"]\))\[ScriptM]"
,CharacterEncoding->"WindowsANSI"]
,{0.23 mPlotMax,0.23 mPlotMax \[Kappa]MaxInf },{-1,0}]]
,Graphics[Text[Style["\!\(\*UnderscriptBox[\"c\", \"_\"]\)(\[ScriptM])=(1-\!\(\*SubscriptBox[\"\[CapitalThorn]\", \"R\"]\))\[ScriptM]=\!\(\*
StyleBox[\"\[Kappa]\",\nFontVariations->{\"Underline\"->True}]\)\!\(\*SubscriptBox[\"m\", 
StyleBox[\"\[LowerRightArrow]\",\nFontSize->14]]\)",CharacterEncoding->"WindowsANSI"],{0.8 mPlotMax,0.8 mPlotMax \[Kappa]MinInf},{1,-1}]]
,DisplayFunction->$DisplayFunction
,PlotRange->{{0,mPlotMax},{0,1.12 \[ScriptC]\[Digamma]Inf[mPlotMax]}}
,AxesLabel->{"\[ScriptM]","\[ScriptC]"}
,ImageSize->HalfPageSize
,Ticks->None];

If[SaveFigs==True
	,ExportFigs["cFuncBounds"]
    ,Print[Show[cFuncBounds]]
];

