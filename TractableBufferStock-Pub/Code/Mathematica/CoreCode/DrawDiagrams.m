(* ::Package:: *)

DrawDiagrams[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_] := 
Block[{},
FindStableArm;
\[ScriptC]Plot=Plot[cE[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot}];
\[ScriptC]PlotLabel=Show[Graphics[Text["\!\(\*SuperscriptBox[\(c\), \(e\)]\)(\[ScriptM]) \[LongRightArrow]  ",{(9 \[ScriptM]MaxPlot)/10,cE[(9 \[ScriptM]MaxPlot)/10]},{1,0}]]];
StableArmLabel=Show[Graphics[Text["Stable Arm \[LongRightArrow] ",{(9 \[ScriptM]MaxPlot)/10,cE[(9 \[ScriptM]MaxPlot)/10]},{1,0}]]];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot}];
Degree45 = Plot[\[ScriptM],{\[ScriptM],-Severance/\[ScriptCapitalR],cE[\[ScriptM]MaxMaxPlot]}];
StableLocus\[ScriptC] = Show[Plot[\[ScriptC]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotRange->{Automatic,{Automatic,\[ScriptC]MaxPlot}}]];
StableLocus\[ScriptC]Label=Show[Graphics[Text["\!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptC]\), \( \), \(e\)]\)=0 \[LongRightArrow] ",{\[ScriptM]E 1.25`,\[ScriptC]EDelEqZero[\[ScriptM]E 1.25`]},{1,0}]]];
StableLocus\[ScriptM] = Show[Plot[\[ScriptM]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot}]];
StableLocus\[ScriptM]Label=Show[Graphics[Text["\!\(\*SubscriptBox[\(\), 
StyleBox[\" \",\nFontSize->10]]\) \!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptM]\), \( \), \(e\)]\)= 0 \[UpperRightArrow]",{0.8 \[ScriptM]MaxPlot,\[ScriptM]EDelEqZero[0.8 \[ScriptM]MaxPlot]},{-1,1}]]
];
SteadyState = Show[Graphics[Text["SS \[LowerRightArrow] ",{\[ScriptM]E,\[ScriptC]E},{1,-1}]]];
StableLoci = Show[StableLocus\[ScriptC],StableLocus\[ScriptC]Label,StableLocus\[ScriptM],StableLocus\[ScriptM]Label,SteadyState];
PhaseArrows=Show[
 PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E+(\[ScriptM]MaxMaxPlot-(-Severance))/20,0.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E/2-\[ScriptC]E/5}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E-(\[ScriptM]MaxMaxPlot-(-Severance))/20,1.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2+\[ScriptC]E/5}]
];
\[ScriptM]EDelEqZeroLine=Show[Graphics[{Dashed,Line[{{1.,0.},{1.,\[ScriptC]MaxPlot}}]}]];
\[ScriptM]EDelEqZeroPivot=Show[ListPlot[{{1,\[ScriptM]EDelEqZero[1.]}}]];
\[ScriptM]EDelEqZeroPivotLabel=Show[Graphics[Text["\[LowerLeftArrow] Pivot at {1.,1.}",{1,1},{-1,-1.5}]]];
TractableBufferStockPhaseDiagOrig=Show[
 StableLoci
,StableArmLabel
,PhaseArrows
(*,Ticks->None*)
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \( \), \(e\)]\)"}
,AxesOrigin->{-Severance/\[ScriptCapitalR],0.}
,PlotRange->{{-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},{0,\[ScriptC]MaxPlot}}];
AllFigs={\[ScriptC]Plot,\[ScriptC]PlotLabel,StableArmLabel,cEPFPlot,Degree45,StableLocus\[ScriptC],StableLocus\[ScriptC]Label,StableLocus\[ScriptM],StableLocus\[ScriptM]Label,SteadyState,StableLoci,PhaseArrows
,\[ScriptM]EDelEqZeroPivot,\[ScriptM]EDelEqZeroPivotLabel,TractableBufferStockPhaseDiagOrig}];


\[ScriptM]TargetDiagram[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_] := Block[{},
DrawDiagrams[\[ScriptM]MaxPlot,\[ScriptM]MaxMaxPlot,\[ScriptC]MaxPlot];
TractableBufferStockPhaseDiag=Show[
StableLocus\[ScriptM]
,StableLocus\[ScriptM]Label
,\[ScriptM]EDelEqZeroPivot
(*,\[ScriptM]EDelEqZeroPivotLabel*)
(*,\[ScriptC]Plot
,\[ScriptC]PlotLabel*)
,StableLocus\[ScriptC]
,StableLocus\[ScriptC]Label
,SteadyState
(*,Ticks->None*)
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","\!\(\*SubsuperscriptBox[\(c\), \( \), \(e\)]\)"}
,AxesOrigin->{-Severance/\[ScriptCapitalR],0.}
,PlotRange->{{-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},{0,\[ScriptC]MaxPlot}}]];


DrawPhaseDiagram[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_] := Block[{},
DrawDiagrams[\[ScriptM]MaxPlot,\[ScriptM]MaxMaxPlot,\[ScriptC]MaxPlot];
\[ScriptC]PlotDashing = restylePlot[\[ScriptC]Plot,{Dashing[Small]}];
TractableBufferStockPhaseDiag=Show[
 StableLoci
,\[ScriptC]PlotDashing
,StableArmLabel
,PhaseArrows
(*,Ticks->None*)
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \( \), \(e\)]\)"}
,AxesOrigin->{-Severance/\[ScriptCapitalR],0.}
,PlotRange->{{-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},{0,\[ScriptC]MaxPlot}}]
];



(* restylePlot is from http://mathematica.stackexchange.com/questions/17250/is-it-possible-to-change-the-color-of-plot-in-show *)
restylePlot[plot_Graphics, styles_List, op : OptionsPattern[Graphics]] :=
 Module[{x = styles}, Show[
   MapAt[# /. {__, ln__Line} :> {Directive @ Last[x = RotateLeft@x], ln} &, plot, 1],
   op
 ]];


DrawPhaseDiagramOld[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_] := DrawPhaseDiagramOld[\[ScriptM]MaxPlot,\[ScriptM]MaxMaxPlot,\[ScriptC]MaxPlot,Black];
DrawPhaseDiagramOld[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_,Color_]:= Block[{},FindStableArm;
\[ScriptC]Plot=Plot[cE[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->{Color,Dashing[{.01}]}];
StableArm = Show[\[ScriptC]Plot
,Graphics[Text["Stable Arm \[LongRightArrow] ",{(9 \[ScriptM]MaxPlot)/10,cE[(9 \[ScriptM]MaxPlot)/10]},{1,0}]]
];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->{Color,Dashing[{.02}]}];
Degree45 = Plot[\[ScriptM],{\[ScriptM],-Severance/\[ScriptCapitalR],cE[\[ScriptM]MaxMaxPlot]},PlotStyle->Dashing[{.01}]];
cFuncPlotBase=cFuncPlot=Plot[cE[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxMaxPlot},PlotStyle->Color];
StableLocus\[ScriptC] = Show[Plot[\[ScriptC]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot}
,PlotStyle->Color]
,Graphics[Text["\!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptC]\), \( \), \(e\)]\)=0 \[LongRightArrow] ",{\[ScriptM]E 1.25`,\[ScriptC]EDelEqZero[\[ScriptM]E 1.25`]},{1,0}]]
];
StableLocus\[ScriptM] = Show[Plot[\[ScriptM]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->Color]
,Graphics[Text["\!\(\*SubscriptBox[\(\), 
StyleBox[\"\[LowerLeftArrow]\",\nFontSize->10]]\) \!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptM]\), \( \), \(e\)]\)= 0",{1,\[ScriptM]EDelEqZero[1]},{-1,-1}]]
];
SteadyState = Show[Graphics[Text["SS \[LowerRightArrow] ",{\[ScriptM]E,\[ScriptC]E},{1,-1}]]];
StableLoci = Show[StableLocus\[ScriptC],StableLocus\[ScriptM],SteadyState];
PhaseArrows=Show[
 PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E+(\[ScriptM]MaxMaxPlot-(-Severance))/20,0.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E/2-\[ScriptC]E/5}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E-(\[ScriptM]MaxMaxPlot-(-Severance))/20,1.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2+\[ScriptC]E/5}]
];
TractableBufferStockPhaseDiag=Show[
 StableLoci
,StableArm
,PhaseArrows
(*,Ticks->None*)
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \( \), \(e\)]\)"}
,AxesOrigin->{-Severance/\[ScriptCapitalR],0.}
,PlotRange->{{-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},{0,\[ScriptC]MaxPlot}}];
TractableBufferStockPhaseDiag];


\[ScriptM]TargetDiagramOld[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_] := \[ScriptM]TargetDiagramOld[\[ScriptM]MaxPlot,\[ScriptM]MaxMaxPlot,\[ScriptC]MaxPlot,Black];
\[ScriptM]TargetDiagramOld[\[ScriptM]MaxPlot_,\[ScriptM]MaxMaxPlot_,\[ScriptC]MaxPlot_,Color_]:= Block[{},FindStableArm;
\[ScriptC]Plot=Plot[cE[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->{Color(*,Dashing[{.01}]*)}];
StableArm = Show[\[ScriptC]Plot
,Graphics[Text["\!\(\*SuperscriptBox[\(\[ScriptC]\), \(\[ScriptE]\)]\)(\[ScriptM]) \[LongRightArrow] ",{(9 \[ScriptM]MaxPlot)/10,cE[(9 \[ScriptM]MaxPlot)/10]},{1,0}]]
];
cEPFPlot = Plot[cEPF[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->{Color(*,Dashing[{.02}]*)}];
Degree45 = Plot[\[ScriptM],{\[ScriptM],-Severance/\[ScriptCapitalR],cE[\[ScriptM]MaxMaxPlot]},PlotStyle->Dashing[{.01}]];
cFuncPlotBase=cFuncPlot=Plot[cE[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxMaxPlot},PlotStyle->Color];
StableLocus\[ScriptC] = Show[Plot[\[ScriptC]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot}
,PlotStyle->Color]
,Graphics[Text["\!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptC]\), \( \), \(e\)]\)=0 \[LongRightArrow] ",{\[ScriptM]E 1.25`,\[ScriptC]EDelEqZero[\[ScriptM]E 1.25`]},{1,0}]]
];
StableLocus\[ScriptM] = Show[Plot[\[ScriptM]EDelEqZero[\[ScriptM]],{\[ScriptM],-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},PlotStyle->Color]
,Graphics[Text["\!\(\*SubscriptBox[\(\), 
StyleBox[\"\[LowerLeftArrow]\",\nFontSize->10]]\) \!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptM]\), \( \), \(e\)]\)= 0",{1,\[ScriptM]EDelEqZero[1]},{-1,-1}]]
];
SteadyState = Show[Graphics[Text["SS \[LowerRightArrow] ",{\[ScriptM]E,\[ScriptC]E},{1,-1}]]];
StableLoci = Show[(*StableLocus\[ScriptC],*)StableLocus\[ScriptM],SteadyState];
PhaseArrows=Show[
 PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E+(\[ScriptM]MaxMaxPlot-(-Severance))/20,0.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,0.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E/2-\[ScriptC]E/5}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E-(\[ScriptM]MaxMaxPlot-(-Severance))/20,1.5 \[ScriptC]E}]
,PhaseArrow[{\[ScriptM]E,1.5 \[ScriptC]E},{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2+\[ScriptC]E/5}]
];
TractableBufferStockPhaseDiag=Show[
 StableLoci
,StableArm
(*,PhaseArrows*)
(*,Ticks->None*)
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","\!\(\*SubsuperscriptBox[\(c\), \( \), \(e\)]\)"}
,AxesOrigin->{-Severance/\[ScriptCapitalR],0.}
,PlotRange->{{-Severance/\[ScriptCapitalR],\[ScriptM]MaxPlot},{0,\[ScriptC]MaxPlot}}];
TractableBufferStockPhaseDiag];


DrawGrowthDiagram[\[ScriptM]MinPlot_,\[ScriptM]MaxPlot_,cGroMaxPlot_]:=Block[{},FindStableArm;
HorizAxis=-0.08`;
CGroPlot=Plot[{Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]]]}
,{\[ScriptM],\[ScriptM]MinPlot,\[ScriptM]MaxPlot}
(*,Ticks->{{{\[ScriptM]E,"\[ScriptM]^e"}}
,{{\[GothicG]+\[Mho],"\[Gamma]"},{((r)-\[CurlyTheta])/\[Rho],"\[Rho]^-1(r-\[CurlyTheta])"}}}*)
,PlotStyle->Black];
BufferFigPlot=Show[CGroPlot
,Graphics[{Dashing[{0.005`,0.025`}],Line[{{\[ScriptM]E,HorizAxis},{\[ScriptM]E,cGroMaxPlot}}]}]
,Graphics[{Dashing[{}],Line[{{0,\[GothicG]+\[Mho]},{\[ScriptM]MaxPlot,\[GothicG]+\[Mho]}}]}]
,Graphics[{Dashing[{}],Line[{{0,((r)-\[CurlyTheta])/\[Rho]},{\[ScriptM]MaxPlot,((r)-\[CurlyTheta])/\[Rho]}}]}
,PlotRange->{{0,\[ScriptM]MaxPlot},{HorizAxis,cGroMaxPlot}}
]
];
TractableBufferStockGrowthFig=Show[BufferFigPlot
,Graphics[Text[Style["\!\(\*
StyleBox[\" \",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\"\[LongLeftArrow]\",\nFontWeight->\"Plain\"]\)\!\(\*
StyleBox[\" \",\nFontWeight->\"Plain\"]\)\[CapitalDelta] log  c\!\(\*SubsuperscriptBox[
StyleBox[\".\",\nFontSize->1], \(\[ScriptT] + 1\), \(e\)]\)(\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\))",Plain],{\[ScriptM]E/2,Log[\[ScriptCapitalC]tp1O\[ScriptCapitalC]t[\[ScriptM]E/2]]},{-1,0}]]
,Axes->{Automatic,Automatic}
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \( \), \(e\)]\)","Growth"}
,AxesOrigin->{((r)-\[CurlyTheta])/\[Rho]-0.1,HorizAxis}
,PlotRange->{{0,\[ScriptM]MaxPlot},{HorizAxis,cGroMaxPlot}}]
];
