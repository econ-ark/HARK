(* ::Package:: *)

StableLoci = Plot[{\[ScriptC]EDelEqZero[\[ScriptM]],\[ScriptM]EDelEqZero[\[ScriptM]]},{\[ScriptM],0,\[ScriptM]Max}];
TractableBufferStockPhaseDiag=Show[\[ScriptC]LowerPlot
,StableLoci
,Graphics[Text["\!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptC]\), \(\[ScriptT] + 1\), \(e\)]\)=0 \[LongRightArrow] ",{\[ScriptM]E 1.25`,\[ScriptC]EDelEqZero[\[ScriptM]E 1.25`]},{1,0}]]
,Graphics[Text["\!\(\*SubsuperscriptBox[\(\[CapitalDelta]\[ScriptM]\), \(\[ScriptT] + 1\), \(e\)]\)= 0 \[LowerRightArrow]",{\[ScriptM]E/3,\[ScriptM]EDelEqZero[\[ScriptM]E/3]},{1,-1}]]
,Graphics[Text["\!\(\*SuperscriptBox[\(\[ScriptC]\), \(e\)]\)(\[ScriptM])=Stable Arm \[LongRightArrow] ",{\[ScriptM]E/2,cE[\[ScriptM]E/2]},{1,0}]]
,Graphics[Text["Steady State \[LowerRightArrow] ",{\[ScriptM]E,\[ScriptC]E},{1,-1}]]
,PhaseArrow[{\[ScriptM]E,\[ScriptC]E/2},{\[ScriptM]E+\[ScriptM]E/10,\[ScriptC]E/2}]
,PhaseArrow[{\[ScriptM]E,\[ScriptC]E/2},{\[ScriptM]E,\[ScriptC]E/2-\[ScriptC]E/5}]
,PhaseArrow[{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2},{\[ScriptM]E-\[ScriptM]E/10,\[ScriptC]E+\[ScriptC]E/2}]
,PhaseArrow[{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2},{\[ScriptM]E,\[ScriptC]E+\[ScriptC]E/2+\[ScriptC]E/5}]
,Ticks->None
,AxesLabel->{"\!\(\*SubsuperscriptBox[\(\[ScriptM]\), \(\[ScriptT]\), \(e\)]\)","\!\(\*SubsuperscriptBox[\(\[ScriptC]\), \(\[ScriptT]\), \(e\)]\)"}
,PlotRange->{{0,\[ScriptM]Max},{0,\[ScriptC]E+\[ScriptC]E/2+\[ScriptC]E/5}}];
