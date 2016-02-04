(* ::Package:: *)

{mPlotMin,mPlotMax} = {1.0,1.9};
{cPlotMin,cPlotMax} = {\[ScriptC][mPlotMin],\[ScriptC][mPlotMax]};
{yPlotMin,yPlotMax} = {((R) \[Beta])^(1/\[Rho])-0.01, \[CapitalGamma] + 0.04};

(* Expected consumption growth factor *)
\[DoubleStruckCapitalE]cLevtp1OcLevt[mt_]                := \[DoubleStruckCapitalE]Fromat[\[CapitalGamma]tp1 \[ScriptC][mtp1] &, mt-\[ScriptC][mt]]/\[ScriptC][mt]; 
\[DoubleStruckCapitalE]cLevtp1OcLevt[mt_,TimeToT_] := \[DoubleStruckCapitalE]Fromat[\[CapitalGamma]tp1 \[ScriptC][mtp1,TimeToT-1] &, mt-\[ScriptC][mt,TimeToT]]/\[ScriptC][mt,TimeToT]; 

GrowthPlot = Plot[{
        \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]
	,((R) \[Beta])^(1/\[Rho])
	,\[CapitalGamma]
	}
	,{mPlot,mPlotMin,mPlotMax}
	,PlotRange->{{mPlotMin,mPlotMax+0.2},{yPlotMin,yPlotMax}}
	,AxesOrigin->{mPlotMin,yPlotMin}
    ,PlotStyle->{Black,Black,Black}
	,DisplayFunction->Identity
];


DivBy20[StartPoint_,EndPoint_] := (StartPoint-EndPoint)/20+EndPoint;

If[$VersionNumber < 6,
ArrowListBelow = 
  Table[{Thickness[.003], 
      Arrow[DivBy20[{mPlot - .05, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot - .05]}, {mPlot, 
            \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}], {mPlot, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}, 
        HeadScaling -> Automatic, HeadCenter -> 0, HeadLength -> .01, 
        HeadWidth -> 1.5]}, {mPlot, mTarget -.03 , mPlotMin, - .05}]
        ];

If[$VersionNumber >= 6,
  ArrowListBelow = 
    Table[{Arrowheads[0.03], 
      Arrow[{DivBy20[{mPlot - .05, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot - .05]}, {mPlot, 
            \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}], {mPlot, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}}]}, {mPlot, mTarget -.03 , mPlotMin, - .05}];
];

If[$VersionNumber < 6,
ArrowListAbove = 
  Table[{Thickness[.003], 
      Arrow[DivBy20[{mPlot + 0.1, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot + .1]}, {mPlot, 
            \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}], {mPlot, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}, 
        HeadCenter -> 0, HeadLength -> .01, 
        HeadWidth -> 1.5]}, {mPlot, mTarget + .03, mPlotMax , .08}]
        ];

If[$VersionNumber >= 6,
ArrowListAbove = 
  Table[{Arrowheads[0.03], 
      Arrow[{DivBy20[{mPlot + 0.1, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot + .1]}, {mPlot, 
            \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}], {mPlot, \[DoubleStruckCapitalE]cLevtp1OcLevt[mPlot]}}]}, {mPlot, mTarget + .03, mPlotMax , .08}];
];

cGroTargetFig = Show[
    GrowthPlot
    ,Graphics[{Dashing[{.01}],Black,Line[{{mTarget,cPlotMin},{mTarget,cPlotMax}}]}]
    ,Graphics[Text[Style["\[CapitalThorn]\[Congruent]\!\(\((R\[Beta])\)\^\(1/\[Rho]\)\)",CharacterEncoding->"WindowsANSI"],{mPlotMax,((R) \[Beta])^(1/\[Rho])},{-1,0}]]
    ,Graphics[Text[Style["\[CapitalGamma]",CharacterEncoding->"WindowsANSI"],{mPlotMax,\[CapitalGamma]},{-1,0}]]
    ,Graphics[Text[Style[" \!\(\[DoubleStruckCapitalE]\_t[\(c\)\_\(t + 1\)/c\_t]\)",Bold,CharacterEncoding->"WindowsANSI"],{mPlotMax,\[DoubleStruckCapitalE]cLevtp1OcLevt[mPlotMax]},{-1,0}]]
    ,Graphics[ArrowListBelow]
    ,Graphics[ArrowListAbove]
    ,ImageSize -> HalfPageSize
,PlotRange->{{mPlotMin,mPlotMax+0.2},{yPlotMin,yPlotMax}}
,Ticks->{{{mTarget,Text[Style["\!\(\*OverscriptBox[\"m\", \"\[Vee]\"]\)",Italic,CharacterEncoding->"WindowsANSI"]]}
,{mTarget,Text[Style["",Italic,CharacterEncoding->"WindowsANSI"]]}},None}
,AxesLabel->{"\!\(m\_t\)","Growth"}
,DisplayFunction->$DisplayFunction
];

If[SaveFigs == True
   ,ExportFigs["cGroTargetFig"]
   ,Print[Show[cGroTargetFig]]
];
