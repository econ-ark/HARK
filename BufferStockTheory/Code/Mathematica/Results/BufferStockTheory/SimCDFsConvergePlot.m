(* ::Package:: *)

(*
 Make plot showing the distribution for a after 2, 5, 10, and 40 periods
*)

CDFPlots = Flatten[{{},
  Table[
       ParametricPlot[
         {CDFatFuncList[[i]][CDFPoint],CDFPoint}
         ,{CDFPoint,0,0.999}
         ,PlotRange->{{0.,CDFatList[[-1,-1]]},{0,1}}
         ,PlotStyle->{{Black,Thickness[.003]}}
         ,DisplayFunction->Identity
    ]
  ,{i,2,Length[CDFatFuncList]}
  ] (* End Table *)
 }  (* End Flatten *)
];

LastPeriod=Length[CDFPlots];

SimCDFsConverge = Show[
  CDFPlots[[2]]
, CDFPlots[[5]]
, CDFPlots[[11]]
, CDFPlots[[LastPeriod]]
  , Graphics[Text["\!\(\(\[ScriptCapitalF]\_1\%a\)\) \[LongRightArrow]",
  {CDFatFuncList[[2]][0.94], 0.9}, {1, 0}]]
  , Graphics[Text["\!\(\(\[ScriptCapitalF]\_4\%a\)\) \[LongRightArrow]",
  {CDFatFuncList[[5]][0.85], 0.82}, {1, 0}]]
, Graphics[Text[" \[LongLeftArrow] \!\(\(\[ScriptCapitalF]\_10\%a\) \)",
  {CDFatFuncList[[10]][0.8], 0.8}, {-1, 0}]]
, Graphics[Text[" \[LongLeftArrow] \!\(\(\[ScriptCapitalF]\_40\%a\) \) \[TildeTilde] \!\(\(\[ScriptCapitalF]\_\[Infinity]\%a\)\) ",
  {CDFatFuncList[[LastPeriod]][0.4], 0.4}, {-1, 0}]]
(*, DisplayFunction -> $DisplayFunction*)
, AxesLabel->{"\[ScriptA]","\[ScriptCapitalF]"}
, ImageSize  ->  FullPageSize
, AxesOrigin -> {0, 0}
(*, PlotRange -> {{0, CDFatList[[-1, -(Round[Length[CDFatList[[-1]]]/1000]+1)]]}, {0, 1}}*)
, DisplayFunction->$DisplayFunction
, BaseStyle -> {FontSize -> 14}
];

If[SaveFigs == True
   ,ExportFigs["SimCDFsConverge"]
   ,Print[Show[SimCDFsConverge]]
];

