(* ::Package:: *)

(*
Generate the figure showing the limits of the MPC as wealth goes to zero and infinity
*)

If[Not[ModelIsSolved],<<SolveInfHorizToToleranceAtTarget.m];

MPCLimitsRaw = Plot[{\[Kappa]Min[[-1]],\[Kappa]Max[[-1]]},{m,0,8},DisplayFunction->Identity,PlotStyle->{{Thickness[0.004],Black}}];
MPCLimitsFunc= Plot[{\[Kappa][m]},{m,0,8},DisplayFunction->Identity,PlotStyle->{Thickness[0.001],Black}
,PlotRange->{All,{0.,1.}}];

MPCLimits = Show[
 MPCLimitsRaw,MPCLimitsFunc
,Graphics[
  Text[Style["\[UpperLeftArrow] (1-\!\(\*SuperscriptBox[\"\[WeierstrassP]\", 
RowBox[{\"1\", \"/\", \"\[Rho]\"}]]\)\!\(\*SubscriptBox[\"\[CapitalThorn]\", \"R\"]\)) \[Congruent] \!\(\*OverscriptBox[\"\[Kappa]\", \"_\"]\)",CharacterEncoding->"WindowsANSI"]
      ,{6,0.87}]]
, Graphics[
  Text[Style["  \!\(\*
StyleBox[\"\[Kappa]\",\nFontVariations->{\"Underline\"->True}]\) \[Congruent] (1-\!\(\*SubscriptBox[\"\[CapitalThorn]\", \"R\"]\)) \[LowerRightArrow]",CharacterEncoding->"WindowsANSI"]      ,{0,(1-(((R) \[Beta])^(1/\[Rho]))/(R))+.005},{-1,-1}]]
, Graphics[
  Text[" \[LongLeftArrow] \[Kappa](\[ScriptM]) \[Congruent] c'(\[ScriptM])"      ,{1,\[Kappa][1]},{-1,-1}]]
,AxesLabel->{"\[ScriptM]",""}
,ImageSize -> {72. 6.5,72. 6.5/GoldenRatio}
,DisplayFunction->$DisplayFunction      
,PlotRange->{All,{0.,1.}}
];

If[SaveFigs == True
  ,ExportFigs["MPCLimits"]
  ,Print[Show[MPCLimits]]
];

