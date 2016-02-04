(* ::Package:: *)

(*
Generate figure showing convergence of consumption rules as period $T$ recedes

Must run cFuncsConvergeSolve.m before running this
*)



cFuncsConverge = Show[
      TableOfcFuncs[[1]]
     ,Graphics[Text[" \[LeftArrow] \!\(c\_\(T\)(\[ScriptM])\)= 45 Degree Line",{6.5,6.5},{-1,0}]]
     ,TableOfcFuncs[[2]]
     ,Graphics[Text[" \!\(c\_\(T-1\)(\[ScriptM])\)",{mPlotLimit,\[ScriptC][mPlotLimit,1]},{-1,0}]]
     ,TableOfcFuncs[[6]]
     ,Graphics[Text[" \!\(c\_\(T-5\)(\[ScriptM])\)",{mPlotLimit,\[ScriptC][mPlotLimit,5]},{-1,-.25}]]
     ,TableOfcFuncs[[11]]
     ,Graphics[Text[" \!\(\*SubscriptBox[\"c\",RowBox[{\"T\", \"-\", \"10\"}]]\)(\[ScriptM])",{mPlotLimit,\[ScriptC][mPlotLimit,10]},{-1,-.45}]]
     ,Last[TableOfcFuncs]
     ,Graphics[Text[" \!\(c(\[ScriptM])\)",{mPlotLimit,\[ScriptC][mPlotLimit,100]},{-1,.25}]]
     ,DisplayFunction->$DisplayFunction
     ,AxesLabel->{"\[ScriptM]","c"}
     ,PlotRange->{{0,mPlotLimit+2.5},{0,7}}
     ,Ticks->None
];

If[SaveFigs == True
   ,ExportFigs["cFuncsConverge"]
   ,Show[cFuncsConverge]
];
