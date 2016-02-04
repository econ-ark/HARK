(* ::Package:: *)

(*
This file calculate the discount rate which gives us the specified target wealth.
*)

SetSystemOptions["EvaluateNumericalFunctionArgument" -> False];
(* Prevents problematic efforts to evaluate the arguments of numerical functions *)


CalDiscRate[Target\[ScriptB]_]:=Block[{temp\[CurlyTheta]Upper,temp\[CurlyTheta]Lower,tempCounter,temp\[CurlyTheta]},
(* I specify the upper bound and lower bound of discount rate to be 1(1/1.019) and 2/3(1/1.5), which is a reasonably wide range of discount rate.*) 
temp\[CurlyTheta]Upper=0.5;
temp\[CurlyTheta]Lower=0.019; (* Here I specify lower bound to be 0.019 to avoid \timeRate hit some boundary conditions. It can be changed when parameters values are changed. *)
\[CurlyTheta]=temp\[CurlyTheta]Upper;
If[Target\[ScriptB]<\[ScriptB]E,Abort[]];
\[CurlyTheta]=temp\[CurlyTheta]Lower;
If[Target\[ScriptB]>\[ScriptB]E,Abort[]];
Do[\[CurlyTheta]=(temp\[CurlyTheta]Upper+temp\[CurlyTheta]Lower)/2;
If[\[ScriptB]E<Target\[ScriptB],temp\[CurlyTheta]Upper=\[CurlyTheta],temp\[CurlyTheta]Lower=\[CurlyTheta]],{tempCounter,20}];
temp\[CurlyTheta]=\[CurlyTheta];
Return[temp\[CurlyTheta]];
];

