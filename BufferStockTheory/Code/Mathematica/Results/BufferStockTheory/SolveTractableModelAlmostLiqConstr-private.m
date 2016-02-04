(* ::Package:: *)

\[CurlyEpsilon] = 0.000000000000001;
\[Mho] = 1/1000000000000;
\[CurlyTheta] = 1/\[Beta] - 1; (* Tractable problem uses time preference rate, not factor *)
\[GothicG] = \[CapitalGamma] (1 - \[Mho]) - 1; 
CheckForGrowthImpatience;
CheckForReturnImpatience;
{ceTargBase, meTargBase} = {ceTarg, meTarg};
ParamsAreSet = True;
\[ScriptM]MaxBound = 2 aGridVecExcBot[[-1]]/meTarg;
FindStableArm;
