(* ::Package:: *)

(* 
These parameter values are chosen not for realism 
but in order to generate figures that illustrate the qualitative 
features of the model as clearly as possible.  More realistic
parameter values should be used for quantitative exercises (like Carroll
and Jeanne (2009))
*)

\[CurlyEpsilon]=0.0001; 
\[Mho]       =\[Mho]Base       = 0.005; (* From Carroll (1992); though that number was for transitory unemployment shocks *)
\[CurlyTheta]=\[CurlyTheta]Base= 0.10;
 r    = rBase   = 0.03;
\[GothicG]   =\[GothicG]Base   =0.00;
\[Rho]       =\[Rho]Base       = 2.0;
CheckFor\[CapitalGamma]Impatience;
CheckForReturnImpatience;
{\[ScriptC]EBase,\[ScriptM]EBase} = {\[ScriptC]E,\[ScriptM]E};
VerboseOutput=True;
ParamsAreSet=True;

