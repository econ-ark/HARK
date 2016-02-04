(* ::Package:: *)

(* 
These parameter values are chosen not for realism 
but in order to generate figures that illustrate the qualitative 
features of the model as clearly as possible.  More realistic
parameter values should be used for quantitative exercises (like Carroll
and Jeanne (2009))
*)

\[Mho]=\[Mho]Base=0.005; (* From Carroll (1992); though that number was for transitory unemployment shocks *)
\[CurlyTheta]=\[CurlyTheta]Base=0.10;
(r)=rBase   = 0.03;
\[GothicG]=\[GothicG]Base=0.00;
\[Rho]=\[Rho]Base= 2.0;
Severance = 0;
PDies = 0;
(* \[Tau] is stake. Stake = 0 indicates there's no stake *)
\[Tau] = 0; 
CheckFor\[CapitalGamma]Impatience;
CheckForRImpatience;
{\[ScriptC]EBase,\[ScriptM]EBase} = {\[ScriptC]E,\[ScriptM]E};

(* Distance from steady state to begin backwards shooting *)
\[CurlyEpsilon] = 0.01 \[ScriptM]E; 

(* Code blows up if run without parameters set; this is a signal used to test it's OK to proceed *)
ParamsAreSet=True;

