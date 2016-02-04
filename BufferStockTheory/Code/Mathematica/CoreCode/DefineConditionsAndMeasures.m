(* ::Package:: *)

(* Objects computed from parameters *)
(* Patience factors, etc *)
(* These need to be defined BEFORE the parameters are assigned values *)


FHW  = \[CapitalGamma]/(R); (* Finite human wealth condition depends on this parameter *)
\[HBar]Inf = If[FHWCHolds == True,1/(1-FHW),10000]; (* If FHW condition fails, choose large value for max human wealth *)

PFGICValue = \[CapitalThorn]/\[CapitalGamma];
GICValue = \[Psi]Probs . (\[CapitalThorn]/(\[CapitalGamma] \[Psi]Vals));
RICValue = \[CapitalThorn]/(R);

\[DoubleStruckCapitalE]\[ScriptCapitalR]       = (R)/\[CapitalGamma]Adj;
\[DoubleStruckCapitalE]\[ScriptR]       = \[DoubleStruckCapitalE]\[ScriptCapitalR] - 1;

\[WeierstrassP]Cancel  = 1-\[WeierstrassP];
\[ScriptCapitalR]        = (R)/\[CapitalGamma];

(* Absolute impatience  *)  \[CapitalThorn]        = ((R) \[Beta])^(1/\[Rho]); 
(* Return impatience    *)  \[CapitalThorn]Rtn     = \[CapitalThorn]/(R);
(* PF Growth impatience *)  \[CapitalThorn]\[CapitalGamma]PF   = \[CapitalThorn]/\[CapitalGamma];
(* Growth impatience    *)  \[CapitalThorn]\[CapitalGamma]     = \[CapitalThorn]/\[CapitalGamma]Adj;

\Inv\[DoubleStruckCapitalE]pShkInvLim      = 1/Exp[\[Sigma]Perm^2];
(* %%% *)\[DoubleStruckCapitalE]pShkTo1m\[Rho]Inv := (Exp[(\[Sigma]Perm^2)\[Rho] (\[Rho]-1)/2])^(1/(1-\[Rho]));

\[CapitalGamma]Adj     = \[CapitalGamma] \Inv\[DoubleStruckCapitalE]pShkInvLim;

FVAPF    := \[Beta] \[CapitalGamma]^(1-\[Rho]); (* Finite Value of Autarky measure for Perfect Foresight model: Value is finite if FVA < 1 *)
FVA      := \[Beta] \[CapitalGamma]^(1-\[Rho]) Exp[(\[Sigma]Perm^2)\[Rho] (\[Rho]-1)/2]; (* Search Derivations.nb for FVA *)

GICHolds   := (\[CapitalThorn]\[CapitalGamma] < 1);     (* Growth Impatience Condition; returns "True" if condition holds, otherwise false *)
GICFails   := Not[GICHolds];
PFGICHolds := (\[CapitalThorn]\[CapitalGamma]PF < 1);   (* Perfect Foresight Growth Impatience Condition; returns "True" if condition holds, otherwise false *)
PFGICFails := Not[PFGICHolds];
RICHolds   := (\[CapitalThorn]Rtn < 1);     (* Return Impatience Condition holds true *)
RICFails   := Not[RICHolds]; 
FVACHolds   := (FVA < 1);      (* Finite Growth Value Condition holds true *)
FVACFails   := Not[FVACHolds];
FHWCHolds   := (FHW < 1);      (* Finite Human Wealth Condition holds true *)
FHWCFails   := Not[FHWCHolds];

\[Kappa]MinInf = 1-\[CapitalThorn]/(R);         (* Minimum Infinite Horizon MPC *)
\[Kappa]MaxInf = 1-\[WeierstrassP]^(1/\[Rho]) \[CapitalThorn]/(R); (* Maximum Infinite Horizon MPC *)

\[Lambda]Above = \[CapitalThorn]/(R);          (* Minimum Infinite Horizon marginal propensity to save *)
\[Lambda]Below = \[WeierstrassP]^(1/\[Rho]) \[Lambda]Above; (* Maximum Infinite Horizon marginal propensity to save *)

\[Bet] := \[Beta] NIntegrate[\[Bullet]^(1-\[Rho]) PDF[LogNormalDistribution[-\[Sigma]Perm^2/2,\[Sigma]Perm],\[Bullet]],{\[Bullet],0,Infinity}];
\[Bet]::usage = "Used to construct constant used in formula to normalize value in contraction mapping proof.";

\[Eta]Bar := (\[Bet]/(1-\[Bet])) (\[WeierstrassP]Cancel^\[Rho]) Min[Select[\[Theta]Vals,#>0.&]]^(1-\[Rho]);
\[Eta]Bar::usage = "Lower bound of constant used to normalize value in contraction mapping proof.";

\[GothicV] := 1/(1-\[Beta] (((R) \[Beta])^((1/\[Rho])-1)));
\[GothicV]::usage = "u[c_t] times \[GothicV] yields the discounted value for a person consuming 1 unit forever.";
