(* ::Package:: *)

SetupShocks := Block[{},
(* Construct the possible values of the shock to income *)

{\[Theta]Vals,\[Theta]Probs} = DiscreteApproxToMeanOneLogNormal[\[Sigma]Tran,TranGridLengthSetup];
{\[Psi]Vals,\[Psi]Probs} = DiscreteApproxToMeanOneLogNormal[\[Sigma]Perm,PermGridLength];

If[\[WeierstrassP]>0,
  (* then modify the shocks and probabilities to incorporate the zero-income event as a transitory shock *)
  \[Theta]Vals = \[Theta]Vals (1-\[WeierstrassP] \[Theta]Min)/\[WeierstrassP]Cancel;
  \[Theta]Vals = Prepend[\[Theta]Vals,\[Theta]Min];
  \[Theta]Probs = \[Theta]Probs \[WeierstrassP]Cancel;
  \[Theta]Probs = Prepend[\[Theta]Probs,\[WeierstrassP]];
];

TranGridLength=Length[\[Theta]Vals];

Inv\Inv\[DoubleStruckCapitalE]pShkInvAct=(Sum[\[Psi]Probs[[\[Psi]ShockLoop]] (\[Psi]Vals[[\[Psi]ShockLoop]])^(-1),{\[Psi]ShockLoop,Length[\[Psi]Vals]}])^(-1);
\[CapitalGamma]Adj = \[CapitalGamma] Inv\Inv\[DoubleStruckCapitalE]pShkInv;
\[DoubleStruckCapitalE]\[ScriptCapitalR]       = (R)/\[CapitalGamma]Adj;
\[DoubleStruckCapitalE]\[ScriptR]       = \[DoubleStruckCapitalE]\[ScriptCapitalR] - 1;

\[Eta] = \[Eta]Bar + 1;

If[VerboseOutput==True,Print["With new shocks, adjusted Growth Patience Factor is ",\[CapitalThorn]\[CapitalGamma]]];
];


