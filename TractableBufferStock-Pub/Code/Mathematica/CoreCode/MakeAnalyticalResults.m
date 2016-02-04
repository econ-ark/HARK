(* ::Package:: *)

(* This file constructs analytical formulae for the derivatives of the Euler equation *)
(* These formulae could be derived (tediously) by hand, but such derivations are error-prone *)
(* Once derived, the formulae automatically update when parameter values are changed *)


(* This file should be executed before parameter values are defined, enforced by the following line *)
If[NameQ["ParamsAreSet"]==True,If[ParamsAreSet==True,Print["MakeAnalyticalResults should be executed before parameter values are defined."];Abort[]]];

(* Derivations necessary for calculating derivatives of consumption function at the target *)

(* Start with Euler equation which reflects first order condition *)
Euler1 = {u'[cE[\[ScriptM]Et]] == \[CapitalThorn]\[CapitalGamma]^\[Rho] (\[Mho] u'[\[Kappa] (\[ScriptM]Et-cE[\[ScriptM]Et]) \[ScriptCapitalR]+Severance]+(1-\[Mho]) u'[cE[(\[ScriptM]Et-cE[\[ScriptM]Et]) \[ScriptCapitalR]+1-SoiTax]])}; 

(* Differentiate the Euler equation with respect to \[ScriptM]Et *)
Euler2 = D[Euler1,\[ScriptM]Et];
Euler3 = D[Euler2,\[ScriptM]Et];
Euler4 = D[Euler3,\[ScriptM]Et];
Euler5 = D[Euler4,\[ScriptM]Et];
Euler6 = D[Euler5,\[ScriptM]Et];
(* Substitute values at the target to obtain an implicit equation for the MPC at the target *)
Euler2AtTarget =
 (
   (
     (
       (
         (
           (
             Euler2 
                /. \[ScriptM]Et -> \[ScriptM]E
             ) /. cE[\[ScriptM]E] -> \[ScriptC]E
           ) /.  (\[ScriptM]E-\[ScriptC]E) \[ScriptCapitalR] -> \[ScriptB]E
         ) /. \[ScriptB]E+1-SoiTax -> \[ScriptM]E 
       ) /. \[Kappa] \[ScriptB]E+Severance -> \[ScriptC]U
     ) /. cE[\[ScriptM]E] -> \[ScriptC]E
   ) /. cE'[\[ScriptM]E] -> \[ScriptC]EP;


(* Implicit equation for cE'' *)
Euler3AtTarget =
    (
      (
        (
          (
            (
              (
                (
                Euler3 /. \[ScriptM]Et -> \[ScriptM]E
                ) /. cE[\[ScriptM]E] -> \[ScriptC]E
              ) /.  (\[ScriptM]E-\[ScriptC]E) \[ScriptCapitalR] -> \[ScriptB]E
            ) /. \[ScriptB]E+1-SoiTax -> \[ScriptM]E 
          ) /. \[Kappa] \[ScriptB]E+Severance -> \[ScriptC]U
        ) /. cE[\[ScriptM]E] -> \[ScriptC]E
      ) /. cE'[\[ScriptM]E] -> \[ScriptC]EP
    ) /. cE''[\[ScriptM]E] -> \[ScriptC]EPP;


(* Implicit equation for cE''' *)
Euler4AtTarget =
  (
    (
      (
        (
          (
            (
              (
                (
                Euler4 /. \[ScriptM]Et -> \[ScriptM]E
                ) /. cE[\[ScriptM]E] -> \[ScriptC]E
              ) /. (\[ScriptM]E-\[ScriptC]E) \[ScriptCapitalR] -> \[ScriptB]E
            ) /. \[ScriptB]E+1-SoiTax -> \[ScriptM]E 
          ) /. \[Kappa] \[ScriptB]E+Severance -> \[ScriptC]U
        ) /. cE[\[ScriptM]E] -> \[ScriptC]E
      ) /. cE'[\[ScriptM]E] -> \[ScriptC]EP
    ) /. cE''[\[ScriptM]E] -> \[ScriptC]EPP
  ) /. cE'''[\[ScriptM]E] -> \[ScriptC]EPPP;


 (* Implicit equation for cE'''' *)
 Euler5AtTarget =
 (
   (
     (
       (
         (
           (
             (
               (
                 (
                 Euler5 /. \[ScriptM]Et -> \[ScriptM]E
                 ) /. cE[\[ScriptM]E] -> \[ScriptC]E
               ) /. (\[ScriptM]E-\[ScriptC]E) \[ScriptCapitalR] -> \[ScriptB]E
             ) /. \[ScriptB]E+1-SoiTax -> \[ScriptM]E  
           ) /. \[Kappa] \[ScriptB]E+Severance -> \[ScriptC]U
         ) /. cE[\[ScriptM]E] -> \[ScriptC]E
       ) /. cE'[\[ScriptM]E] -> \[ScriptC]EP
     ) /. cE''[\[ScriptM]E] -> \[ScriptC]EPP
   ) /. cE'''[\[ScriptM]E] -> \[ScriptC]EPPP
 ) /. cE''''[\[ScriptM]E] -> \[ScriptC]EPPPP;


 (* Implicit equation for cE'''''*)
 Euler6AtTarget =
(
 (
   (
     (
       (
         (
           (
             (
               (
                 (
                 Euler6 /. \[ScriptM]Et -> \[ScriptM]E
                 ) /. cE[\[ScriptM]E] -> \[ScriptC]E
               ) /. (\[ScriptM]E-\[ScriptC]E) \[ScriptCapitalR] -> \[ScriptB]E
             ) /. \[ScriptB]E+1-SoiTax -> \[ScriptM]E  
           ) /. \[Kappa] \[ScriptB]E+Severance -> \[ScriptC]U
         ) /. cE[\[ScriptM]E] -> \[ScriptC]E
       ) /. cE'[\[ScriptM]E] -> \[ScriptC]EP
     ) /. cE''[\[ScriptM]E] -> \[ScriptC]EPP
   ) /. cE'''[\[ScriptM]E] -> \[ScriptC]EPPP
 ) /. cE''''[\[ScriptM]E] -> \[ScriptC]EPPPP
)/.cE'''''[\[ScriptM]E] -> \[ScriptC]EPPPPP;


(* Solve for analytical formulae for derivatives of the consumption functions at the target *)
\[ScriptC]EPAnalytical    = \[ScriptC]EP     /. Solve[Euler2AtTarget, \[ScriptC]EP];  (* This is a quadratic; later must select the solution in the range [0,1] *)
\[ScriptC]EPPAnalytical   = \[ScriptC]EPP    /. Solve[Euler3AtTarget, \[ScriptC]EPP][[1]]; (* These are linear solutions, given the lower-order derivatives *)
\[ScriptC]EPPPAnalytical  = \[ScriptC]EPPP   /. Solve[Euler4AtTarget, \[ScriptC]EPPP][[1]];
\[ScriptC]EPPPPAnalytical = \[ScriptC]EPPPP  /. Solve[Euler5AtTarget, \[ScriptC]EPPPP][[1]]; (* Needed only for the version that uses second derivatives in interpolation *)
\[ScriptC]EPPPPPAnalytical= \[ScriptC]EPPPPP /. Solve[Euler6AtTarget, \[ScriptC]EPPPPP][[1]]; 


(* Now construct analytical solution for parameters of extrapolating function using the first four derivatives *)
(* There will be two solutions which are numerically equivalent; we use the first, arbitrarily *)

Off[Solve::"ifun"]; (* Turn off an error message that warns about the solution -- it's OK *)
ClearAll[\[Phi]Solve0, \[Phi]Solve1, \[Gamma]Solve0, \[Gamma]Solve1];
ClearAll[s0, s1, s2, s3];
{ExpParams1,ExpParams2} = Assuming[{{\[Phi]Solve0,\[Phi]Solve1,\[Gamma]Solve0,\[Gamma]Solve1} \[Element] Reals},Solve[{
    s0 ==            Exp[\[Phi]Solve0] +           Exp[\[Gamma]Solve0]
  , s1 == -\[Phi]Solve1   Exp[\[Phi]Solve0] - \[Gamma]Solve1   Exp[\[Gamma]Solve0]
  , s2 ==  \[Phi]Solve1^2 Exp[\[Phi]Solve0] + \[Gamma]Solve1^2 Exp[\[Gamma]Solve0]
  , s3 == -\[Phi]Solve1^3 Exp[\[Phi]Solve0] - \[Gamma]Solve1^3 Exp[\[Gamma]Solve0]}
 , {\[Phi]Solve0, \[Phi]Solve1, \[Gamma]Solve0, \[Gamma]Solve1}]];

\[Phi]Analytical0 = ((\[Phi]Solve0 /. ExpParams1) /. C[1] -> 0) /. C[2] -> 0;
\[Phi]Analytical1 = ((\[Phi]Solve1 /. ExpParams1) /. C[1] -> 0) /. C[2] -> 0;
\[Gamma]Analytical0 = ((\[Gamma]Solve0 /. ExpParams1) /. C[1] -> 0) /. C[2] -> 0;
\[Gamma]Analytical1 = ((\[Gamma]Solve1 /. ExpParams1) /. C[1] -> 0) /. C[2] -> 0;
On[Solve::"ifun"]; 

