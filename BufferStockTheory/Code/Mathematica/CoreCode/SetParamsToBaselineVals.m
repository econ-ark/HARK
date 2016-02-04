(* ::Package:: *)

SetParamsBaseline := Block[{},
(* Parameter values for the baseline version of the model without constraints *)

(* Income growth factor            *) \[CapitalGamma] = 1.03;
(* Gross interest factor           *) (R) = 1.04;

(* Discount factor                 *) \[Beta] = 0.96;
(* Coeff of Rel Risk Aversion      *) \[Rho] = 2;                    
(* Prob of unemployment events     *) \[WeierstrassP]  = 0.005;
(* Value of income when unemployed *) \[Theta]Min = 0.0;
(* Std dev of transitory lognormal *) \[Sigma]Tran = 0.1;             
(* Std dev of permanent  lognormal *) \[Sigma]Perm = 0.1;             

TranGridLengthSetup = 3;  (* Num of pts in discrete approx to lognormal dist *)
PermGridLength      = 3;  (* Num of pts in discrete approx to lognormal dist *)

(* SetupGrids defines dense grid for small a, less dense for larger a *)
aGridVecExcBotSmallLength = 5;
aGridVecExcBotLargeLength = 10;
\[CurlyEpsilon] = 10^-14;
CoarseAccuracy = 5; (* Accuracy goal for targets *)
TimesToNestSmallGrid = 20 (* Controls degree of multi in multi-exponential growth for small gridpoints *)
];

aLargeMax = 100 \[HBar]Inf; (* Large number defines the maximum gridpoint for a *)
aSmallMax = 3;  (* Small number defines the gridpoint below which aGridVecExcBot will be dense *)


SetParamsBaseline;

