(* ::Package:: *)

(* Lines with %%% in them are private code that is automatically removed in creating the public archive *)
(* Lines with %%% contain extra material that is worth preserving even if not useful to public readers *)
(* Material between any line containing beginCDCPrivate *)
(* and the next line containing endCDCPrivate is also excised (so, these two lines will disappear!) *)

(* Define policy, value, and other functions *)
(* Many of the functions below require that the solution for the last period has already been constructed *)
(* The solutions for periods that have already been solved reside in lists like cInterpFunc *) 

(* u           *) u[c_?NumericQ]   := (c^(1-\[Rho]))/(1-\[Rho]);
(* u'          *) uP[c_?NumericQ]  := c^-\[Rho];  
(* u''         *) uPP[c_?NumericQ] := -\[Rho] c^-(\[Rho]+1);
(* u' Inverse  *) nP[z_?NumericQ]  := z^-(1/\[Rho]);            
(* u'' Inverse *) nPP[z_?NumericQ] := -(z/\[Rho])^(-1/(1+\[Rho]));
SetAttributes[{u,uP,uPP,nP,nPP},Listable];

(* With no date argument, functions below return results for the last-solved period *)
(* Digamma \[Digamma] marks 'perfect foresight' versions *)
(* {Fin,Inf} designate finite vs infinite horizon solutions *)

(* Finite and Infinite Horizon bounds for consumption rule *)
\[ScriptC]\[Digamma]Inf[m_?NumericQ]                       := (m-1+\[HBar]Inf                )\[Kappa]MinInf;
\[ScriptC]\[Digamma]Fin[m_?NumericQ]                       := (m-1+\[HBar][[-1]]             )\[Kappa]Min[[-1]];
\[ScriptC]\[Digamma]Fin[m_?NumericQ,TimeToT_]              := (m-1+\[HBar][[TimeToT+1]])\[Kappa]Min[[TimeToT+1]];
\[ScriptC]From\[Kappa]MinInf[m_?NumericQ]                := m \[Kappa]MinInf;
\[ScriptC]From\[Kappa]MinFin[m_?NumericQ]                := m \[Kappa]Min[[-1]];
\[ScriptC]From\[Kappa]MinFin[m_?NumericQ,TimeToT_]       := m \[Kappa]Min[[TimeToT+1]];
\[ScriptC]From\[Kappa]MaxInf[m_?NumericQ]                := m \[Kappa]MaxInf;
\[ScriptC]From\[Kappa]MaxFin[m_?NumericQ]                := m \[Kappa]Max[[-1]];
\[ScriptC]From\[Kappa]MaxFin[m_?NumericQ,TimeToT_]       := m \[Kappa]Max[[TimeToT+1]];
\[ScriptC]BotBoundInf[m_?NumericQ]                := \[ScriptC]From\[Kappa]MinInf[m];
\[ScriptC]BotBoundFin[m_?NumericQ]                := \[ScriptC]From\[Kappa]MinFin[m];
\[ScriptC]BotBoundFin[m_?NumericQ,TimeToT_]       := \[ScriptC]From\[Kappa]MinFin[m,TimeToT];
\[ScriptC]TopBoundInf[m_?NumericQ]                := Min[\[ScriptC]From\[Kappa]MaxInf[m],\[ScriptC]\[Digamma]Inf[m]];
\[ScriptC]TopBoundFin[m_?NumericQ]                := Min[\[ScriptC]From\[Kappa]MaxFin[m],\[ScriptC]\[Digamma]Fin[m]];
\[ScriptC]TopBoundFin[m_?NumericQ,TimeToT_]       := Min[\[ScriptC]From\[Kappa]MaxFin[m,TimeToT],\[ScriptC]\[Digamma]Fin[m,TimeToT]];

(* Locus of budget-stable points *)
\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt[m_?NumericQ]      := 1+(m-1)((\[DoubleStruckCapitalE]\[ScriptR])/(\[DoubleStruckCapitalE]\[ScriptCapitalR]));
\[ScriptC]SuchThat\[DoubleStruckCapitalE]mtp1Equalsmt::usage = "Sustainable consumption (leaves expected m unchanged).";


(* See appendix SolnMethTermFunc for definitions/derivations of the following *)
(* TimeToT indicates the number of periods that remain until the terminal period T *)
\[Chi]'[\[Mu]_?NumericQ]          := \[Chi]Full[[Horizon+1]]'[\[Mu]]; (* +1 because the first element in the list is [[1]] for period T-0 *)
\[Chi]'[\[Mu]_?NumericQ,TimeToT_] := \[Chi]Full[[TimeToT+1]]'[\[Mu]]; 
\[Chi][\[Mu]_?NumericQ]           := \[Chi]Full[[Horizon+1]][\[Mu]];
\[Chi][\[Mu]_?NumericQ,TimeToT_]  := \[Chi]Full[[TimeToT+1]][\[Mu]];

\[ScriptC]::usage = "Consumption ratio [ScriptC] as a function of cash-on-hand m.";
\[ScriptC][m_?NumericQ]            := If[Horizon== 0
    ,(*then*)\[ScriptC]Lim[m,0]
    ,(*else*)
      If[m<\[ScriptC]LinearSplineData[[Horizon+1,2,1,1]]    (* First nonzero m gridpoint *)
        ,(*then*)\[ScriptC]LinearSplineFunc[[Horizon+1]][m] (* Assume consumption is linear below bottom gridpoint *)
        ,(*else*)(1-(Exp[\[Chi][Log[m]        ]]-\[CurlyEpsilon]))(\[ScriptC]Lim[m              ]+\[Theta]Min)]]; (* See ./Documentation/Derivations.nb *)
\[ScriptC][m_?NumericQ,TimeToT_]   := If[TimeToT == 0
    ,(*then*)\[ScriptC]Lim[m,0]
    ,(*else*)
      If[m<\[ScriptC]LinearSplineData[[TimeToT+1,2,1,1]]
        ,(*then*)\[ScriptC]LinearSplineFunc[[TimeToT+1]][m]
        ,(*else*)(1-(Exp[\[Chi][Log[m],TimeToT]]-\[CurlyEpsilon]))(\[ScriptC]Lim[m,TimeToT]+\[Theta]Min)]];

\[Kappa]::usage = "Marginal Propensity to Consume [Kappa] as a function of cash-on-hand \[ScriptM].";
\[Kappa][m_?NumericQ]                 := If[Horizon == 0
    ,(*then*)\[Kappa]Lim[m,0]
    ,(*else*)
    If[m<\[ScriptC]LinearSplineData[[Horizon+1,2,1,1]]     (* First nonzero m gridpoint *)
      ,(*then*)\[ScriptC]LinearSplineFunc[[Horizon+1]]'[m] (* Constant MPC below first gridpoint *)
      ,(*else*)((\[Theta]Min + \[ScriptC]Lim[m]) (1+\[CurlyEpsilon]-\[ScriptC][m]/(\[Theta]Min + \[ScriptC]Lim[m]))((m \[ScriptC][m] \[Kappa]Lim[m])/
                                 ((\[Theta]Min + \[ScriptC]Lim[m])^2 (1+\[CurlyEpsilon]-\[ScriptC][m]/(\[Theta]Min + \[ScriptC]Lim[m])))-\[Chi]'[Log[m]]))/m]]; (* See Derivations.nb *)
\[Kappa][m_?NumericQ,TimeToT_]  := If[TimeToT == 0
    ,(*then*)\[Kappa]Lim[m,0]
    ,(*else*)
    If[m<\[ScriptC]LinearSplineData[[Horizon+1,2,1,1]]
       ,(*then*)\[ScriptC]LinearSplineFunc[[TimeToT+1]]'[m] (* Below bottom solved point, use linear consumption rule *)
       ,(*else*)((\[Theta]Min + \[ScriptC]Lim[m]) (1+\[CurlyEpsilon]-\[ScriptC][m,TimeToT]/(\[Theta]Min + \[ScriptC]Lim[m]))((m \[ScriptC][m,TimeToT] \[Kappa]Lim[m])/
                                  ((\[Theta]Min + \[ScriptC]Lim[m])^2 (1+\[CurlyEpsilon]-\[ScriptC][m,TimeToT]/(\[Theta]Min + \[ScriptC]Lim[m])))-\[Chi]'[Log[m],TimeToT]))/m]];

\[ScriptV]::usage = "Value [ScriptV] as a function of cash-on-hand m.";
\[ScriptV][m_?NumericQ] := 
  If[(mBot = First[vNormInterpData[[Horizon+1]]][[mPos=1,1]]) <= m <= (mTop = Last[vNormInterpData[[Horizon+1]]][[mPos=1,1]])
    ,(* then *) -Exp[vNormInterpFunc[[Horizon+1]][m]](\[Eta]+m^(1-\[Rho]))
    ,(* else *)If[m<=mBot
        ,(*then*)mBound=mBot;vBound=vNormInterpData[[Horizon+1, 1,2]];-Exp[vBound](\[Eta]+mBound^(1-\[Rho]))-NIntegrate[uP[\[ScriptC][\[Bullet]        ]],{\[Bullet],m,mBound}]
        ,(*else*)mBound=mTop;vBound=vNormInterpData[[Horizon+1,-1,2]];-Exp[vBound](\[Eta]+mBound^(1-\[Rho]))+NIntegrate[uP[\[ScriptC][\[Bullet]        ]],{\[Bullet],mBound,m}]]];
\[ScriptV][m_?NumericQ,TimeToT_] := 
  If[(mBot = First[vNormInterpData[[Horizon+1]]][[mPos=1,1]]) <= m <= (mTop = Last[vNormInterpData[[Horizon+1]]][[mPos=1,1]])
    ,(* then *) -Exp[vNormInterpFunc[[TimeToT+1]][m]](\[Eta]+m^(1-\[Rho]))
    ,(* else *)If[m<=mBot
        ,(*then*)mBound=mBot;vBound=vNormInterpData[[TimeToT+1, 1,2]];-Exp[vBound](\[Eta]+mBound^(1-\[Rho]))-NIntegrate[uP[\[ScriptC][\[Bullet]        ]],{\[Bullet],m,mBound}]
        ,(*else*)mBound=mTop;vBound=vNormInterpData[[TimeToT+1,-1,2]];-Exp[vBound](\[Eta]+mBound^(1-\[Rho]))+NIntegrate[uP[\[ScriptC][\[Bullet]        ]],{\[Bullet],mBound,m}]]];

\[ScriptV]P::usage = "Marginal [ScriptV]P as a function of cash-on-hand m.";
\[ScriptV]P[m_?NumericQ] := 
  If[(mBot = First[vNormInterpData[[Horizon+1]]][[mPos=1,1]]) <= m <= (mTop = Last[vNormInterpData[[Horizon+1]]][[mPos=1,1]])
       ,(* then *) vPNormVal = vNormInterpFunc[[Horizon+1]]'[m];
                   (((1+m vPNormVal)+(m^\[Rho]) vPNormVal \[Eta]-\[Rho]) \[ScriptV][m              ])/(m+(m^\[Rho]) \[Eta])
       ,(* else *) uP[\[ScriptC][m              ]]];
\[ScriptV]P[m_?NumericQ,TimeToT_]  := 
  If[(mBot = First[vNormInterpData[[Horizon+1]]][[mPos=1,1]]) <= m <= (mTop = Last[vNormInterpData[[Horizon+1]]][[mPos=1,1]])
       ,(* then *) vPNormVal = vNormInterpFunc[[TimeToT+1]]'[m];
                   (((1+m vPNormVal)+(m^\[Rho]) vPNormVal \[Eta]-\[Rho]) \[ScriptV][m,TimeToT])/(m+(m^\[Rho]) \[Eta])
       ,(* else *) uP[\[ScriptC][m,TimeToT]]];

(* beginCDCPrivate *)
\[ScriptV]Raw[m_?NumericQ]            := -Exp[vNormFuncFull[[Horizon+1]][m]](\[Eta]+m^(1-\[Rho]));
\[ScriptV]Raw[m_?NumericQ,TimeToT_]   := -Exp[vNormFuncFull[[TimeToT+1]][m]](\[Eta]+m^(1-\[Rho]))
\[ScriptV]PRaw[m_?NumericQ]           := Block[{},
    vPNormVal = vNormFuncFull[[Horizon+1]]'[m];
    (((1+m vPNormVal)+(m^\[Rho]) vPNormVal \[Eta]-\[Rho]) \[ScriptV]Raw[m              ])/(m+(m^\[Rho]) \[Eta])];
\[ScriptV]PRaw[m_?NumericQ,TimeToT_]  := Block[{},vPNormVal = vNormFuncFull[[TimeToT+1]]'[m];
                   (((1+m vPNormVal)+(m^\[Rho]) vPNormVal \[Eta]-\[Rho]) \[ScriptV]Raw[m,TimeToT])/(m+(m^\[Rho]) \[Eta])];
\[ScriptV]FromGapTest[m_?NumericQ]                  := If[Log[m] <= (\[Mu]Top=Last[vGapInterpData[[Horizon+1]]][[1,1]])
    ,(* then *) (1/(1-Exp[vGapFull[[Horizon+1]][Log[m]]]   ))\[ScriptV]Lim[m]
    ,(* else *) (1/(1-Exp[vGapInterpData[[Horizon+1,-1,2]]]))\[ScriptV]Lim[Exp[\[Mu]Top]]
           +NIntegrate[uP[\[ScriptC][\[Bullet]              ]],{\[Bullet],Exp[\[Mu]Top],m}]];
\[ScriptV]FromGap[m_?NumericQ]             := (1/(1-Exp[vGapFull[[Horizon+1]][Log[m]]]   ))\[ScriptV]Lim[m];
\[ScriptV]FromGapTest[m_?NumericQ,TimeToT_]   := If[Log[m] <= (\[Mu]Top=Last[vNormInterpData[[TimeToT+1]]][[1,1]])
    ,(* then *) (1/(1-Exp[vGapFull[[TimeToT+1]][Log[m]]]   ))\[ScriptV]Lim[m]
    ,(* else *) (1/(1-Exp[vGapInterpData[[TimeToT+1,-1,2]]]))\[ScriptV]Lim[Exp[\[Mu]Top]]
           +NIntegrate[uP[\[ScriptC][\[Bullet],TimeToT]],{\[Bullet],Exp[\[Mu]Top],m}]];
\[ScriptV]FromGapP[m_?NumericQ]                 := Block[{},
    vGapVal=vGapFull[[Horizon+1]][Log[m]];vPGapVal=vGapFull[[Horizon+1]]'[Log[m]];
    \[ScriptV]PLim[m]/(1-E^vGapVal)+(E^vGapVal*\[ScriptV]Lim[m]vPGapVal)/(m (1 - E^vGapVal)^2)];
\[ScriptV]FromGapPTest[m_?NumericQ]                 := 
  If[Exp[First[vGapInterpData[[Horizon+1]]][[1,1]]] <= m <= Exp[Last[vGapInterpData[[Horizon+1]]][[1,1]]]
       ,(* then *) vGapVal=vGapFull[[Horizon+1]][Log[m]];vPGapVal=vGapFull[[Horizon+1]]'[Log[m]];
                   \[ScriptV]PLim[m]/(1-E^vGapVal)+(E^vGapVal*\[ScriptV]Lim[m]vPGapVal)/(m (1 - E^vGapVal)^2)
       ,(* else *) uP[\[ScriptC][m              ]]];
\[ScriptV]FromGapP[m_?NumericQ,TimeToT_]  := Block[{}, vGapVal=vGapFull[[TimeToT+1]][Log[m]];vPGapVal=vGapFull[[TimeToT+1]]'[Log[m]];
                   \[ScriptV]PLim[m]/(1-E^vGapVal)+(E^vGapVal*\[ScriptV]Lim[m]vPGapVal)/(m (1 - E^vGapVal)^2)];
\[ScriptV]FromGapPTest[m_?NumericQ,TimeToT_]  := 
  If[Exp[First[vGapInterpData[[TimeToT+1]]][[1,1]]] <= m <= Exp[Last[vGapInterpData[[TimeToT+1]]][[1,1]]]
       ,(* then *) vGapVal=vGapFull[[TimeToT+1]][Log[m]];vPGapVal=vGapFull[[TimeToT+1]]'[Log[m]];
                   \[ScriptV]PLim[m]/(1-E^vGapVal)+(E^vGapVal*\[ScriptV]Lim[m]vPGapVal)/(m (1 - E^vGapVal)^2)
       ,(* else *) uP[\[ScriptC][m,TimeToT]]];

(*#
\[ScriptV]FromvOuOfc[m_?NumericQ]                  := If[m <= vOuOfcInterpData[[Horizon+1,-1,1,1]],
     (* then *) (-Exp[vOuOfcFuncFull[[Horizon+1]][m]]+\[GothicV])u[\[ScriptC][m]]
    ,(* else *) vOuOfcInterpData[[Horizon+1,-1,2]]+NIntegrate[uP[\[ScriptC][\[Bullet]]],{\[Bullet],vOuOfcInterpData[[Horizon+1,1,1,1]],m}]];
\[ScriptV]FromvOuOfc[m_?NumericQ,TimeToT_]   := If[m <= vOuOfcInterpData[[TimeToT+1,-1,1,1]],
     (* then *) (-Exp[vOuOfcFuncFull[[TimeToT+1]][m]]+\[GothicV])u[\[ScriptC][m,TimeToT]]
    ,(* else *) vOuOfcInterpData[[TimeToT+1,-1,2]]+NIntegrate[uP[\[ScriptC][\[Bullet],TimeToT]],{\[Bullet],vOuOfcInterpData[[TimeToT,1,1,1]],m}]];
\[ScriptV]FromvOuOfcP[m_?NumericQ]                 := If[m <= vOuOfcInterpData[[Horizon+1,-1,1,1]],
     (* then *)vOuOfcPVal=vOuOfcFuncFull[[Horizon+1]]'[m];
       -(1/u[\[ScriptC][m]]) \[ScriptC][m]^-\[Rho] (vOuOfcPVal \[GothicV] u[\[ScriptC][m]]^2 \[ScriptC][m]^\[Rho]-vOuOfcPVal u[\[ScriptC][m]] \[ScriptC][m]^\[Rho] \[ScriptV]FromvOuOfc[m]-\[ScriptV]FromvOuOfc[m] \[Kappa][m])
    ,(* else *) uP[\[ScriptC][m]]];
\[ScriptV]FromvOuOfcP[m_?NumericQ,TimeToT_]  := If[m <= vOuOfcInterpData[[TimeToT+1,-1,1,1]],
     (* then *) vOuOfcVal=vOuOfcFuncFull[[TimeToT+1]][m];vOuOfcPVal=vOuOfcFuncFull[[TimeToT+1]]'[m];cVal=\[ScriptC][m,TimeToT];\[Kappa]Val=\[Kappa][m,TimeToT];
       -(1/u[cVal]) cVal^-\[Rho] (vOuOfcPVal \[GothicV] u[cVal]^2 cVal^\[Rho]-vOuOfcPVal u[cVal] cVal^\[Rho] vOuOfcVal-vOuOfcVal \[Kappa]Val)
    ,(* else *) uP[\[ScriptC][m,TimeToT]]];
*)

\[ScriptV]FromvOu[m_?NumericQ]                  := If[m <= (mTop=Last[vOuInterpData[[Horizon+1]]][[1,1]])
    ,(* then *) vOuFuncFull[[Horizon+1]][m] u[m]
    ,(* else *) Last[vOuInterpData[[Horizon+1]]][[2]]u[mTop]+NIntegrate[uP[\[ScriptC][\[Bullet]              ]],{\[Bullet],mTop,m}]];
\[ScriptV]FromvOu[m_?NumericQ,TimeToT_]   := If[m <= (mTop=Last[vOuInterpData[[TimeToT+1]]][[1,1]])
    ,(* then *) vOuFuncFull[[TimeToT+1]][m] u[m]
    ,(* else *) Last[vOuInterpData[[TimeToT+1]]][[2]]u[mTop]+NIntegrate[uP[\[ScriptC][\[Bullet],TimeToT]],{\[Bullet],mTop,m}]];
\[ScriptV]FromvOuP[m_?NumericQ]                 := If[m <= (mTop=Last[vOuInterpData[[Horizon+1]]][[1,1]])
    ,(* then *){vOuVal,vOuPVal}={vOuFuncFull[[Horizon+1]][m],vOuFuncFull[[Horizon+1]]'[m]};
        (vOuPVal*u[m]^2 + \[ScriptV]FromvOu[m]*uP[m])/u[m]
    ,(* else *)uP[\[ScriptC][m               ]]];
\[ScriptV]FromvOuP[m_?NumericQ,TimeToT_]  := If[m <= (mTop=Last[vOuInterpData[[TimeToT+1]]][[1,1]])
    ,(* then *){vOuVal,vOuPVal}={vOuFuncFull[[TimeToT+1]][m],vOuFuncFull[[TimeToT+1]]'[m]};
        (vOuPVal*u[m]^2 + \[ScriptV]FromvOu[m]*uP[m])/u[m]
    ,(* else *) uP[\[ScriptC][m,TimeToT]]];
(* 
\[ScriptV]Old[m_?NumericQ]                  :=  ReleaseHold[\[ScriptV]FuncFull[[Horizon+1]][m]];
\[ScriptV]Old[m_?NumericQ,TimeToT_]   :=  ReleaseHold[\[ScriptV]FuncFull[[TimeToT+1]][m]];
\[ScriptV]P[m_?NumericQ]                 :=  ReleaseHold[\[ScriptV]FuncFull[[Horizon+1]]'[m]];
\[ScriptV]P[m_?NumericQ,TimeToT_]  :=  ReleaseHold[\[ScriptV]FuncFull[[TimeToT+1]]'[m]];
\[ScriptV]Old[m_?NumericQ]            :=  \[ScriptV]FromNorm[m];
\[ScriptV]Old[m_?NumericQ,TimeToT_]   :=  \[ScriptV]FromNorm[m,TimeToT+1];
\[ScriptV]OldP[m_?NumericQ]           :=  \[ScriptV]FromNormP[m]
\[ScriptV]OldP[m_?NumericQ,TimeToT_]  :=  \[ScriptV]FromNormP[m,TimeToT+1];
*)
(* endCDCPrivate *)
SetAttributes[{\[ScriptC],\[Kappa],\[ScriptV],\[ScriptV]P},Listable];
(* %%% *) SetAttributes[{\[ScriptV]FromvOu},Listable];


\[DoubleStruckCapitalE]Fromat::usage = "\[DoubleStruckCapitalE]Fromat[f_,a_] takes the expectation of the function f where f depends on any of \[CapitalGamma]tp1,\[ScriptCapitalR]tp1,mtp1,btp1.";
\[DoubleStruckCapitalE]Fromat[f_,a_?NumericQ] := Block[{\[CapitalGamma]tp1,\[ScriptCapitalR]tp1,mtp1,btp1}, 
  Sum[
    \[CapitalGamma]tp1 = \[CapitalGamma] \[Psi]Vals[[PermLoop]];
    \[ScriptCapitalR]tp1 = (R)/\[CapitalGamma]tp1;
    btp1 = \[ScriptCapitalR]tp1 a;
    mtp1 = btp1 + \[Theta]Vals[[TranLoop]];
    \[Theta]Probs[[TranLoop]] \[Psi]Probs[[PermLoop]] f[mtp1]
  ,{TranLoop,TranGridLength},{PermLoop,PermGridLength}
  ]
];
SetAttributes[\[DoubleStruckCapitalE]Fromat,{HoldFirst,Listable}];  (* Prevents the function f from being evaluated until it should be *)

(* Calculate the expectation as of the end of period t of a variety of useful objects in period t+1 *)
\[DoubleStruckCapitalE]All[a_?NumericQ] := Block[{\[GothicC]Vec,\[Kappa]Vec,\[GothicC]PVec,\[GothicV]PVec,\[GothicV]PPVec,atp1Vec,mtp1Vec,\[GothicV]Vec},
  {\[GothicV]PVec,\[GothicV]PPVec,atp1Vec,mtp1Vec,\[GothicV]Vec}=
      \[DoubleStruckCapitalE]Fromat[\[ScriptC]Tmp=\[ScriptC][mtp1];\[Kappa]Tmp=\[Kappa][mtp1];{\[Beta] (R) uP[\[CapitalGamma]tp1 \[ScriptC]Tmp],\[Beta] (R) uPP[\[CapitalGamma]tp1 \[ScriptC]Tmp]\[Kappa]Tmp (R),mtp1-\[ScriptC]Tmp,mtp1,\[Beta] \[CapitalGamma]tp1^(1-\[Rho]) \[ScriptV][mtp1]}&,a];
  \[GothicC]Vec  = nP[\[GothicV]PVec];
  \[GothicC]PVec = \[GothicV]PPVec/uPP[\[GothicC]Vec]; 
  \[Kappa]Vec = (\[GothicC]PVec/(1+\[GothicC]PVec));
  Chop[{\[GothicC]Vec,\[Kappa]Vec,atp1Vec,mtp1Vec,\[GothicV]Vec}]
];
SetAttributes[\[DoubleStruckCapitalE]All,Listable];

(* %%% *)
(* %%% *)\[DoubleStruckCapitalE]All[a_?NumericQ,TimeToT_] := Block[{\[GothicC]Vec,\[Kappa]Vec,\[GothicC]PVec,\[GothicV]PVec,\[GothicV]PPVec,atp1Vec,mtp1Vec},
(* %%% *)  {\[GothicV]PVec,\[GothicV]PPVec,atp1Vec,mtp1Vec}
(* %%% *)      =\[DoubleStruckCapitalE]Fromat[{
(* %%% *)          \[Beta] (R) uP[\[CapitalGamma]tp1 \[ScriptC][mtp1,TimeToT]]
(* %%% *)         ,\[Beta] (R) uPP[\[CapitalGamma]tp1 \[ScriptC][mtp1,TimeToT]]\[Kappa][mtp1,TimeToT] (R)
(* %%% *)         ,\[ScriptC][mtp1,TimeToT]-\[Kappa][mtp1,TimeToT]\[ScriptCapitalR]tp1 at
(* %%% *)         ,mtp1-\[ScriptC][mtp1,TimeToT]
(* %%% *)         ,mtp1
(* %%% *)        }&,a];
(* %%% *)  \[GothicC]Vec  = nP[\[GothicV]PVec];
(* %%% *)  \[GothicC]PVec = \[GothicV]PPVec/uPP[\[GothicC]Vec]; 
(* %%% *)  \[Kappa]Vec = (\[GothicC]PVec/(1+\[GothicC]PVec));
(* %%% *)  {\[GothicC]Vec,\[Kappa]Vec,atp1Vec,mtp1Vec}
(* %%% *)];
(* %%% *)

(* See SolnMethEndogGpts document for a brief explanation of the following *)
(* See "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimzation Problems"*)
(* for more detailed explanations *)

(* Marginal and marginal marginal value from ending the period with assets a *)
\[GothicV]P[a_?NumericQ]         := \[Beta] (R) \[DoubleStruckCapitalE]Fromat[ uP[\[CapitalGamma]tp1 \[ScriptC][mtp1]] &,a];
\[GothicV]PP[a_?NumericQ]        := \[Beta] (R) \[DoubleStruckCapitalE]Fromat[uPP[\[CapitalGamma]tp1 \[ScriptC][mtp1]] \[Kappa][mtp1] (R) &,a];
\[GothicV]P[a_?NumericQ,Future_] := \[Beta] (R) \[DoubleStruckCapitalE]Fromat[ uP[\[CapitalGamma]tp1 \[ScriptC][mtp1,Future-1]] &,a];
\[GothicV]PP[a_?NumericQ]        := \[Beta] (R) \[DoubleStruckCapitalE]Fromat[uPP[\[CapitalGamma]tp1 \[ScriptC][mtp1]] \[Kappa][mtp1] (R) &,a];
\[GothicV]PP[a_?NumericQ,Future_]:= \[Beta] (R) \[DoubleStruckCapitalE]Fromat[uPP[\[CapitalGamma]tp1 \[ScriptC][mtp1,Future-1]] \[Kappa][mtp1,Future-1] (R) &,a];

usage::\[GothicC]From\[ScriptA] = "\[GothicC]From\[ScriptA][a] is the 'consumed' function: It reveals the c that must have happened to end the period with assets a.";
\[GothicC]From\[ScriptA][a_?NumericQ]  := If[a == 0,0,nP[\[GothicV]P[a]]];
\[GothicC]PFrom\[ScriptA][a_?NumericQ] := If[a == 0,\[Kappa]Max[[-1]],\[DoubleStruckCapitalE]Fromat[\[Beta] (R) uPP[\[CapitalGamma]tp1 \[ScriptC][mtp1]] \[Kappa][mtp1] (R) &,a]/uPP[\[GothicC]From\[ScriptA][a]]];
SetAttributes[{\[GothicC]From\[ScriptA],\[GothicC]PFrom\[ScriptA],\[GothicV]P,\[GothicV]PP},Listable];
