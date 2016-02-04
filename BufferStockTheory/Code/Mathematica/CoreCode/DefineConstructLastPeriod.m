(* ::Package:: *)

ConstructLastPeriod := Block[{},
(* %%% *)  ModelIsSolved=False; (* If we're just setting up the last period, the model obviously has not been solved *)
(* Rationales for terminal functions are implicit in the appendix on the liquidity constrained solution *)

(* Implement terminal/limiting function consistent with results in ApndxLiqConstrScenariosStandAlone.tex *)
If[PFGICFails && RICFails(* Growth patient and return patient *)
  ,(*then*)
   Print["Limiting perfect foresight unconstrained solution is \[ScriptC]\[Digamma][m]=\[Infinity], which can't be used as upper bound."];
   ConstructLastPeriodToConsumeEverything;
   Print["Last period has been set up as \!\(\*SubscriptBox[\"\[ScriptC]\", \"T\"]\)[m]=m."];
  ,(*else*)
  If[PFGICFails 
    ,(*then*) ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod[3] (* Nondegenerate limiting solution exists; see appendix for logic *)
    ,(*else PFGIC holds *)
    If[RICHolds
      ,(*then*) ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod[3]
      ,(*else*) ConstructLastPeriodToConsumeEverything (* MPC asymptotes to zero so is not well approximated by anything *)
      ] (* End If RICHolds *)
    ]  (* End If PFGICFails *)
  ] (* End If PFGICValue > 1 && RICValue >= 1 *)
]; (* End ConstructLastPeriod *)

(* See ApndxSolnMethTermFunc.tex for explanation of limting/terminal function *)
ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod[JoinPeriod_] := Block[{},
  HorizonLimit=Inf; (* Indicator for If[] statements later in code; alternative value would be "Fin" for "Finite" *)
  SetupCommon;     (* Set up elements that are common for all terminal conditions *)
  MakeSmoothApproxToLiqConstrSoln[JoinPeriod];  (* For ambient parameter values, construct liquidity constrained solution *)

(* Set human wealth list \[HBar] to value that depends on whether human wealth is infinite or not *)
  If[FHWCHolds (* If infinite-horizon human wealth is finite *)
   ,(* then *) \[HBar]={1/(1-FHW)}
   ,(* else *) \[HBar]={Infinity}];

  {\[Kappa]Max,\[Kappa]Min}={{\[Kappa]MaxInf},{\[Kappa]MinInf}}; (* Lists of minimal and maximal MPC's by period (from end) *)
  If[\[Theta]Min > 0., \[Kappa]Max[[1]] = 1.];     (* If positive minimum income, then the MPC will be 1 below the kink *)

(* Set limiting solution to (adjusted) liquidity constrained solution *)
  \[ScriptC]Lim[m_,TimeToT_] := \[ScriptC]Liq[m];
  \[ScriptC]Lim[m_         ] := \[ScriptC]Liq[m];
  \[Kappa]Lim[m_,TimeToT_] := \[Kappa]Liq[m];
  \[Kappa]Lim[m_]          := \[Kappa]Liq[m];

(* Time periods to future date when assets of aGridVecExcBot[[-1]] would unconstrainedly lead to today's c=m=1 *)
  nFromLasta = IntegerPart[1+nFromLasta\[Sharp][Last[aGridVecExcBot]]];
(* All dates from now to then *)
  nList = Table[n,{n,0,nFromLasta,1}];
(* Corresponding values of m, c, a, and \[Kappa] *)
  mVecExcBot = m\[Sharp][nList];
  cLimVecExcBot = cVecExcBot    = \[ScriptC]Lim[mVecExcBot];
  aVecExcBot = mVecExcBot-cVecExcBot;
  aVecExcBot[[1]] = -\[CurlyEpsilon];  (* Prevents error condition that would be caused by duplicate 0. values in aVec *)
  \[Kappa]VecExcBot    = \[Kappa]LimVecExcBot = \[Kappa]Lim[mVecExcBot];

(* %%% *)  mVecIncBotData    = {mVecIncBot=Join[{0.},               mVecExcBot]};
(* Setup interpolating points for value function approximation; v=Value, vP=v', vPP=v'' *)

(* Value function for terminal period is constructed using envelope condition *)
  vLast = v\[Sharp][n\[Sharp][Last[mVecExcBot]]]; (* Picks a level for the function to match PF Liq Constr level at last point *)
(* Constructed value obtained by numerical integration of u'[\[ScriptC][m]] because Envelope Theorem says it is equal to \[ScriptV]'[m] *)
  vAdd  = Table[NIntegrate[uP[\[ScriptC][m]],{m,mVecExcBot[[i]],mVecExcBot[[i+1]]}],{i,Length[mVecExcBot]-1}];
(* Create v values by accumulating the integrated increments downward from last value *)
  \[GothicV]VecExcBot   = Append[vLast-Reverse[Accumulate[Reverse[vAdd]]],vLast]-u[cVecExcBot];

  \[DoubleStruckCapitalE]btp1Froma = aVecExcBot (R)/\[CapitalGamma];
  \[DoubleStruckCapitalE]mtp1Froma = \[DoubleStruckCapitalE]btp1Froma + 1;
  \[DoubleStruckCapitalE]ctp1Froma = \[ScriptC]Lim[\[DoubleStruckCapitalE]mtp1Froma];
  \[DoubleStruckCapitalE]atp1Froma = \[DoubleStruckCapitalE]mtp1Froma-\[DoubleStruckCapitalE]ctp1Froma;

(* beginCDCPrivate *)
(*
(*  cVecExcBot[[-1]]=c\[Sharp][n\[Sharp][mVecExcBot[[-1]]]];*)

(* Set up \[Chi] *)
(* %%% # delete because done in MakeDataIntoFuncs.m *)    \[Chi]VecExcBot = Log[\[CurlyEpsilon]+Chop[1-cVecExcBot/(cLimVecExcBot+\[Theta]Min)]];
(* %%% # delete because done in MakeDataIntoFuncs.m *)    \[Chi]PVecExcBot = (mVecExcBot (cVecExcBot \[Kappa]LimVecExcBot - (cLimVecExcBot + \[Theta]Min) \[Kappa]VecExcBot))/
(* %%% # delete because done in MakeDataIntoFuncs.m *)                ((cLimVecExcBot + \[Theta]Min) (cLimVecExcBot - cVecExcBot + cLimVecExcBot \[CurlyEpsilon] + \[Theta]Min + \[CurlyEpsilon] \[Theta]Min));
(* %%% # delete because done in MakeDataIntoFuncs.m *)  \[Mu]VecExcBot = Log[mVecExcBot];

(* %%% # delete because done in MakeDataIntoFuncs.m *)  \[Chi]Data = Transpose[{Transpose[{Log[mVecExcBot]}],\[Chi]VecExcBot,\[Chi]PVecExcBot}];
  \[Chi]Data  = Transpose[{AddBracesTo[\[Mu]VecExcBot],\[Chi]VecExcBot,\[Chi]PVecExcBot}]; 
(* For accurate approximation, take the log of the \[Digamma]-bounded values *)
  vNormVecExcBot   = Log[-vVecExcBot/(\[Eta]+mVecExcBot^(1-\[Rho]))];
  vPNormVecExcBot  = (mVecExcBot vPVecExcBot - vVecExcBot + mVecExcBot^\[Rho] vPVecExcBot \[Eta] + vVecExcBot \[Rho])/(mVecExcBot vVecExcBot + mVecExcBot^\[Rho] vVecExcBot \[Eta]);
  vPPNormVecExcBot = (-mVecExcBot vPVecExcBot^2 (mVecExcBot+mVecExcBot^\[Rho] \[Eta])^2+mVecExcBot vPPVecExcBot vVecExcBot (mVecExcBot+mVecExcBot^\[Rho] \[Eta])^2-vVecExcBot^2 (-1+\[Rho]) (mVecExcBot+mVecExcBot^\[Rho] \[Eta] \[Rho]))/(mVecExcBot vVecExcBot^2 (mVecExcBot+mVecExcBot^\[Rho] \[Eta])^2);

  AppendTo[\[ScriptC]LinearSplineData,Transpose[{AddBracesTo[mVecIncBot],cVecIncBot}]];
  AppendTo[\[ScriptC]LinearSplineFunc,Interpolation[Last[\[ScriptC]LinearSplineData],InterpolationOrder->1]];

  AppendTo[\[Chi]InterpData,\[Chi]Data];
  AppendTo[\[Chi]InterpFunc,Interpolation[Last[\[Chi]InterpData],InterpolationOrder->1]];

  (* Extract various features of the extrapolation, for convenient use below *)
  {mInterpMin,mInterpMax,\[Chi]PInterpLast}={\[Chi]InterpData[[-1]][[1,1,1]],\[Chi]InterpData[[-1]][[-1,1,1]],\[Chi]InterpData[[-1]][[-1,3]]};

  AppendTo[\[Chi]Full, (*# \[Chi]FullNow =  *)
    Evaluate[
      Piecewise[{
        {\[Chi]InterpData[[-1]][[1,2]]                            ,# < mInterpMin}  (* Assume \[Chi] is constant below the lower bound *)
       ,{\[Chi]InterpFunc[[-1]][#]                                ,    mInterpMin <= # <= mInterpMax}
       ,{\[Chi]InterpData[[-1]][[-1,2]]+(#-mInterpMax)\[Chi]PInterpLast,                       mInterpMax < #} (* Linear beyond upper limit with slope matching slope at last point *)
     }] (* End Piecewise *)
    ] (*End Evaluate*) &];

  vData       = Transpose[{AddBracesTo[mVecExcBot],vVecExcBot,vPVecExcBot,vPPVecExcBot}];
  AppendTo[\[ScriptV]InterpData,vData];
  AppendTo[\[ScriptV]InterpFunc,Interpolation[Last[\[ScriptV]InterpData]]];

  {mInterpMin,mInterpMax}={\[ScriptV]InterpData[[-1]][[1,1,1]],\[ScriptV]InterpData[[-1]][[-1,1,1]]};
(* beginCDCPrivate *)
(*
  \[Kappa]Bot = First[cVecExcBot]/First[mVecExcBot]; (* MPC assumed constant to first gridpoint *)
  \[ScriptV]FuncFull = {\[ScriptV]FuncNow = Evaluate[
      Piecewise[{
        {\[ScriptV]InterpData[[-1]][[1,2]]+(u[# \[Kappa]Bot]-u[1 \[Kappa]Bot])/\[Kappa]Bot, # < mInterpMin} (* Conveniently, u[\[Kappa] m]/\[Kappa] = Integrate[u'[\[Kappa] m]] *)
       ,{\[ScriptV]InterpFunc[[-1]][#]                     ,      mInterpMin <= # <= mInterpMax}
       ,{Hold[v\[Sharp][n\[Sharp][#]]]                          ,                         mInterpMax < # }
      }] ](* End Piecewise *) &
  };

  vNormData       = Transpose[{AddBracesTo[mVecExcBot],vNormVecExcBot,vPNormVecExcBot,vPPNormVecExcBot}];
(*#  vOuOfcData         = Transpose[{AddBracesTo[mVecExcBot],vOuOfcVecExcBot,vOuOfcPVecExcBot}];*)
  AppendTo[vNormInterpData,vNormData];
(*#  AppendTo[vOuOfcInterpData,vOuOfcData];*)
  AppendTo[vNormInterpFunc,Interpolation[Last[vNormInterpData]]];
(*#  AppendTo[vOuOfcInterpFunc  ,Interpolation[vOuOfcInterpData[[-1]]]];*)

  vNormSlopeToBot     = First[vNormVecExcBot]/First[mVecExcBot](*-Log[-u[\[Kappa]MaxInf]]*);
  \[Kappa]ToBot = First[cVecExcBot]/First[mVecExcBot];
  vNormSlopeAboveTop = (vNormVecExcBot[[-1]]-vNormVecExcBot[[-2]])/(mVecExcBot[[-1]]-mVecExcBot[[-2]]);
  {mInterpMin,mInterpMax}={Last[vNormInterpData][[1,1,1]],Last[vNormInterpData][[-1,1,1]]};

  AppendTo[vNormFuncFull,
    (*# %%% vNormFuncNow = *)
    Evaluate[
      Piecewise[{
        {Log[-(vVecExcBot[[1]]+(u[\[Kappa]ToBot #]-u[mInterpMin \[Kappa]ToBot]))/(\[Eta]+#^(1-\[Rho]))]   , # < mInterpMin}
       ,{vNormInterpFunc[[-1]][#]                                  ,     mInterpMin <= # <=  mInterpMax}
       ,{Last[vNormVecExcBot]+vNormSlopeAboveTop (# - Last[mVecExcBot]),  mInterpMax < #}
      }] (* End Piecewise *)
    ] (* End Evaluate *) & 
      (*# %%% End vNormFuncNow = *)
  ]; (* End AppendTo *)

*)(* beginCDCPrivate *)
  AppendTo[vOuFuncFull,{}]; (* End AppendTo *)
  AppendTo[vOuInterpData,{}]; (* End AppendTo *)
  AppendTo[vOuInterpFunc,{}]; (* End AppendTo *)
(*#
  vOuOfcSlopeToBot      = First[vOuOfcVecExcBot]/First[mVecExcBot](*-Log[-u[\[Kappa]MaxInf]]*);
  vOuOfcSlopeToBot      = 1.;
  vOuOfcSlopeAboveTop = Last[vOuOfcPVecExcBot];
  AppendTo[vOuOfcFuncFull,vOuOfcFuncNow = 
    Evaluate[
      Piecewise[{
        {vOuOfcSlopeToBot # + vOuOfcInterpData[[-1]][[1,2]]-vOuOfcSlopeToBot, # < vOuOfcInterpData[[-1]][[1,1,1]]}
       ,{vOuOfcInterpFunc[[-1]][#], vOuOfcInterpData[[-1]][[-1,1,1]] >= # >=  vOuOfcInterpData[[-1]][[1,1,1]]}
       ,{Last[vOuOfcVecExcBot]+vOuOfcSlopeAboveTop (# - Last[mVecExcBot]), # >=  vOuOfcInterpData[[-1]][[-1,1,1]]}
      }] (* End Piecewise *)
    ] (* End Evaluate *) & (* End vOuOfcFuncNow = *)
   ]; (* End AppendTo *)
(* %%% *)   \[ScriptV]Norm2InterpData = {Transpose[{AddBracesTo[mVecExcBot],v\[Sharp][nList]/u[mVecExcBot]}]};
(* %%% *)   \[ScriptV]Norm2InterpFunc = {Interpolation[\[ScriptV]Norm2InterpData[[-1]],InterpolationOrder->1]};
*)
  \[DoubleStruckCapitalE]atp1InterpFunc = {0. &}; (* Not right, but doesn't matter because never used*)
  \[DoubleStruckCapitalE]mtp1InterpFunc = {1. &}; 

(* %%% *)AppendTo[vGapFull,{}];
(* %%% *)AppendTo[vGapInterpData,{}];
(* %%% *)AppendTo[vGapInterpFunc,{}];
*)
(* endCDCPrivate *)

MakeDataIntoFuncs;
]; (* End ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod *)

ConstructLastPeriodToConsumeEverything := Block[{},
  HorizonLimit=Fin;
  SetupCommon;     (* Set up elements that are common for all terminal conditions *)

(* Set human wealth and MPC lists *)
  \[HBar]=\[Kappa]Max=\[Kappa]Min={1.};

(* Set limiting solution to the finite horizon perfect foresight solution *)
  \[ScriptC]Lim[m_,TimeToT_] := \[ScriptC]\[Digamma]Fin[m,TimeToT];
  \[ScriptC]Lim[m_]                := \[ScriptC]\[Digamma]Fin[m,Length[\[HBar]]-1]; (* Minus 1 because with 0 future periods Length[\[HBar]]=1 *)
  \[Kappa]Lim[m_,TimeToT_] := \[Kappa]Min[[TimeToT+1]];
  \[Kappa]Lim[m_]                := \[Kappa]Min[[Length[\[HBar]]]];
  \[ScriptV]Lim[m_]  := u[m];
  \[ScriptV]PLim[m_] := uP[m];
  \[ScriptV]PPLim[m_]:= uPP[m];
  SetAttributes[{\[ScriptV]Lim,\[ScriptV]PLim,\[ScriptV]PPLim,\[ScriptC]Lim,\[Kappa]Lim},Listable];
  SetupGrids;
(* Set up points where c = m = a from whatever aGrid is ambient *)
  aVecExcBot   =mVecExcBot = AppendTo[aGridVecExcBot,aGridVecExcBot[[-1]]+1];
  cLimVecExcBot=cVecExcBot = \[ScriptC]Lim[mVecExcBot];
  \[Kappa]VecExcBot  = \[Kappa]Lim[mVecExcBot];
  \[GothicV]VecExcBot   = 0. cVecExcBot;

  \[DoubleStruckCapitalE]btp1Froma = aVecExcBot (R)/\[CapitalGamma];
  \[DoubleStruckCapitalE]mtp1Froma = \[DoubleStruckCapitalE]btp1Froma + 1;
  \[DoubleStruckCapitalE]ctp1Froma = \[ScriptC]Lim[\[DoubleStruckCapitalE]mtp1Froma];
  \[DoubleStruckCapitalE]atp1Froma = 0. (\[DoubleStruckCapitalE]mtp1Froma-\[DoubleStruckCapitalE]ctp1Froma); (* Spend everything if you survive death *)

MakeDataIntoFuncs;

];

(* SetupCommon initializes variables that are needed regardless of what the terminal function is *)
SetupCommon := Block[{},
  Horizon=0; (* Number of periods already solved *)
  \[Eta]=0; (* value-function normalizing constant; works for zero *)
  \[DoubleStruckCapitalE]atp1InterpFunc=\[DoubleStruckCapitalE]mtp1InterpFunc=vGapFull=vGapInterpFunc=vGapInterpData=vOuFuncFull=vOuInterpData=vOuInterpFunc=\[ScriptV]InterpData=\[ScriptV]InterpFunc=\[ScriptV]FuncFull=\[Chi]Full=\[Chi]InterpData=\[Chi]InterpFunc=\[HBar]=vNormFuncFull = vNormInterpFunc = vNormInterpData=\[ScriptC]LinearSplineData=\[ScriptC]LinearSplineFunc=aTargetList = cTargetList = \[Kappa]TargetList = bTargetList = mTargetList = {};
  AppendTo[aTargetList,aTarget=0.];
  AppendTo[cTargetList,cTarget=1.];
  AppendTo[\[Kappa]TargetList,\[Kappa]Target=1.];
  AppendTo[bTargetList,bTarget=0.];
  AppendTo[mTargetList,mTarget=1.];
];


