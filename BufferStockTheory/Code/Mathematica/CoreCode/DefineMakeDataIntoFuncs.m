(* ::Package:: *)

MakeDataIntoFuncs := Block[{},

  If[Mod[Length[\[HBar]],10] == 0 && VerboseOutput==True,Print["Solved ",Length[\[HBar]]," periods."]];

(* Add zeros *)
  mVecIncBot=Prepend[mVecExcBot,0.];
  cVecIncBot=Prepend[cVecExcBot,0.];

(* Limiting consumption and MPCs at the gridpoints *)
  {cLimVecExcBot,\[Kappa]LimVecExcBot} = {\[ScriptC]Lim[mVecExcBot],\[Kappa]Lim[mVecExcBot]};

(* Value and its derivatives *)
  vVecExcBot     = u[  cVecExcBot]+\[GothicV]VecExcBot; (* \[GothicV] is end-of-period value *)
  vPVecExcBot    = uP[ cVecExcBot];            (* Envelope theorem *)
  vPPVecExcBot   = uPP[cVecExcBot]\[Kappa]VecExcBot;  (* Derivative of envelope condition *)

If[Length[FailedPoints=Select[cLimMinusc=cLimVecExcBot-cVecExcBot,#<0 &]] > 0, (* If there are >0 instances where limiting consumption fails to exceed optimal consumption *)
  Print["Limiting consumption vector does not strictly exceed computed optimal consumption vector in period ",Horizon+1];
  Print["Points of failure are: ",FailedPointsPos=Flatten[Map[Position[cLimMinusc,#] &,FailedPoints]]];
  Print[MatrixForm[Transpose[{Table[i,{i,Length[cVecExcBot]}],mVecExcBot,cVecExcBot,cLimVecExcBot,cLimMinusc}],TableHeadings->{None,{"Pos","m","cVec","cLimVec","cLimVec-cVec"}}]];
  Print["This can probably be fixed by reducing the JoinPeriod in ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod, or increasing the number of gridpoints."];
(* %%% *)    cLimFunc=Interpolation[cLimData=Transpose[{mVecExcBot,cLimVecExcBot}],InterpolationOrder->1];
(* %%% *)    cFunc   =Interpolation[cData   =Transpose[{mVecIncBot,   cVecIncBot}],InterpolationOrder->1];
(* %%% *)    nBot = nWherecLimFirstIsBelowc = Position[cLimMinusc,Select[cLimMinusc,#<0 &][[1]]][[1,1]];
(* %%% *)    nTop = nWherecLimLastIsBelowc = Position[cLimMinusc,Select[cLimMinusc,#<0 &][[-1]]][[1,1]];
(* %%% *)    {mBot,mTop} = {mVecExcBot[[nBot-2]],mVecExcBot[[nTop+2]]};
(* %%% *)    cFuncPlot=Plot[{cLimFunc[m],cFunc[m]},{m,mBot,mTop}];
(* %%% *)    cDiffPlot=Plot[{cLimFunc[m]-cFunc[m]},{m,mBot,mTop}];
  Interrupt[]];

  \[Mu]VecExcBot = Log[mVecExcBot];
  \[Chi]VecExcBot = Log[\[CurlyEpsilon]+Chop[1-cVecExcBot/(cLimVecExcBot+\[Theta]Min)]];
  \[Chi]PVecExcBot = (mVecExcBot (cVecExcBot \[Kappa]LimVecExcBot - (cLimVecExcBot + \[Theta]Min) \[Kappa]VecExcBot))/
              ((cLimVecExcBot + \[Theta]Min) (cLimVecExcBot - cVecExcBot + cLimVecExcBot \[CurlyEpsilon] + \[Theta]Min + \[CurlyEpsilon] \[Theta]Min));

  \[Chi]Data  = Transpose[{AddBracesTo[\[Mu]VecExcBot],\[Chi]VecExcBot,\[Chi]PVecExcBot}]; 
(*
(* %%% *) (* Limiting value *)
(* %%% *)   vLimVecExcBot       = \[ScriptV]Lim[mVecExcBot];
(* %%% *)   vLimPVecExcBot      = \[ScriptV]PLim[mVecExcBot];
(* %%% *)   vLimPPVecExcBot     = \[ScriptV]PPLim[mVecExcBot];
*)
(* Normalize value by factor that induces finite limits -- see paper; take the log of the negative because the result approaches log-linear *)
(* To verify formulas, see Derivations.nb *)
  vNormVecExcBot   = Log[-vVecExcBot/(\[Eta]+mVecExcBot^(1-\[Rho]))];
  vPNormVecExcBot  = (vPVecExcBot (mVecExcBot + mVecExcBot^\[Rho] \[Eta]) + vVecExcBot (-1 + \[Rho]))/(vVecExcBot (mVecExcBot + mVecExcBot^\[Rho] \[Eta]));
  vPPNormVecExcBot = (-mVecExcBot vPVecExcBot^2 (mVecExcBot + mVecExcBot^\[Rho] \[Eta])^2 + mVecExcBot vPPVecExcBot vVecExcBot (mVecExcBot + 
      mVecExcBot^\[Rho] \[Eta])^2 - vVecExcBot^2 (-1 + \[Rho]) (mVecExcBot + 
      mVecExcBot^\[Rho] \[Eta] \[Rho]))/(mVecExcBot vVecExcBot^2 (mVecExcBot +
      mVecExcBot^\[Rho] \[Eta])^2);

(* %%% *)  vOuVecExcBot   = vVecExcBot/u[mVecExcBot];
(* %%% *)  vPOuVecExcBot  = (vPVecExcBot*u[mVecExcBot] - vVecExcBot*uP[mVecExcBot])/u[mVecExcBot]^2;
(* %%% *)  vPPOuVecExcBot = (vPPVecExcBot*u[mVecExcBot]^2 - 2 vPVecExcBot*u[mVecExcBot] uP[mVecExcBot] + 2 vVecExcBot uP[mVecExcBot]^2 - 
(* %%% *)                  vVecExcBot*u[mVecExcBot]*uPP[mVecExcBot])/u[mVecExcBot]^3;

(* beginCDCPrivate *)
(*
  vGap0  = First[vGapVecExcBot   = Log[\[CurlyEpsilon]+Chop[1-vLimVecExcBot/vVecExcBot]]];
  vGapP0 = First[vGapPVecExcBot  = ((mVecExcBot*vLimVecExcBot*vPVecExcBot)/vVecExcBot^2 - (mVecExcBot*vLimPVecExcBot)/vVecExcBot)/(1 - vLimVecExcBot/vVecExcBot)];
  vGapPP0= First[vGapPPVecExcBot = (mVecExcBot*(mVecExcBot*vLimVecExcBot^2*vPVecExcBot^2 - 
   vLimVecExcBot*(2*mVecExcBot*vPVecExcBot^2 + 
     vLimVecExcBot*(mVecExcBot*vPPVecExcBot + vPVecExcBot))*
    vVecExcBot + (vLimVecExcBot*(vLimPVecExcBot + 
       mVecExcBot*(vLimPPVecExcBot + vPPVecExcBot) + vPVecExcBot) + 
     mVecExcBot*vLimPVecExcBot*(-vLimPVecExcBot + 2*vPVecExcBot))*
    vVecExcBot^2 - (mVecExcBot*vLimPPVecExcBot + vLimPVecExcBot)*
    vVecExcBot^3))/(vVecExcBot^2*(-vLimVecExcBot + vVecExcBot)^2)];

(* %%% *)  vGapVecOrder1   = Interpolation[Transpose[{AddBracesTo[\[Mu]VecExcBot],vGapVecExcBot  }],InterpolationOrder->1];
(* %%% *)  vGapPVecOrder1  = Interpolation[Transpose[{AddBracesTo[\[Mu]VecExcBot],vGapPVecExcBot }],InterpolationOrder->1];
(* %%% *)  vGapPPVecOrder1 = Interpolation[Transpose[{AddBracesTo[\[Mu]VecExcBot],vGapPPVecExcBot}],InterpolationOrder->1];
*)
(* endCDCPrivate *)

(* %%% *)  vOuVecExcBot   = vVecExcBot/u[mVecExcBot];
(* %%% *)  vPOuVecExcBot  = (vPVecExcBot*u[mVecExcBot] - vVecExcBot*uP[mVecExcBot])/u[mVecExcBot]^2;
(* %%% *)  vPPOuVecExcBot = (vPPVecExcBot*u[mVecExcBot]^2 - 2 vPVecExcBot*u[mVecExcBot] uP[mVecExcBot] + 2 vVecExcBot uP[mVecExcBot]^2 - 
(* %%% *)                  vVecExcBot*u[mVecExcBot]*uPP[mVecExcBot])/u[mVecExcBot]^3;

  AppendTo[\[ScriptC]LinearSplineData,Transpose[{AddBracesTo[mVecIncBot],cVecIncBot}]];
  AppendTo[\[ScriptC]LinearSplineFunc,Interpolation[Last[\[ScriptC]LinearSplineData],InterpolationOrder->1]];

  \[Chi]Interp=Interpolation[\[Chi]Data];

  \[Mu]0=\[Mu]VecExcBot[[1]];

If[Horizon == 0
  ,(*then use simple \[Chi] function*)

  AppendTo[\[Chi]InterpData,\[Chi]Data];
  AppendTo[\[Chi]InterpFunc,Interpolation[\[Chi]InterpData[[-1]]]];

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
  ,(* else use more complex approach *)
(* Problem:  Below the lowest gridpoint, extrapolation misbehaves *)
(* Solution: Extrapolate function to the point where \[Chi]P would be zero, then assume constant \[Chi] below that *)
  {\[Chi]0,\[Chi]P0,\[Chi]PP0,\[Chi]PPP0} = {\[Chi]Interp[\[Mu]0],\[Chi]Interp'[\[Mu]0],\[Chi]Interp''[\[Mu]0],\[Chi]Interp'''[\[Mu]0]};

  \[Mu]GapToPlaceWhere\[Chi]PHitsZero = -\[Chi]P0/\[Chi]PP0;
  \[Mu]Bot=\[Mu]Where\[Chi]PHitsZero = \[Mu]0+\[Mu]GapToPlaceWhere\[Chi]PHitsZero;   
  \[Chi]Bot=\[Chi]Where\[Chi]PHitsZero = \[Chi]0+\[Mu]GapToPlaceWhere\[Chi]PHitsZero \[Chi]P0 + ((\[Mu]GapToPlaceWhere\[Chi]PHitsZero^2)/2)\[Chi]PP0;
  \[Chi]Extrap = Interpolation[
      {
        {{\[Mu]Where\[Chi]PHitsZero},\[Chi]Where\[Chi]PHitsZero}
       ,{{\[Mu]0},\[Chi]0}
      },InterpolationOrder->1
  ]; 

(* Add the newly constructed data and resulting interpolation to the list of solutions *)
  AppendTo[\[Chi]InterpData,\[Chi]Data];
  AppendTo[\[Chi]InterpFunc,Interpolation[\[Chi]InterpData[[-1]]]];
(*  HowFarBackForLastSlope=2;(*IntegerPart[Length[aVecExcBotLarge]/5]+2;*) (* Smooth over any fluctuations in the slope over the last few points *)
  SlopeBeyondLastPoint = (\[Chi]VecExcBot[[-1]]-\[Chi]VecExcBot[[-HowFarBackForLastSlope]])/(\[Mu]VecExcBot[[-1]]-\[Mu]VecExcBot[[-HowFarBackForLastSlope]]);
*)  SlopeBeyondLastPoint = Last[\[Chi]PVecExcBot];
  AppendTo[\[Chi]Full, 
    Evaluate[ (* Evaluate[] causes the expressions like \[Chi]VecExcBot[[1]] to be represented as numbers rather than preserved as variables *)
      Piecewise[{
(*        {\[Chi]0+\[Mu]GapToPlaceWhere\[Chi]PHitsZero \[Chi]P0,# <= \[Mu]Bot}  (* Assume function is constant below the lower bound *)
       ,{\[Chi]0+(#-\[Mu]0)\[Chi]P0  ,      \[Mu]Bot < # < \[Mu]0 }  (* Approximate function obtained by extrapolating \[Chi]Interp to point where its slope would have hit zero *)
*)
       {\[Chi]0   ,       # < \[Mu]0 }  (* Approximate function obtained by extrapolating \[Chi]Interp to point where its slope would have hit zero *)
(*,{\[Chi]InterpFunc[[-1]][\[Mu]VecExcBot[[1]]],# < \[Mu]VecExcBot[[1]]}*)  (* Use the interpolating function for the main body of the function *)
,{\[Chi]InterpFunc[[-1]][#],\[Mu]VecExcBot[[1]] <= # <= \[Mu]VecExcBot[[-1]]} (* Use the interpolating function for the main body of the function *)
(*       ,{\[Chi]VecExcBot[[-1]]+(\[Chi]VecExcBot[[-1]]-(#-\[Mu]VecExcBot[[-1]])SlopeBeyondLastPoint), # > \[Mu]VecExcBot[[-1]]}*)
       ,{(\[Chi]VecExcBot[[-1]]+(#-\[Mu]VecExcBot[[-1]])SlopeBeyondLastPoint), # > \[Mu]VecExcBot[[-1]]}
 (* Linear beyond upper limit with slope matching slope over last interval *)
       }] (* End Piecewise *)
    ] (*End Evaluate*) & 
  ] (* End AppendTo *)
];
(* beginCDCPrivate *)
  vData       = Transpose[{AddBracesTo[mVecExcBot],vVecExcBot,vPVecExcBot,vPPVecExcBot}];
(* %%% *)  vOuData     = Transpose[{AddBracesTo[mVecExcBot],vOuVecExcBot,vPOuVecExcBot,vPPOuVecExcBot}];
  AppendTo[\[ScriptV]InterpData  ,vData];
(* %%% *)  AppendTo[vOuInterpData,vOuData];
  AppendTo[\[ScriptV]InterpFunc  ,Interpolation[Last[\[ScriptV]InterpData]]];
(* %%% *)  AppendTo[vOuInterpFunc,Interpolation[Last[vOuInterpData]]];

  {mInterpMin,mInterpMax} = {\[ScriptV]InterpData[[-1]][[1,1,1]],\[ScriptV]InterpData[[-1]][[-1,1,1]]};
  AppendTo[\[ScriptV]FuncFull,\[ScriptV]FuncNow = Evaluate[
      Piecewise[{
        {\[ScriptV]InterpData[[-1]][[1,2]]+Hold[NIntegrate[uP[\[ScriptC][\[Bullet]]],{\[Bullet],mInterpMin, #}]], # < mInterpMin}
       ,{\[ScriptV]InterpFunc[[-1]][#], mInterpMin <= # <= mInterpMax}
       ,{Last[vVecExcBot]+Hold[NIntegrate[uP[\[ScriptC][\[Bullet]]],{\[Bullet],mInterpMax, #}]], mInterpMax <= # }
      }] ](* End Piecewise *) &
  ];
(* %%% *)  AppendTo[vOuFuncFull,vOuFuncNow = Evaluate[
(* %%% *)      Piecewise[{
(* %%% *)(*        {vOuInterpData[[-1]][[1,2]]+Hold[NIntegrate[uP[\[ScriptC][\[Bullet]]],{\[Bullet],vOuInterpData[[-1]][[1,1,1]], #}]], # < vOuInterpData[[-1]][[1,1,1]]},*)
(* %%% *)         {vOuInterpFunc[[-1]][#], (*vOuInterpData[[-1]][[-1,1,1]] >= *)# >= 0. vOuInterpData[[-1]][[1,1,1]]}
(* %%% *)(*       ,{Last[vOuVecExcBot]u[Last[mVecExcBot]]+Hold[NIntegrate[uP[\[ScriptC][\[Bullet]]],{\[Bullet],vOuInterpData[[-1]][[-1,1,1]], #}]], # >= vOuInterpData[[-1]][[-1,1,1]]}*)
(* %%% *)      }] ](* End Piecewise *) &
(* %%% *)  ];
(* endCDCPrivate *)

  vNormData       = Transpose[{AddBracesTo[mVecExcBot],vNormVecExcBot,vPNormVecExcBot,vPPNormVecExcBot}];

  AppendTo[vNormInterpData,vNormData];
  AppendTo[vNormInterpFunc,Interpolation[vNormInterpData[[-1]]]];
(* beginCDCPrivate *)
  vNormSlopeToBot      = (First[vNormVecExcBot](*-Log[-u[\[Kappa]MaxInf]]*))/First[mVecExcBot];
  \[Kappa]ToBot = First[cVecExcBot]/First[mVecExcBot];
  vNormSlopeAboveTop = Last[vPNormVecExcBot];
  {mInterpMin,mInterpMax}={Last[vNormInterpData][[1,1,1]],Last[vNormInterpData][[-1,1,1]]};
  AppendTo[vNormFuncFull,
   vNormFuncNow = 
    Evaluate[
      Piecewise[{
        {
        Log[-(vVecExcBot[[1]]+(u[\[Kappa]ToBot #]-u[\[Kappa]ToBot mInterpMin]))/(\[Eta]+#^(1-\[Rho]))]
           , # < mInterpMin}
       ,{vNormInterpFunc[[-1]][#]
           , mInterpMin <= # <= mInterpMax}
       ,{Last[vNormVecExcBot]+vNormSlopeAboveTop (# - Last[mVecExcBot])
           , mInterpMax <= #}
      }] (* End Piecewise *)
    ] (* End Evaluate *) &;
  ]; (* End AppendTo *)
(* endCDCPrivate *)

(* beginCDCPrivate *)
(*#
  {vNormLast,vPNormLast,vPPNormLast,vPPPNormLast} = {Last[vNormVecExcBot],Last[vPNormVecExcBot],Last[vPPNormVecExcBot],vNormInterpFunc[[-1]]'''[Last[mVecExcBot]]};
  mGapToPlaceWherevPPNormHitsZero = -vPPPNormLast/vPPNormLast;
  mWherevPPNormHitsZero = Last[mVecExcBot]+mGapToPlaceWherevPPNormHitsZero;   
  vNormWherevPPNormHitsZero = vNormLast+mGapToPlaceWherevPPNormHitsZero vPNormLast;
(* %%% *)  vNormExtrap2 = Interpolation[
(* %%% *)      {
(* %%% *)        {{Last[mVecExcBot]},vNormLast}
(* %%% *)       ,{{mWherevPPNormHitsZero},vNormWherevPPNormHitsZero}
(* %%% *)      },InterpolationOrder->1
(* %%% *)  ]; 
*)
(* endCDCPrivate *)

  AppendTo[\[DoubleStruckCapitalE]atp1InterpFunc,Interpolation[Transpose[{AddBracesTo[aVecExcBot],\[DoubleStruckCapitalE]atp1Froma}],InterpolationOrder->1]];
  AppendTo[\[DoubleStruckCapitalE]mtp1InterpFunc,Interpolation[Transpose[{AddBracesTo[Rest[mVecIncBot]],\[DoubleStruckCapitalE]mtp1Froma}],InterpolationOrder->1]];

(* beginCDCPrivate *)
(*
  AppendTo[vGapInterpData,Transpose[{AddBracesTo[\[Mu]VecExcBot],vGapVecExcBot,vGapPVecExcBot,vGapPPVecExcBot}]];
  AppendTo[vGapInterpFunc,Interpolation[Last[vGapInterpData]]];
*)
(*
  \[Mu]GapToPlaceWherevGapPHitsZero = -vGapP0/vGapPP0;
  \[Mu]WherevGapPHitsZero = \[Mu]0+\[Mu]GapToPlaceWherevGapPHitsZero;   
  vGapWherevGapPHitsZero = vGap0+\[Mu]GapToPlaceWherevGapPHitsZero vGapP0 +((\[Mu]GapToPlaceWherevGapPHitsZero^2)/2) vGapPP0;
  vGapExtrap = Interpolation[
      {
        {{\[Mu]0},vGap0,vGapP0}
       ,{{\[Mu]WherevGapPHitsZero},vGapWherevGapPHitsZero,0.}
      },InterpolationOrder->1
  ]; 
  vGapPLast = Last[vGapInterpFunc]'[Last[\[Mu]VecExcBot]];
  AppendTo[vGapFull,vGapFuncNow = 
    Evaluate[
      Piecewise[{
        {vGapExtrap[\[Mu]WherevGapPHitsZero],# <= \[Mu]WherevGapPHitsZero}
       ,{vGapExtrap[#], \[Mu]WherevGapPHitsZero < # < \[Mu]0}
       ,{vGapInterpFunc[[-1]][#], Last[\[Mu]VecExcBot] >= # >=  \[Mu]0}
       ,{Last[vGapVecExcBot]+vGapPLast (# - Last[\[Mu]VecExcBot]), # >=  Last[\[Mu]VecExcBot]}
      }] (* End Piecewise *)
    ] (* End Evaluate *) &
  ]; (* End AppendTo *)
*)
(* endCDCPrivate *)
]; (* End AddNewDataAndFuncsToList *)
