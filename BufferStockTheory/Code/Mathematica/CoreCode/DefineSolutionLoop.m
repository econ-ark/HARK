(* ::Package:: *)

SolveAnotherPeriod::usage = "SolveAnotherPeriod constructs the solution for period t given the ambient existence of a solution for period t+1";
SolveAnotherPeriod := Block[{
(* %%% *) (*a,cVecIncBot,cVecExcBot,\[Kappa]VecIncBot,\[Kappa]VecExcBot,mVecIncBot,mVecExcBot,\[DoubleStruckCapitalE]atp1Froma,\[DoubleStruckCapitalE]atp1FromaVecExcBot,\[DoubleStruckCapitalE]mtp1FromaVecExcBot,\[DoubleStruckCapitalE]mtp1Froma*)
},
  aVecAugmentWithPointsNearKink; (* Adding some points near the kink (if any) increases accuracy considerably *)

(* If minimum tran shock \[Theta] is positive, the problem has a liquidity constraint, so there is a well-defined expectation for marginal value even for a = 0. *)
  If[\[Theta]Min > 0.,(*then*)PrependTo[aVecExcBot,0.]];

(* Construct expectation of next period's situation from this period's ending points aVecExcBot *)
 {cVecExcBot,\[Kappa]VecExcBot,\[DoubleStruckCapitalE]atp1Froma,\[DoubleStruckCapitalE]mtp1Froma,\[GothicV]VecExcBot}=
     Transpose[\[DoubleStruckCapitalE]All[aVecExcBot]]; (* Computing expectations of a set of things is much faster than one-by-one *)

(* Endogenous gridpoints method (Google "Carroll Endogenous Gridpoints") yields m values from c and a values *)
  mVecExcBot  = aVecExcBot+cVecExcBot;
  
(* Construct limiting minimum and maximum MPC's as wealth goes to \[Infinity] and 0., and human wealth *)
  AppendTo[\[HBar]   ,1+FHW \[HBar][[-1]]]; (* Recursive formula for human wealth ratio \[HBar] *)
  AppendTo[\[Kappa]Max,1/(1+\[Lambda]Below/Last[\[Kappa]Max])]; (* eq:MaxMPCInv in BufferStockTheory.tex *)
  AppendTo[\[Kappa]Min,1/(1+\[Lambda]Above/Last[\[Kappa]Min])]; (* eq:MinMPCInv in BufferStockTheory.tex *)

  MakeDataIntoFuncs; (* Given the vectors constructed above, add appropriate data and functions to ambient environment *)

  Horizon=Length[\[HBar]]-1; (* Finished constructing the solution to the current period, so update status of Horizon variable *)

  If[HorizonLimit == Inf,AppendNewTargets]; (* Calculating a target makes sense only for the inf horiz version *)

  AugmentGridWhere\[Kappa]ChangesMost (* Adding points at 'most curved' part increases accuracy just where it is needed *)

];  (* End SolveAnotherPeriod *)


aVecAugmentWithPointsNearKink := 
  If[NumericQ[mWhere\[ScriptC]LimHasKink],  (* if mWhere\[ScriptC]LimHasKink is a numerical variable, then the limiting function has a kink at the corresponding point *)
    (* Augment the aVec with points that generate an m around the place where the limiting function has its kink *)
    aWhere\[ScriptC]LimKinkBitesSearch = Reap[FindRoot[\[GothicC]From\[ScriptA][aSeek]+aSeek == mWhere\[ScriptC]LimHasKink,{aSeek,aTarget},StepMonitor:>Sow[aSeek]]];
    aAtKink = aSeek /. aWhere\[ScriptC]LimKinkBitesSearch[[1]];  
    aTries  = aWhere\[ScriptC]LimKinkBitesSearch[[PosOfListOfSearchPoints=2,1]];  (* List of points converging to solution *)
    aPointsNearKink = Select[Union[aTries,aAtKink+aAtKink-aTries],#>0.&];(* Produce symmetric set of 'close' points around solution *)
    aVecExcBot = Union[aGridVecExcBot,aPointsNearKink]
 ]; (* End If[NumericQ] *)

(* Now add a point between the two points where the difference in MPC's is greatest (if that difference exceeds a threshold)  *)
(* This makes the grid adapt to have greater density at places where the function is most curved *)
AugmentGridWhere\[Kappa]ChangesMost := Block[{},
  \[Kappa]Diffs=Abs[Differences[\[Kappa]VecExcBot]];
(* %%% \[Kappa]Num = Table[n,{n,Length[\[Kappa]Diffs]}];\[Kappa]Table=Transpose[{\[Kappa]Num,\[Kappa]Diffs}];*)
  \[Kappa]DiffMaxPos = Position[\[Kappa]Diffs,Max[\[Kappa]Diffs]][[1,1]];
  aGridVecExcBotPosVec=Position[aGridVecExcBot,aVecExcBot[[\[Kappa]DiffMaxPos]]];
  If[Length[aGridVecExcBotPosVec]>0,(*then*) aGridVecExcBotPos = aGridVecExcBotPosVec[[1,1]]];
  If[\[Kappa]Diffs[[\[Kappa]DiffMaxPos]] > 0.05 && Length[aGridVecExcBotPosVec]>0, (* If MPC differs by more than x, insert new point *)
    If[\[Kappa]DiffMaxPos > 1
      ,(*then*) aGridVecExcBot=Union[Insert[aGridVecExcBot,(aGridVecExcBot[[aGridVecExcBotPos+1]]+aGridVecExcBot[[aGridVecExcBotPos]])/2,aGridVecExcBotPos+1]]
      ,(*else*) PrependTo[aGridVecExcBot,aGridVecExcBot[[1]]/2]] (* Higest \[CapitalDelta]\[Kappa] was below first gridpoint *)
  ];
];

AppendNewTargets := Block[{},
    {aTarget,mTarget,cTarget,\[Kappa]Target,bTarget}=FindTargets[CoarseAccuracy,Horizon];
    AppendTo[aTargetList,aTarget];
    AppendTo[cTargetList,cTarget];
    AppendTo[\[Kappa]TargetList,\[Kappa]Target];
    AppendTo[bTargetList,bTarget];
    AppendTo[mTargetList,mTarget];
];


SolveToToleranceOf[mTargetTol_] := Block[{mTargetPrev,mTargetNext},
  {mTargetPrev,mTargetNext} = {mTarget,0.};
  While[Abs[mTargetPrev-mTargetNext]>mTargetTol,
    mTargetPrev = mTarget;
    SolveAnotherPeriod; (* New targets are found when period is solved *)
    mTargetNext = mTarget;
    If[VerboseOutput==True,Print[mTargetPrev-mTargetNext," is latest tolerance for target change."]]
  ]; (* end While *)
  If[VerboseOutput==True,Print["Converged."]];
];

SolveInfHorizToToleranceAtTarget[mTargetTolerance_] := Block[{},
  ConstructLastPeriodAsSmoothedInfHorPFLiqConstrSolnJoinedAtPeriod[2];
  (* Solve for successively finer approximations to the shock distribution *)
  (* This is much faster than solving for fine approximations all the way from the start, *)
  (* and ultimately just as accurate *)

  PermGridLength=TranGridLengthSetup=1;SetupShocks;SolveAnotherPeriod;
  SolveToToleranceOf[mTargetTolerance/8];

  PermGridLength=TranGridLengthSetup=3;SetupShocks;SolveAnotherPeriod;
  SolveToToleranceOf[mTargetTolerance/4];
  
  PermGridLength=TranGridLengthSetup=7;SetupShocks;SolveAnotherPeriod;
  SolveToToleranceOf[mTargetTolerance/2];
  
  PermGridLength=TranGridLengthSetup=13;SetupShocks;SolveAnotherPeriod;
  SolveToToleranceOf[mTargetTolerance];
  
  ModelIsSolved=True
];

FindTargets[DesiredAccuracy_,Horizon_] := Block[{},
  If[Horizon == 0, Return[{aTarget=0.,mTarget=1.,cTarget=1.,\[Kappa]Target=1.,bTarget=0.}]];
  If[\[CapitalThorn]\[CapitalGamma] > 1,Print["Growth Impatience Condition does not hold; no target exists."];Abort[]];

  (* Suppress some unimportant error messages *) Off[InterpolatingFunction::dmval];
  (* %%% *) On[InterpolatingFunction::dmval]; (* Keep the messages on for CDCPrivate version *)
  (* The approximate solutions use the cheaply-computed interpolations \[DoubleStruckCapitalE]mtp1 and \[DoubleStruckCapitalE]atp1 constructed above *)
  (* The final solutions use those approximate solutions as starting points to find a more accurate answer *)

  aTargetApprox = aSeek /. FindRoot[\[DoubleStruckCapitalE]atp1InterpFunc[[Horizon]][aSeek] ==  aSeek,{aSeek,0.01}];
  If[aTargetApprox <= 0., aTargetApprox = 0.01];
  aTarget       = Chop[ aSeek /. FindRoot[\[DoubleStruckCapitalE]Fromat[mtp1-\[ScriptC][mtp1,Horizon-1] &,aSeek] == aSeek,{aSeek,aTargetApprox}(*,AccuracyGoal->DesiredAccuracy*)]];
  mTargetApprox = mSeek /. FindRoot[\[DoubleStruckCapitalE]mtp1InterpFunc[[Horizon]][mSeek-\[ScriptC][mSeek]] ==  mSeek,{mSeek,1+aTargetApprox}];
  If[mTargetApprox <= 1., mTargetApprox = 1.01];
  mTarget       = Chop[mSeek /. FindRoot[\[DoubleStruckCapitalE]Fromat[mtp1 &,mSeek-\[ScriptC][mSeek,Horizon]] == mSeek,{mSeek,mTargetApprox}(*,AccuracyGoal->DesiredAccuracy*)]];
  {bTarget,cTarget,\[Kappa]Target} = {mTarget-1,\[ScriptC][mTarget],\[Kappa][mTarget]};
  On[InterpolatingFunction::dmval]; (* Restore warnings for other contexts *)
  Return[{aTarget,mTarget,cTarget,\[Kappa]Target,bTarget}]
];

FindTargets[DesiredAccuracy_] := FindTargets[DesiredAccuracy,Horizon];  
