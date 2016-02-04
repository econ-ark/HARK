(* ::Package:: *)

(* Constructs approximation to the perfect foresight liquidity constrained solution *)
(* See Documentation folder for details and illustration of the approach *)

MakeSmoothApproxToLiqConstrSoln[JoinAtNode_] := Block[{},

(* Idea: Smoothly join the consumption function at the kink to the continuous function above some join point *)
(* First, figure out how many periods out to go, based on maximum a in aGrid *)

nMin = 1;
nMax = IntegerPart[nFromLasta\[Sharp][aGridVecExcBot[[-1]]]+1]; (* nFromLasta\[Sharp] yields time distance to date when constraint would bind *)

Clear[cFrom1Func,\[Kappa]From1Func,cFrom1,\[Kappa]From1,\[Kappa]PFrom1,cFrom\[Natural],\[Kappa]From\[Natural],\[Kappa]PFrom\[Natural],\[Kappa]From\[Natural]Func,cFrom\[Natural]Func,c\[Natural],\[Kappa]\[Natural],\[Kappa]P\[Natural],m\[Natural],mWhere\[ScriptC]LimHasKink,mLast,mWhere\[ScriptC]LimHasKink];

(* Approximate consumption function around point mWhere\[ScriptC]LimHasKink -- note that we impose a \[Kappa] of 1 at m=mWhere\[ScriptC]LimHasKink *)
cFrom1Func[m_,\[Phi]_,\[Xi]_] := mWhere\[ScriptC]LimHasKink+(m-mWhere\[ScriptC]LimHasKink)1-(\[Phi]/2) (m-mWhere\[ScriptC]LimHasKink)^2+(\[Xi]/6)(m-mWhere\[ScriptC]LimHasKink)^3;
\[Kappa]From1Func[m_,\[Phi]_,\[Xi]_] :=     1                                    -\[Phi]    (m-mWhere\[ScriptC]LimHasKink)  +(\[Xi]/2)(m-mWhere\[ScriptC]LimHasKink)^2;
cFrom1   = mWhere\[ScriptC]LimHasKink+(m-mWhere\[ScriptC]LimHasKink)1-(\[Phi]/2) (m-mWhere\[ScriptC]LimHasKink)^2+(\[Xi]/6)(m-mWhere\[ScriptC]LimHasKink)^3;
\[Kappa]From1   =     1                                  - \[Phi]    (m-mWhere\[ScriptC]LimHasKink)  +(\[Xi]/2)(m-mWhere\[ScriptC]LimHasKink)^2;
\[Kappa]PFrom1  =                                        - \[Phi]                            +\[Xi] (m-mWhere\[ScriptC]LimHasKink);

(* Time distance at which the approximate function joins the exact continuous one *)
nJoin = JoinAtNode;

cFrom\[Natural]  = c\[Natural] + \[Kappa]\[Natural] (m-m\[Natural]) + (\[Kappa]P\[Natural]/2) (m-m\[Natural])^2 + (\[Kappa]PP\[Natural]/6) (m-m\[Natural])^3;
\[Kappa]From\[Natural]  =      \[Kappa]\[Natural]        +  \[Kappa]P\[Natural]    (m-m\[Natural])   + (\[Kappa]PP\[Natural]/2) (m-m\[Natural])^2;
\[Kappa]PFrom\[Natural] =                +  \[Kappa]P\[Natural]             +  \[Kappa]PP\[Natural]    (m-m\[Natural]);
\[Kappa]From\[Natural]Func[m_] :=      \[Kappa]\[Natural]        +  \[Kappa]P\[Natural]    (m-m\[Natural])   + (\[Kappa]PP\[Natural]/2) (m-m\[Natural])^2;
cFrom\[Natural]Func[m_] := c\[Natural] + \[Kappa]\[Natural] (m-m\[Natural]) + (\[Kappa]P\[Natural]/2) (m-m\[Natural])^2 + (\[Kappa]PP\[Natural]/6) (m-m\[Natural])^3;


(* Analytically solve for parameter values such that level and MPC match at mWhere\[ScriptC]LimHasKink point *)
Off[Solve::ratnz]; (* Turn off warning message that is useless in this context *)
Solve\[Natural] = Solve[
  {cFrom1  == cFrom\[Natural]
  ,\[Kappa]From1  == \[Kappa]From\[Natural]
  ,\[Kappa]PFrom1 == \[Kappa]PFrom\[Natural]},{m,\[Phi],\[Xi]}];
On[Solve::ratnz]; (* Turn warning message back on*)

(* Now set the parameter values so that the analytical solution converts to numerical *)
{c\[Natural],\[Kappa]\[Natural],\[Kappa]P\[Natural],m\[Natural],mLast} = {c\[Sharp][nJoin],dcdb[nJoin],dcdbb[nJoin],m\[Sharp][nJoin],m\[Sharp][nMax]};
mWhere\[ScriptC]LimHasKink = m\[Sharp][1]; 
\[Kappa]PP\[Natural] = (dcdbb[nJoin+0.00001]-dcdbb[nJoin+0.00001])/0.00002;

(* Extract the solutions from the solved formula *)
mSplice    = Select[  m /. Solve\[Natural],mWhere\[ScriptC]LimHasKink < # < m\[Natural] &][[1]];
mSplicePos = Position[m /. Solve\[Natural],mSplice][[1,1]];
\[Phi]Solved = (\[Phi] /. Solve\[Natural])[[mSplicePos]];
\[Xi]Solved = (\[Xi] /. Solve\[Natural])[[mSplicePos]];
cSplice = cFrom\[Natural] /. m -> mSplice;
\[Kappa]Splice = \[Kappa]From\[Natural] /. m -> mSplice;
\[Kappa]PSplice= \[Kappa]PFrom\[Natural] /. m -> mSplice;

\[ScriptC]Liq[m_]:= Piecewise[
{{m,m < mWhere\[ScriptC]LimHasKink}
,{cFrom1Func[m,\[Phi]Solved,\[Xi]Solved],mWhere\[ScriptC]LimHasKink <= m <= mSplice}
,{cFrom\[Natural]Func[m],mSplice < m < m\[Natural]}
,{cLiqOrder3Func[m],m\[Natural] <= m <= mLast}
,{cLiqExact[m],m > mLast}
}];
\[Kappa]Liq[m_]:= Piecewise[
{{1,m < mWhere\[ScriptC]LimHasKink}
,{\[Kappa]From1Func[m,\[Phi]Solved,\[Xi]Solved],mWhere\[ScriptC]LimHasKink <= m <= mSplice}
,{\[Kappa]From\[Natural]Func[m],mSplice < m < m\[Natural]}
,{cLiqOrder3Func'[m],m\[Natural] <= m <= mLast}
,{\[Kappa]LiqExact[m],m > mLast}
}];

SetAttributes[{\[ScriptC]Liq,\[Kappa]Liq},Listable];

{bKinksExcBot,mKinksExcBot,cKinksExcBot,aKinksExcBot,dcdbExcBot,dcdbbExcBot}=
  Transpose[Map[{b\[Sharp][#],m\[Sharp][#],c\[Sharp][#],m\[Sharp][#]-c\[Sharp][#],dcdb[#],dcdbb[#]}&,Table[n,{n,nMin,nMax,1}]]];
(* beginCDCPrivate
mKinksIncBot = Prepend[mKinksExcBot,0.];
cKinksIncBot = Prepend[cKinksExcBot,0.];
dcdbIncBot   = Prepend[dcdbExcBot,1.];
dcdbbIncBot  = Prepend[dcdbbExcBot,0.];
  endCDCPrivate *)
cLiqOrder1Func=Interpolation[
  cLiqOrder1=Transpose[{Transpose[{mKinksExcBot}],cKinksExcBot}],InterpolationOrder->1];

nOrder3Start=nJoin;
(* beginCDCPrivate
nOrder2Start=nOrder3Start
cLiqOrder2Func=Interpolation[cLiqOrder2=
Prepend[
 Prepend[
(*  Prepend[*)
    Take[ 
      Transpose[{Transpose[{mKinksExcBot}],cKinksExcBot,dcdbExcBot}],{nOrder2Start,Length[cKinksExcBot]}
    ],{{1.},1.,1.}]
     ,{{mKinksExcBot[[1]]},cKinksExcBot[[1]],1.}]
(*    ,{{mAdd},cAdd,cStartingFrom3m}]*)
];
endCDCPrivate *)
(* Approximation using level and first two derivatives can be evaluated almost instantly *)
cLiqOrder3Func=Interpolation[
 cLiqOrder3=
(*  Prepend[*)
(*   Prepend[*)
    Prepend[
      Take[
        Transpose[{Transpose[{mKinksExcBot}],cKinksExcBot,dcdbExcBot,dcdbbExcBot}]
      ,{nOrder3Start,Length[cKinksExcBot]}]
     ,{{mSplice},cSplice,\[Kappa]Splice,\[Kappa]PSplice}]
(*    ,{{mKinksExcBot[[1]]},cKinksExcBot[[1]],1.,0.}]*)
(*  ,{{0.},0.,1.,0.}]*)
];
]; (* End MakeSmoothApproxToLiqConstrSoln *)
