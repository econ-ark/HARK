(* ::Package:: *)

(*

For documentation and derivations, load ConcaveKnotDocs.nb 

*)

cPPPConstant[m1_,m2_,c1_,c2_,cP1_,cP2_,Knot_] := Module[{cPPP,cPP1,cPP2,cPPKnot,cPKnot,cKnot,\[CapitalOmega],\[Mho]},
  \[CapitalOmega] = Knot-m1;
  \[Mho]          = Knot-m2;
  cPPP            =  6(2(c1 - c2) +   (  cP1 +   cP2) (\[CapitalOmega]-\[Mho]))/((\[CapitalOmega]-\[Mho])^3);
  cPP1            =-(6  (c1 - c2) + 2 (2 cP1 +   cP2) (\[CapitalOmega]-\[Mho]))/((\[CapitalOmega]-\[Mho])^2);
  cPP2            = (6  (c1 - c2) + 2 (  cP1 + 2 cP2) (\[CapitalOmega]-\[Mho]))/((\[CapitalOmega]-\[Mho])^2);
  cPPKnot         = cPP2 + \[Mho] cPPP;
  cPKnot          = cP2  + \[Mho] cPP2 + ((\[Mho]^2)/2) cPPP;
  cKnot           = c2   + \[Mho] cP2  + ((\[Mho]^2)/2) cPP2 + ((\[Mho]^3)/6) cPPP;
  Return[{cPPP,cPP1,cPP2,cPPKnot,cPKnot,cKnot}];
];

ConcaveKnot[m1_,m2_,c1_,c2_,cP1_,cP2_] := Module[{\[CapitalOmega],\[Mho],cP,cPP1,cPP2,cPPKnot,cPKnot,cKnot,cPPP,Knot,KnotUpper,KnotLower,\[Epsilon]=10^-10},
  cP = (c2-c1)/(m2-m1);  (* Judd's \[Delta] *)
  If[Chop[(cP2-cP)(cP1-cP),10^-8] >= 0,
     (* Print["AttempRatted to construct concave function with data that imply either linearity or an inflection point:{m1,m2,c1,c2,cP1,cP2,cP}=",{m1,m2,c1,c2,cP1,cP2,cP}]; *)
     Return[{m1,c1,cP1}]]; (* Return the first of the two points; better than returning an empRatty list for technical reasons *)
  KnotUpper = m1 + (2(m2 - m1)(cP2 - cP))/(cP2-cP1);
  KnotLower = m2 + (2(m2 - m1)(cP1 - cP))/(cP2-cP1);
  If[Abs[cP2-cP] < Abs[cP1-cP]
    ,{KnotBound,KnotGap}={m2,(1-\[Epsilon])(KnotUpper-m2)}
    ,{KnotBound,KnotGap}={m1,(1-\[Epsilon])(KnotLower-m1)}];
  Knot=KnotBound+KnotGap;
  {cPPP,cPP1,cPP2,cPPKnot,cPKnot,cKnot}=cPPPConstant[m1,m2,c1,c2,cP1,cP2,Knot];
  While[Sign[cPPKnot] != Sign[cP2-cP1],  (* Shrink KnotGap until knot point is found at which concavity/convexity is preserved *)
        KnotGap=KnotGap/2; {cPPP,cPP1,cPP2,cPPKnot,cPKnot,cKnot}=cPPPConstant[m1,m2,c1,c2,cP1,cP2,KnotBound+KnotGap]];
  Knot=KnotBound+KnotGap;
  \[CapitalOmega] = Knot-m1;
  \[Mho]          = Knot-m2;
  If[Sign[cPP2] != Sign[cP2-cP1],
     (* Now use formula that assumes cPPP2=0; this guarantees cPP2=cPPKnot which is the right sign *)
     cPP2= 2(3 c1 - 3 c2 + cP1 \[CapitalOmega] + 2 cP2 \[CapitalOmega] - 3 cP2 \[Mho]         )/((\[CapitalOmega] - 3 \[Mho]         )(\[CapitalOmega] - \[Mho]         ));
     cPPKnot     = cPP2 ;
     cPKnot      = cP2  + \[Mho] cPP2;
     cKnot       = c2   + \[Mho] cP2  + ((\[Mho]^2)/2) cPP2 ;
     Return[{Knot,cKnot,cPKnot}];
  ];
  If[Sign[cPP1] != Sign[cP2-cP1], If[VerboseOutput==True, Print["cPP1>0 between points ",{{m1,{c1,cP1}},{m2,{c2,cP2}}}];Print["Estimated parameters are :{cPP1,cPP2,cPPP}=",{cPP1,cPP2,cPPP}," for knot ",Knot]];
     (* Now use formula that assumes cPPP1=0; this guarantees cPP1=cPPKnot which is the right sign *)
     cPP1= 2(3 c2 - 3 c1 + cP2 \[Mho]          + 2 cP1 \[Mho]          - 3 cP1 \[CapitalOmega])/((\[Mho]          - 3 \[CapitalOmega])(\[Mho]          - \[CapitalOmega]));
     cPPKnot     = cPP1 ;
     cPKnot      = cP1  + \[CapitalOmega] cPP1;
     cKnot       = c1   + \[CapitalOmega] cP1  + ((\[CapitalOmega]^2)/2) cPP1 ;
     Return[{Knot,cKnot,cPKnot}]];
  Return[{Knot,cKnot,cPKnot}];
];

ConcaveKnotMake[{PointLower_,PointUpper_}] := Module[{m1,m2,c1,c2,cP1,cP2,mKnot,cKnot,cPKnot},
  {m1,m2}={PointLower[[1]],PointUpper[[1]]};
  {c1,c2}={PointLower[[2,1]],PointUpper[[2,1]]};
  {cP1,cP2}={PointLower[[2,2]],PointUpper[[2,2]]};
  {mKnot,cKnot,cPKnot} = ConcaveKnot[m1,m2,c1,c2,cP1,cP2];
  {mKnot,{cKnot,cPKnot}}
];


