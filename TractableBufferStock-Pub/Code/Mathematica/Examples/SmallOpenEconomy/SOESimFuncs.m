(* ::Package:: *)

(* CDC Note: Lines containing ::: (including this line!) will not be copied to the public version of the notebook *)
(* So lines containing ::: are useful for extra code that does not need to be public but might be valuable to preserve *)


(* Setup basic structure of a population's characteristics *)

(* Variables that characterize individual populations/generations *)
CensusVars = {"\[ScriptB]","\[ScriptY]","\[ScriptM]","\[ScriptC]","\[ScriptA]","\[Kappa]","\[ScriptV]","\[ScriptX]","\[ScriptCapitalL]"};

(* Variables that characterize the aggregate economy as a whole *)
NIPAAggVars   = {"\[ScriptCapitalW]","\[CapitalTau]","\[ScriptCapitalY]Gro"};
NIPAIndVars   = {"\[ScriptW]","\[Tau]","\[ScriptY]Gro"};

(* \[Bullet]Pos variables indicate the location of the object \[Bullet] in a population's data structure *)
Do[ToExpression[CensusVars[[i]]<>"Pos = "<>ToString[i]<>";" ],{i,Length[CensusVars]}];
Do[ToExpression[NIPAAggVars[[i]]<>"Pos = "<>ToString[i]<>";" ],{i,Length[NIPAAggVars]}];
Do[ToExpression[NIPAIndVars[[i]]<>"Pos = "<>ToString[i]<>";" ],{i,Length[NIPAIndVars]}];

NIPAAggPrint := Print[MatrixForm[NIPAAgg,TableHeadings->{Automatic,NIPAAggVars}]];
NIPAIndPrint := Print[MatrixForm[NIPAInd,TableHeadings->{Automatic,NIPAIndVars}]];
CensusPrint := Print[MatrixForm[CensusMeans,TableHeadings->{Automatic,CensusVars}]];


(* There are two natural choices for the structure of the population in period 0 *)

(* With stakes, the entire population mass of the country is identically at the target *)
(* Without stakes, the entire population mass of the country has wealth of zero *)

(* In either case, the productivity mass of the active labor force is the same \[ScriptCapitalL] *)

CensusMakeNoStakes := Block[{},
  Census           = {{{\[ScriptB]0=0.},{\[ScriptY]0=1.},{\[ScriptM]0=1.},{\[ScriptC]0=cE[1]},{\[ScriptA]0=1-\[ScriptC]0},{\[Kappa]0=cE'[\[ScriptM]0]},{\[ScriptX]0=\[ScriptY]0-\[ScriptC]0},{\[ScriptV]0=vE[\[ScriptM]0]},{\[ScriptCapitalL]0=\[ScriptCapitalL]E}}};
  CensusMeans      = {};
  TabulateLastCensus;
  NIPAAgg       = {{\[ScriptCapitalW]0=1,\[CapitalTau]0=0,\[ScriptCapitalY]Gro0=\[ScriptCapitalN] \[GothicCapitalG]}};
  NIPAInd       = {{\[ScriptW]0=1,\[Tau]0=0,\[ScriptY]Gro0=\[GothicCapitalG]/(1-\[Mho])}};
];

CensusMakeStakes := Block[{},
  Census           = {Table[ToExpression[{CensusVars[[i]]<>"E"}],{i,Length[CensusVars]}]};
  CensusMeans      = {};
  TabulateLastCensus;
  NIPAAgg       = {{\[ScriptCapitalW]0=1,\[CapitalTau]0=\[ScriptB]E (1-(1/\[ScriptCapitalN]))\[ScriptCapitalW]0 \[ScriptCapitalL]0,\[ScriptCapitalY]Gro0=\[ScriptCapitalN] \[GothicCapitalG]}};
  NIPAInd       = {{\[ScriptW]0=1,\[Tau]0=\[ScriptB]E (1-(1/\[ScriptCapitalN])),\[ScriptY]Gro0=\[GothicCapitalG]/(1-\[Mho])}};
];

(* Calculate the mean values of population variables using the masses of the various populations *)
TabulateLastCensus := Block[{},
  \[ScriptCapitalL] = Total[Last[Census][[\[ScriptCapitalL]Pos]]];
  AppendTo[
   CensusMeans,
    Append[
      Table[ToExpression["Total[Last[Census][[" <> CensusVars[[i]] <> "Pos]]Last[Census][[\[ScriptCapitalL]Pos]]]"]/\[ScriptCapitalL]
      ,{i, Length[CensusVars]-1}]
    ,\[ScriptCapitalL]]];
  CensusMeansT=Transpose[CensusMeans];
];



Clear[\[ScriptCapitalN]];

(* Steady state aggregate labor supply for active households; formula derived in handout *)
\[ScriptCapitalL]E = 1/(1-(1-\[Mho])/\[ScriptCapitalN]);

(* 

The variable Census contains, for each date, a list of points in the state
space, along with (for each point included) the mass of efficiency
units of labor corresponding to households living at that point.  The
sum of the masses in the list constitutes the aggregate mass of
efficiency units of labor at the date

*)

(* AddNewGen[{{\[ScriptB]E,\[ScriptCapitalN],\[GothicCapitalG]}}] would add a new period in which the old generations have moved forward in time one period 
and the new generation starts with stake \[ScriptB]E and is larger than last period's newborn generation by the factor \[ScriptCapitalN] *)
AddNewGen[{\[ScriptB]Stake_,\[ScriptCapitalL]Gro_,\[ScriptCapitalW]Gro_}] := Block[{},
  NewPop = {
    \[ScriptB]New = \[ScriptB]Stake
   ,\[ScriptY]New = \[ScriptB]Stake (((1-\[Mho])(R)/\[ScriptCapitalW]Gro-1))/((1-\[Mho])(R)/\[ScriptCapitalW]Gro)+1
   ,\[ScriptM]New = \[ScriptB]New + 1
   ,\[ScriptC]New = cEInterp[\[ScriptM]New]
   ,\[ScriptA]New = \[ScriptM]New-\[ScriptC]New
   ,\[Kappa]New = cEInterp'[\[ScriptM]New]
   ,\[ScriptV]New = vE[\[ScriptM]New]
   ,\[ScriptX]New = Chop[\[ScriptY]New-\[ScriptC]New]
   ,\[ScriptCapitalL]New = 1};
  If[VerboseOutput != False,Print["New population:"]];
  If[VerboseOutput != False,Print[MatrixForm[Transpose[{CensusVars,NewPop}]]]];
  tFromtm1Pop = {
    \[ScriptB]tFromtm1 = Last[Census][[\[ScriptA]Pos]] (1-\[Mho])(R)/\[ScriptCapitalW]Gro
   ,\[ScriptY]tFromtm1 = Last[Census][[\[ScriptA]Pos]] ((1-\[Mho])(R)/\[ScriptCapitalW]Gro-1)+1
   ,\[ScriptM]tFromtm1 = \[ScriptB]tFromtm1 + 1
   ,\[ScriptC]tFromtm1 = cEInterp[\[ScriptM]tFromtm1]
   ,\[ScriptA]tFromtm1 = \[ScriptM]tFromtm1 - \[ScriptC]tFromtm1
   ,\[Kappa]tFromtm1 = Map[cEInterp',\[ScriptM]tFromtm1]
   ,\[ScriptV]tFromtm1 = Map[vE[#] &, \[ScriptM]tFromtm1]
   ,\[ScriptX]tFromtm1 =Chop[\[ScriptY]tFromtm1-\[ScriptC]tFromtm1]
   ,\[ScriptCapitalL]tFromtm1 = Last[Census][[\[ScriptCapitalL]Pos]] (1-\[Mho])/\[ScriptCapitalL]Gro};
  If[VerboseOutput != False,Print["Old population moved forward:"]];
  If[VerboseOutput != False,Print[MatrixForm[Transpose[{CensusVars,tFromtm1Pop}]]]];
  \[ScriptM]PosMax = Length[Last[Census][[\[ScriptM]Pos]]];
  NearestSubPopFind = Interpolation[Transpose[{Last[Census][[\[ScriptM]Pos]],Table[i,{i,Length[Last[Census][[\[ScriptM]Pos]]]}]}],InterpolationOrder->1];
  NearestSubPop     = Min[\[ScriptM]PosMax,Max[1,Round[NearestSubPopFind[\[ScriptM]New]]]];
  NearestSubPopBelow= Min[\[ScriptM]PosMax,Max[1,Floor[NearestSubPopFind[\[ScriptM]New]+\[CurlyEpsilon]]]];
  If[Abs[Chop[\[ScriptB]New-\[ScriptB]tFromtm1[[NearestSubPop]]]] < \[CurlyEpsilon],
     (* then the new population is very close to an already-existing population, so just add its mass to the existing mass *)
     tFromtm1Pop[[\[ScriptCapitalL]Pos,NearestSubPopBelow]] = tFromtm1Pop[[\[ScriptCapitalL]Pos,NearestSubPopBelow]]+\[ScriptCapitalL]New;
     AppendTo[Census,tFromtm1Pop];,
     (* else it's not close enough to an existing population, so add it to the lists in the appropriate locations *)
     AppendTo[Census,Table[ToExpression["Insert[tFromtm1Pop[[" <> CensusVars[[i]] <> "Pos]],NewPop[[" <> CensusVars[[i]] <> "Pos]],NearestSubPopBelow]"], {i, Length[CensusVars]}]]
  ];
  TabulateLastCensus;
  \[ScriptCapitalW]= \[ScriptCapitalW]Gro Last[NIPAAgg][[\[ScriptCapitalW]Pos]];
  \[ScriptCapitalL]= Total[Last[Census][[\[ScriptCapitalL]Pos]]];
  \[Tau]= \[ScriptB]Stake (1-(1/\[ScriptCapitalN]));
  \[CapitalTau]=\[Tau] \[ScriptCapitalW] \[ScriptCapitalL];
  AppendTo[NIPAAgg,{\[ScriptCapitalW],\[CapitalTau],\[ScriptCapitalW]Gro \[ScriptCapitalN]}];NIPAAggT=Transpose[NIPAAgg];
  AppendTo[NIPAInd,{1,\[Tau],\[ScriptCapitalW]Gro/(1-\[Mho])}];NIPAIndT=Transpose[NIPAInd];
  If[VerboseOutput != False,Print["Merged population:"]];
  If[VerboseOutput != False,Print[MatrixForm[Transpose[{CensusVars,Last[Census]}]]]];
];
