(* ::Package:: *)

SetupGrids := Block[{},
(* Construct the grid of possible values of a *)
(* Large grid captures behavior as assets go to infinity - use exponential growth for a *)
(* Small grid captures behavior as assets go to zero - use multi-exponential growth for a *)
  aSmallMaxNest  = Nest[Log[#+1]&,aSmallMax,TimesToNestSmallGrid] //N; 
  aGridVecExcBotSmall =  Table[Nest[Exp[#]-1 &,aNestLoop,TimesToNestSmallGrid],{aNestLoop,0.,aSmallMaxNest,aSmallMaxNest/(aGridVecExcBotSmallLength-1)}];
  aLargeTop = aLargeMax/(1.1 aGridVecExcBotSmall[[-1]]);
  aGridVecExcBotLarge = Table[(Exp[laLoop Log[aGridVecExcBotSmall[[-1]]]]) //N,{laLoop,0,Log[aLargeMax+1]-Log[aGridVecExcBotSmall[[-1]]],(Log[aLargeTop]-Log[aGridVecExcBotSmall[[-1]]])/(aGridVecExcBotLargeLength-1)}];

  aVecExcBot = aGridVecExcBot = Rest[aVecIncBot = aGridVecIncBot = Union[aGridVecExcBotSmall,aGridVecExcBotLarge aGridVecExcBotSmall[[-1]]]];
];

