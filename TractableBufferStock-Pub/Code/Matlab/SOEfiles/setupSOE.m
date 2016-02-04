% This script defines the relevant SOE variables and functions
% It needs the base parameters to be defined in order to run.

% Set population growth parameter
scriptN = 1.01;

% Set up basic structure of a population's characteristics

% Variables that characterize individual populations/generations
CensusVars = ['scriptb ';'scripty ';'scriptm ';'scriptc ';'scripta ';'kappa   ';'scriptv ';'scriptx ';'scriptL '];

% Variables that characterize the aggregate economy as a whole
NIPAAggVars = ['scriptW    ';'Tau        ';'scriptYGro '];
NIPAIndVars = ['scriptw    ';'tau        ';'scriptyGro '];

% *Pos variables indicate the location of the object * in a population's data structure
scriptbPos = 1;
scriptyPos = 2;
scriptmPos = 3;
scriptcPos = 4;
scriptaPos = 5;
kappaPos = 6;
scriptvPos = 7;
scriptxPos = 8;
scriptLPos = 9;

scriptWPos = 1;
TauPos = 2;
scriptYGroPos = 3;

scriptwPos = 1;
tauPos = 2;
scriptyGroPos = 3;

% Steady level of population, solved in handout
scriptLE = 1/(1-(1-mho)/scriptN);
