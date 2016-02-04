% There are two natural choices for the structure of the population in period 0
% Without stakes, the entire population mass of the country has wealth of zero

scriptb0 = 0;
scripty0 = 1;
scriptm0 = 1;
scriptc0 = cE(1);
scripta0 = 1-scriptc0;
kappa0 = D(@cE,scriptm0);
scriptv0 = vE(scriptm0);
scriptx0 = scripty0-scriptc0;
scriptL0 = scriptLE;

Census = [scriptb0 scripty0 scriptm0 scriptc0 scripta0 kappa0 scriptv0 scriptx0 scriptL0];
CensusMeans = [];
TabulateLastCensus;

scriptW0 = 1;
Tau0 = 0;
scriptYGro = scriptN * bigG;
NIPAAgg = [scriptW0;Tau0;scriptYGro];

scriptw0 = 1;
tau0 = 0;
scriptyGro = bigG/(1-mho);
NIPAInd = [scriptW0;Tau0;scriptYGro];
