% There are two natural choices for the structure of the population in period 0
% With stakes, the entire population mass of the country is identically at the target

scriptb0 = scriptbE;
scripty0 = scriptyE;
scriptm0 = scriptmE;
scriptc0 = scriptcE;
scripta0 = scriptaE;
kappa0 = kappaE;
scriptv0 = scriptvE;
scriptx0 = scriptxE;
scriptL0 = scriptLE;

Census = [scriptb0 scripty0 scriptm0 scriptc0 scripta0 kappa0 scriptv0 scriptx0 scriptL0];
CensusMeans = [];
TabulateLastCensus;

scriptW0 = 1;
Tau0 = scriptbE * (1-(1/scriptN))*scriptW0 * scriptL0;
scriptYGro = scriptN * bigG;
NIPAAgg = [scriptW0;Tau0;scriptYGro];

scriptw0 = 1;
tau0 = scriptbE * (1-(1/scriptN));
scriptyGro = bigG/(1-mho);
NIPAInd = [scriptW0;Tau0;scriptYGro];
