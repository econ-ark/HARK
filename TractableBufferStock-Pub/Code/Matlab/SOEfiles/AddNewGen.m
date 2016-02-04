% This function adds a new period in which the old generations have moved
% forward in time one period and the new generation starts with stake
% scriptbStake and is larger than last period's newborn generation by the factor scriptLGro

function nulloutput = AddNewGen(inputs)
scriptbStake = inputs(1);
scriptLGro = inputs(2);
scriptWGro = inputs(3);
globalizeTBSvars;

scriptbNew = scriptbStake;
scriptyNew = scriptbStake*(((1-mho)*bigR/scriptWGro-1))/((1-mho)*bigR/scriptWGro)+1;
scriptmNew = scriptbNew + 1;
scriptcNew = cE(scriptmNew);
scriptaNew = scriptmNew - scriptcNew;
kappaNew = D(@cE,scriptmNew);
scriptvNew = vE(scriptmNew);
scriptxNew = scriptyNew - scriptcNew;
scriptLNew = 1;
NewPop = [scriptbNew scriptyNew scriptmNew scriptcNew scriptaNew kappaNew scriptvNew scriptxNew scriptLNew];

if VerboseOutput==1
    disp('New population:')
    disp([CensusVars num2str(transpose(NewPop))]);
end

scriptbtFromtm1 = Census(:,scriptaPos)*(1-mho)*bigR/scriptWGro;
scriptytFromtm1 = Census(:,scriptaPos)*((1-mho)*bigR/scriptWGro-1)+1;
scriptmtFromtm1 = scriptbtFromtm1 + 1;
j = 1;
scriptctFromtm1 = [];
kappatFromtm1 = [];
scriptvtFromtm1 = [];
while j<=length(scriptmtFromtm1)
    scriptctFromtm1 = [scriptctFromtm1; cE(scriptmtFromtm1(j))];
    kappatFromtm1 = [kappatFromtm1; D(@cE,scriptmtFromtm1(j))];
    scriptvtFromtm1 = [scriptvtFromtm1; vE(scriptmtFromtm1(j))];
    j = j+1;
end
scriptatFromtm1 = scriptmtFromtm1 - scriptctFromtm1;
scriptxtFromtm1 = scriptytFromtm1 - scriptctFromtm1;
scriptLtFromtm1 = Census(:,scriptLPos)*(1-mho)/scriptLGro;
tFromtm1Pop = [scriptbtFromtm1 scriptytFromtm1 scriptmtFromtm1 scriptctFromtm1 scriptatFromtm1 kappatFromtm1 scriptvtFromtm1 scriptxtFromtm1 scriptLtFromtm1];

if VerboseOutput==1
    disp('Old population moved forward:');
    disp([CensusVars num2str(transpose(tFromtm1Pop))]);
end

% This section tries to do what NearestSubPopFind does with interpolation.
% It does not use NearestSubPopBelow and it might need some work.
[tempa,tempb] = min(abs(scriptmNew - Census(:,scriptmPos)));
NearestSubPop = tempb;
if abs(scriptbNew - scriptbtFromtm1(NearestSubPop)) < epsilon
    tFromtm1Pop(NearestSubPop,scriptLPos) = tFromtm1Pop(NearestSubPop,scriptLPos) + scriptLNew;
    Census = tFromtm1Pop;
else
    Census = [NewPop;tFromtm1Pop];
end

TabulateLastCensus;
gens = size(NIPAAgg,2);
scriptW = scriptWGro * NIPAAgg(scriptWPos,gens);
scriptL = sum(Census(:,scriptLPos));
tau = scriptbStake * (1-(1/scriptLGro));
Tau = tau * scriptW * scriptL;
NIPAAgg = [NIPAAgg [scriptW;Tau;scriptWGro*scriptLGro]];
NIPAAggT = transpose(NIPAAgg);
NIPAInd = [NIPAInd [1;tau;scriptWGro/(1-mho)]];
NIPAIndT = transpose(NIPAInd);

if VerboseOutput==1
    disp('Merged population:');
    disp([CensusVars num2str(transpose(Census))]);
end
