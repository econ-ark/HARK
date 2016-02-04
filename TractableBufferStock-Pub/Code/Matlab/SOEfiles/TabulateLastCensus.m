% Calculate the mean values of population variables using the masses of the
% various populations.

scriptL = sum(Census(:,scriptLPos));
NewCensusMeans = [];
j = 1;
while j<size(CensusVars,1)
    NewCensusMeans = [NewCensusMeans sum(Census(:,j).*Census(:,scriptLPos))/scriptL];
    j=j+1;
end
NewCensusMeans = [NewCensusMeans scriptL];
CensusMeans = [CensusMeans ; NewCensusMeans];
CensusMeansT = transpose(CensusMeans);
