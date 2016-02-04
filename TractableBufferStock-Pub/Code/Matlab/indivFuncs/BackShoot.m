% A function which iterates the reverse Euler equations until it reaches a
% point outside some predefined boundaries.
% It takes as an input an nx4 matrix (only caring about the last row) and
% returns an nx4 matrix with more points appended.

function PointsList = BackShoot(InitialPoints)
globalizeTBSvars;
if VerboseOutput == 1
    disp('Solving for Euler points...');
end
scriptmMaxBound = 100 * scriptmE;
scriptmMinBound = 1;
Counter = 0;
InitialSize = size(InitialPoints);
InitialLength = InitialSize(1);
if InitialSize(2) ~= 5
    error('BackShoot was passed a matrix that does not have width 5, terminating.')
end
% Set the first point to be used as the last point from the inputed matrix
PointsList = InitialPoints;
scriptmPrev = InitialPoints(InitialLength,1);
scriptcPrev = InitialPoints(InitialLength,2);
kappaPrev = InitialPoints(InitialLength,3);
scriptvPrev = InitialPoints(InitialLength,4);
kappaPPrev = InitialPoints(InitialLength,5);
% Add points to the PointsList until a point exceeds the specified bounds
while ((scriptmPrev > scriptmMinBound) && (scriptmPrev <= scriptmMaxBound))
    scriptmNow = scriptmEtFromtp1(scriptmPrev, scriptcPrev);
    scriptcNow = scriptcEtFromtp1(scriptmPrev, scriptcPrev);
    kappaNow = kappaEtFromtp1(scriptmPrev, scriptcPrev, kappaPrev, scriptmNow, scriptcNow);
    kappaPNow = kappaEPtFromtp1(scriptmPrev, scriptcPrev, kappaPrev, scriptmNow, scriptcNow, kappaNow, kappaPPrev);
    scriptvNow = CRRA(scriptcNow) + mybeta * (biggamma^(1-rho)) * ((1-mho) * scriptvPrev + mho * vUPF(scriptR * (scriptmNow - scriptcNow)));
    newDataPoint = [scriptmNow scriptcNow kappaNow scriptvNow kappaPNow];
    PointsList = [PointsList ; newDataPoint];
    scriptmPrev = scriptmNow;
    scriptcPrev = scriptcNow;
    kappaPrev = kappaNow;
    scriptvPrev = scriptvNow;
    kappaPPrev = kappaPNow;
    Counter = Counter + 1;
end
% Tell the user when the points exceed the bounds
if scriptmPrev <= scriptmMinBound && VerboseOutput==1
    disp(['Went below minimum bound after ' num2str(Counter-1) ' backwards Euler iterations.']);
    disp(['Last point: {' num2str(newDataPoint) '}']); 
end
if scriptmPrev > scriptmMaxBound && VerboseOutput==1;
    disp(['Went above maximum bound after ' num2str(Counter-1) ' backwards Euler iterations.']);
    disp(['Last point: {' num2str(newDataPoint) '}']);
end
