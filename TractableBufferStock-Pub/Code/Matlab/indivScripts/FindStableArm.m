% This script finds the consumption function and related objects

% Generate the Euler points below the steady state
StableArmBelowSS = sortLowerArm(EulerPointsStartingFromSSPlus(-epsilon));

% This section augments precision in the lower arm by using other small
% values for epsilon and then combining the various sets of Euler points
IterLength = length(StableArmBelowSS);
DscriptmFirst = StableArmBelowSS(IterLength,1) - StableArmBelowSS(IterLength-1,1);
PrecisionAugmentationFactor = floor(1 + 0.1/mho);
if ((PrecisionAugmentationFactor > 1) && (VerboseOutput == 1))
    disp(['Solving the lower arm ' num2str(PrecisionAugmentationFactor-1) ' more times from different starting points to augment precision.']);
end
i = 1;
while i < PrecisionAugmentationFactor
    tempEpsilon = -epsilon - DscriptmFirst * (i/PrecisionAugmentationFactor);
    tempArm = EulerPointsStartingFromSSPlus(tempEpsilon);
    tempArm = tempArm(1:min([IterLength length(tempArm)]),:);
    StableArmBelowSS = [StableArmBelowSS ; tempArm];
    i = i + 1;
end
StableArmBelowSS = sortLowerArm(StableArmBelowSS);

% Generate the Euler points above the steady state
StableArmAboveSS = EulerPointsStartingFromSSPlus(epsilon);
% The stable arm consists of Euler points below, above, and at the SS
ETarget = [scriptmE scriptcE kappaE scriptvE kappaEP];
StableArmPoints = [StableArmBelowSS ; ETarget ; StableArmAboveSS];
SAlength = length(StableArmPoints);

scriptmBot = StableArmPoints(1,1);
scriptcBot = StableArmPoints(1,2);
kappaBot = StableArmPoints(1,3);
scriptvBot = StableArmPoints(1,4);
kappaPBot = StableArmPoints(1,5);

scriptmTop = StableArmPoints(SAlength,1);
scriptcTop = StableArmPoints(SAlength,2);
kappaTop = StableArmPoints(SAlength,3);
scriptvTop = StableArmPoints(SAlength,4);
kappaPTop = StableArmPoints(SAlength,5);

scriptmVec = StableArmPoints(:,1);
scriptcVec = StableArmPoints(:,2);
kappaVec = StableArmPoints(:,3);
scriptvVec = StableArmPoints(:,4);
kappaPVec = StableArmPoints(:,5);

uPVec = CRRAp(scriptcVec);
uPPVec = CRRApp(scriptcVec);
uPPPVec = CRRAppp(scriptcVec);
scriptvPVec = uPVec;
scriptvPPVec = uPPVec .* kappaVec;

if VerboseOutput==1
    disp('Stable arm points constructed.');
end

kappaEPAtZero = 2 * (scriptcBot - scriptmBot * kappaEMax) / (scriptmBot^2);

% Generate the coefficients for the quintic splines that compose the
% consumption function interpolation
EulerPoints = [0 0 kappaEMax kappaEPAtZero; StableArmPoints(:,[1 2 3 5])];
consumptionCoeffs = generateInterpMatrix(EulerPoints);

% Generate the coefficients for the quintic splines that compose the
% value function interpolation
vEPoints = [scriptmVec scriptvVec scriptvPVec scriptvPPVec];
vECoeffs = generateInterpMatrix(vEPoints);

% These are the values of the precautionary saving interpolation (and its
% first three derivatives at the highest Euler point.  These values are
% used below to solve for parameters of an exponential function which
% serves as an extrapolation of precautionary saving above the highest
% Euler point.
s0 = psavEInterp(scriptmTop);
s1 = kappa - kappaTop;
s2 = -kappaPTop;
s3 = -(60*consumptionCoeffs(1,SAlength)*scriptmTop^2 + 24*consumptionCoeffs(2,SAlength)*scriptmTop + 6*consumptionCoeffs(3,SAlength));

% These values are taken from the Mathematica-generated solution.  When the
% parameter values are changed, the program will not find the correct
% extrapolation parameters.  The user is invited to explore this and find a
% better solution.
ephi0 = 0.00945646;
phi1 = 0.00462748;
egamma0 = -0.0730653;
gamma1 = 0.000626896;
cheatseed = [ephi0 phi1 egamma0 gamma1];
extrapParams = fsolve(@PhiGammaSolve,cheatseed);
ephi0 = extrapParams(1);
phi1 = extrapParams(2);
egamma0 = extrapParams(3);
gamma1 = extrapParams(4);

% phi0 = s0;
% phi1 = s1/s0;

if VerboseOutput == 1
    disp('Consumption and value functions are ready to be used.');
end
