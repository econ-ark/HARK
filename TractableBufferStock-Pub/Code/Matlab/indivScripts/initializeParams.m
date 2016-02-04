% This script defines the base values for parameters in the TBS model.
% It then calls a script to define the working values of these parameters.

epsilon = 0.0001;
mhoBase = 0.05;
thetaBase = 0.10;
littleRBase = 0.05;
littleGBase = -0.02;
rhoBase = 1.01;
VerboseOutput = 1;

if VerboseOutput == 1
    disp('Output will be verbose.');
    disp('Parameter base values have been set.');
end

resetParams;
