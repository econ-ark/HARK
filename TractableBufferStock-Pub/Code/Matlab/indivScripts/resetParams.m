% This script sets the parameters to their base values.
% It then updates other values dependent on these parameters.

mho = mhoBase;
theta = thetaBase;
littleR = littleRBase;
littleG = littleGBase;
rho = rhoBase;

if VerboseOutput==1
    disp('Parameters have been reset to their base values.');
end

setValues;
