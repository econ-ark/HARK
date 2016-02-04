% This function is used to solve for the values of ephi0, phi1, egamma0, and
% gamma1 so that they can be used in psavEExtrap.

function parameters = PhiGammaSolve(x)
global s0 s1 s2 s3;
parameters = [x(1) - x(3) - s0;
              -x(1) * x(2) + x(3) * x(4) - s1 ;
              x(1) * x(2)^2 - x(3) * x(4)^2 - s2;
              -x(1) * x(2)^3 + x(3) * x(4)^3 - s3];
