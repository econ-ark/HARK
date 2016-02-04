% This function sorts a bottom arm of Euler points, inverting it to
% smallest amount of assets to largest.

function flippedlist = sortLowerArm(EulerPoints)
tempLowerArm1 = sort(EulerPoints);
tempLowerArm2 = sort(EulerPoints, 'descend');
flippedlist = [tempLowerArm1(:,[1 2]) tempLowerArm2(:,3) tempLowerArm1(:,[4 5])];
