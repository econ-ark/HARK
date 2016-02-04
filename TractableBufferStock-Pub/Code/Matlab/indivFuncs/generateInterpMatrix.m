% This function takes an nx4 matrix, representing n data points in [x f(x) f'(x) f''(x)]
% format, and returns a 6x(n-1) matrix of coefficients representing the
% piecewise fifth degree polynomials for this data.

function matrix = generateInterpMatrix(dataPoints)
if size(dataPoints,2) ~= 4
    error('Data for generating coefficient matrix not formatted properly.');
end
dataLength = size(dataPoints,1) - 1;
matrix = [];
for j = 1:dataLength
    matrix = [matrix quinticCoeffs(dataPoints(j,:),dataPoints(j+1,:))];
end
