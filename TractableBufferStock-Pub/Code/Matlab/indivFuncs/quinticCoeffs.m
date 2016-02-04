% This function takes two inputs of 1x4 row vectors, (representing x, f(x),
% f'(x), and f''(x)) and returns a 1x6 vector of coefficients for the third degree
% polynomial described by this data.

function coeff = quinticCoeffs(x1data, x2data)
if size(x1data) ~= [1 4]
    error('First data point for interpolation is not formatted properly.');
end
if size(x2data) ~= [1 4]
    error('Second data point for interpolation is not formatted properly.');
end
x1 = x1data(1);
x2 = x2data(1);
Y = [x1data(2) ; x2data(2) ; x1data(3) ; x2data(3) ; x1data(4) ; x2data(4) ];
X = [ x1^5 x1^4 x1^3 x1^2 x1 1 ; x2^5 x2^4 x2^3 x2^2 x2 1 ; 5*x1^4 4*x1^3 3*x1^2 2*x1 1 0 ; 5*x2^4 4*x2^3 3*x2^2 2*x2 1 0 ; 20*x1^3 12*x1^2 6*x1 2 0 0 ; 20*x2^3 12*x2^2 6*x2 2 0 0];
coeff = X\Y;
