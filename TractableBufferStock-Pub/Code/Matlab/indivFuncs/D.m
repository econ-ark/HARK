% This is my attempt at a derivative function.  It takes two arguments, a
% function handle and the value at which to evaluate the function.  For
% now, only pass it R-->R functions.

function x = D(func, value)
h = 0.00001; % Odd behavior occurs when h is set to very low values
x = (feval(func,value) - feval(func,value-h))/h;
