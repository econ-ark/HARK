% This function plots the inputted function handle between the given minimum
% and maximum bounds.

function [X Y] = plotMyFunc(func,min, max)

X=[];
Y=[];
i = (max - min)/200;
for x = min:i:max
    X = [X; x];
    Y = [Y; feval(func,x)];
end
% plot(X,Y);

