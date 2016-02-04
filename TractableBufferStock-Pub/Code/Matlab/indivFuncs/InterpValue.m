% This function takes three inputs: the value at which to evaluate, the set
% of interpolation points, and the coefficients for those points (from
% generateInterpMatrix) and returns the value of the piecewise quintic
% spline at the requested value.

function output = InterpValue(input,points,coeffs)
globalizeTBSvars;
% Make sure the requested value is between the highest and lowest
% interpolation points; terminate if not.
if input < points(1,1)
    error(['Requested input value (' num2str(input) ') is below lowest interpolation point (' num2str(points(1,1)) ').']);
end
if input > points(length(points),1)
    error(['Requested input value (' num2str(input) ') is above highest interpolation point (' num2str(points(length(points),1)) ').']');
end

% Perform a binary search for the interpolation points between which the
% input falls.
found = 0;
searchsize = floor(log(length(points))/log(2));
j = searchsize - 1;
searchAt = 2^searchsize;
while (j >= 0 && found == 0)
    if searchAt < length(points)
        if ((input >= points(searchAt,1)) && (input <= points(searchAt+1,1)))
            found = 1;
        elseif (input > points(searchAt,1))
            searchAt = searchAt + 2^j;
        elseif (input < points(searchAt,1))
            searchAt = searchAt - 2^j;
        end
    else
        searchAt = searchAt - 2^j;
    end
    j = j-1;
end

% Return the value of the quintic spline using the correct coefficients
output = [input^5 input^4 input^3 input^2 input 1] * coeffs(:,searchAt);
