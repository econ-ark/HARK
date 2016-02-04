% This function adds another consumption and asset point to a path of (m,c)
% points by using the appropriate time iterating functions.

function nextPoint = SimAddAnotherPoint(mcPath)
last = length(mcPath);
mNextVal = scriptmEtp1Fromt(mcPath(last,1),mcPath(last,2));
cNextVal = cE(mNextVal);
nextPoint = [mNextVal cNextVal];
