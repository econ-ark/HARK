% This script ensures that the consumer specified by the parameters is
% growth impatient and terminates operation if he is not.

if scriptPGrowth >= 1
    error('Aborting: Employed Consumer Not Growth Impatient.')
end
