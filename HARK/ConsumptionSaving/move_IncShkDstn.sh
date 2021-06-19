#!/bin/bash

dest='IncShk'

for fact in permPos tranPos permShkValsBcst tranShkValsBcst permShkPrbs permShkVals tranShkPrbs tranShkVals permShkMin tranShkMin UnempPrb WorstIncPrb; do
    cmd="rpl -Rfs 'bilt."$fact"' 'bilt."$dest"."$fact"' ConsIndShockModel*.*"
    echo "$cmd"
done

rpl -Rs 'hasattr(bilt, "tranShkVals")' 'hasattr(pars.IncShks, "tranShkVals")' *


