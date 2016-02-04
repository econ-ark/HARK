(TeX-add-style-hook "ApndxLiqConstr"
 (lambda ()
    (LaTeX-add-labels
     "eq:EulerPFGICFails"
     "eq:EulerPFGICFailsEnd"
     "eq:cPreHist"
     "PDVc"
     "eq:bPound"
     "eq:bToInfty"
     "eq:FHWCfails"
     "fig:PFGICHoldsFHWCFailsRICFails")
    (TeX-run-style-hooks
     "../tables/LiqConstrScenarios")))

