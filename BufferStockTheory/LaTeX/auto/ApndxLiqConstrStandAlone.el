(TeX-add-style-hook "ApndxLiqConstrStandAlone"
 (lambda ()
    (LaTeX-add-labels
     "sec:LiqConstrAppendix"
     "eq:EulerPFGICFails"
     "eq:EulerPFGICFailsEnd"
     "eq:cPreHist"
     "PDVc"
     "eq:bPound"
     "eq:bToInfty"
     "eq:FHWCfails"
     "fig:PFGICHoldsFHWCFailsRICFails")
    (TeX-add-symbols
     '("Metin" 2))
    (TeX-run-style-hooks
     "latex2e"
     "bejournal10"
     "bejournal"
     "CDCDocStartForBE"
     "../tables/LiqConstrScenarios"
     "./ApndxLiqConstr")))

