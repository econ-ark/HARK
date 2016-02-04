(TeX-add-style-hook "ApndxSolnMethEndogGptsStandAlone"
 (lambda ()
    (LaTeX-add-labels
     "eq:cEulerEndog"
     "eq:MPTHC"
     "eq:MPCfromMPTHC")
    (TeX-add-symbols
     '("Metin" 2))
    (TeX-run-style-hooks
     "latex2e"
     "bejournal10"
     "bejournal"
     "CDCDocStartForBE"
     "./ApndxSolnMethEndogGpts"
     "bibMake")))

