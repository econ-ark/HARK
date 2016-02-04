(* ::Package:: *)

Params={"\[CapitalGamma]","(R)","\[Beta]","\[Rho]"," \[WeierstrassP] ","\[Sigma]Perm","\[Sigma]Tran","FHW","FVAPF","\[Bet]","\[DoubleStruckCapitalE]pShkInv","\[CapitalGamma]Adj","\[CapitalThorn]","\[CapitalThorn]Rtn","\[CapitalThorn]\[CapitalGamma]PF","\[CapitalThorn]\[CapitalGamma]"};
Print[MatrixForm[Transpose[{Params,Map[ToExpression[#]&,Params]}]]]
