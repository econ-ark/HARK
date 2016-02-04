(* ::Package:: *)

(* Analytical solutions for consumption, PDV of consumption, human wealth *)
(* bank balances, monetary resources, and their first two *)
(* derivatives as a function of the periods n until the constraint binds *)

(* These construct prehistory if PF-GIC holds *)
(* See ApndxLiqConstrStandAlone.tex and Derivations.nb for derivations *)
c\[Sharp][n_]     := \[CapitalThorn]\[CapitalGamma]PF^-n;
dc\[Sharp]dn[n_]  := - \[CapitalThorn]\[CapitalGamma]PF^-n Log[\[CapitalThorn]\[CapitalGamma]PF];
dc\[Sharp]dnn[n_] :=   \[CapitalThorn]\[CapitalGamma]PF^-n (Log[\[CapitalThorn]\[CapitalGamma]PF])^2;
\[DoubleStruckCapitalC]\[Sharp][n_]     := c\[Sharp][n]  (1-\[CapitalThorn]Rtn^(n+1))/(1-\[CapitalThorn]Rtn);
d\[DoubleStruckCapitalC]\[Sharp]dn[n_]  := dc\[Sharp]dn[n](1-\[CapitalThorn]Rtn^(n+1))/(1-\[CapitalThorn]Rtn)-c\[Sharp][n](Log[\[CapitalThorn]Rtn]\[CapitalThorn]Rtn^(n+1))/(1-\[CapitalThorn]Rtn);
d\[DoubleStruckCapitalC]\[Sharp]dnn[n_] :=-((\[CapitalThorn]Rtn^(1+n) dc\[Sharp]dn[n] Log[\[CapitalThorn]Rtn])/(1-\[CapitalThorn]Rtn))-(\[CapitalThorn]Rtn^(1+n) c\[Sharp][n] (Log[\[CapitalThorn]Rtn]^2))/(1-\[CapitalThorn]Rtn)-(\[CapitalThorn]Rtn^(1+n) Log[\[CapitalThorn]Rtn] dc\[Sharp]dn[n])/(1-\[CapitalThorn]Rtn)+((1-\[CapitalThorn]Rtn^(1+n)) dc\[Sharp]dnn[n])/(1-\[CapitalThorn]Rtn);
\[HBar]\[Sharp][n_]     := (1-\[ScriptCapitalR]^-(n+1))/(1-\[ScriptCapitalR]^-1);
d\[HBar]\[Sharp]dn[n_]  :=   (\[ScriptCapitalR]^-(n+1)Log[\[ScriptCapitalR]])/(1-\[ScriptCapitalR]^-1);
d\[HBar]\[Sharp]dnn[n_] := -(\[ScriptCapitalR]^(-1-n) (Log[\[ScriptCapitalR]])^2)/(1-1/\[ScriptCapitalR]);
b\[Sharp][n_]     := \[DoubleStruckCapitalC]\[Sharp][n]   - \[HBar]\[Sharp][n];
dcdb[n_]   := dc\[Sharp]dn[n]/db\[Sharp]dn[n];
dcdbb[n_]  := (dc\[Sharp]dnn[n]/db\[Sharp]dn[n]-(dc\[Sharp]dn[n]db\[Sharp]dnn[n]/(db\[Sharp]dn[n]^2)))/db\[Sharp]dn[n];
db\[Sharp]dn[n_]  := d\[DoubleStruckCapitalC]\[Sharp]dn[n]-d\[HBar]\[Sharp]dn[n];
db\[Sharp]dnn[n_] := d\[DoubleStruckCapitalC]\[Sharp]dnn[n]-d\[HBar]\[Sharp]dnn[n];
m\[Sharp][n_]     := b\[Sharp][n]+1;
dm\[Sharp]dn[n_]  := db\[Sharp]dn[n];
dm\[Sharp]dnn[n_] := db\[Sharp]dnn[n];
s\[Sharp][n_]     := \[ScriptC]\[Digamma]Inf[m\[Sharp][n]]-c\[Sharp][n];
ds\[Sharp]dn[n_]  := \[ScriptC]\[Digamma]Inf'[m\[Sharp][n]]dm\[Sharp]dn[n]-dc\[Sharp]dn[n];
n\[Sharp][m_]     := n /. FindRoot[m\[Sharp][n]==m,{n,- Log[m]/Log[\[CapitalThorn]\[CapitalGamma]PF]}];
v\[Sharp][n_]     := u[c\[Sharp][n]](1-(\[Beta] \[CapitalThorn]^(1-\[Rho]))^(n+1))/(1-\[Beta] \[CapitalThorn]^(1-\[Rho]))+u[1]((\[CapitalGamma]^(1-\[Rho]))\[Beta])^(n+1) /(1-\[Beta] \[CapitalGamma]^(1-\[Rho]));
dv\[Sharp]dn[n_]   := (\[CapitalGamma]^\[Rho] (\[Beta] \[CapitalGamma]^(1-\[Rho]))^n Log[\[Beta] \[CapitalGamma]^(1 - \[Rho])] u[1])/(-\[Beta] \[CapitalGamma] + \[CapitalGamma]^\[Rho]) + 
              (\[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n Log[\[CapitalThorn]^(1 - \[Rho]) \[Beta]] u[c\[Sharp][n]])/(-\[CapitalThorn]^\[Rho] + \[CapitalThorn] \[Beta]) + 
              (\[CapitalThorn]^\[Rho] - \[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n) c\[Sharp][n]^-\[Rho] dc\[Sharp]dn[n]/(\[CapitalThorn]^\[Rho] - \[CapitalThorn] \[Beta]);
dv\[Sharp]dnn[n_]  := (c\[Sharp][n]^(-1-\[Rho]) (c\[Sharp][n]^(1+\[Rho]) (-(\[CapitalThorn]^\[Rho]-\[CapitalThorn] \[Beta]) \[CapitalGamma]^\[Rho] (\[Beta] \[CapitalGamma]^(1-\[Rho]))^n Log[\[Beta] \[CapitalGamma]^(1-\[Rho])]^2 u[1]+\[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n (-\[Beta] \[CapitalGamma]+\[CapitalGamma]^\[Rho]) Log[\[CapitalThorn]^(1-\[Rho]) \[Beta]]^2 u[c\[Sharp][n]])+
              (\[CapitalThorn]^\[Rho]-\[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n) (-\[Beta] \[CapitalGamma]+\[CapitalGamma]^\[Rho]) \[Rho] dc\[Sharp]dn[n]^2+
              (-\[Beta] \[CapitalGamma]+\[CapitalGamma]^\[Rho]) c\[Sharp][n] (2 \[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n Log[\[CapitalThorn]^(1-\[Rho]) \[Beta]] dc\[Sharp]dn[n]-(\[CapitalThorn]^\[Rho]-\[CapitalThorn] \[Beta] (\[CapitalThorn]^(1-\[Rho]) \[Beta])^n) dc\[Sharp]dnn[n])))/((\[CapitalThorn]^\[Rho]-\[CapitalThorn] \[Beta]) (\[Beta] \[CapitalGamma]-\[CapitalGamma]^\[Rho]));
dvdm[n_]   := dv\[Sharp]dn[n]/dm\[Sharp]dn[n];
dvdmm[n_]  := (dv\[Sharp]dnn[n]dm\[Sharp]dn[n]-dv\[Sharp]dn[n]dm\[Sharp]dnn[n])/(dm\[Sharp]dn[n]^3);
nFromLasta\[Sharp][a_]:= nSeek /. FindRoot[m\[Sharp][nSeek]-c\[Sharp][nSeek]==a,{nSeek,-Log[a]/Log[\[CapitalThorn]\[CapitalGamma]PF]}];
cLiqExact[m_] := If[m>m\[Sharp][1],c\[Sharp][n\[Sharp][m]],m];
\[Kappa]LiqExact[m_] := If[m>m\[Sharp][1],dcdb[n\[Sharp][m]],1.];
SetAttributes[{cLiqExact,\[Kappa]LiqExact,v\[Sharp],dvdm,dvdmm},Listable];
