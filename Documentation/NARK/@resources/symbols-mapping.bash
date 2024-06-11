#!/bin/bash

letter_rules=(
    "\\aNrm:a"
    "\\bNrm:b"
    "\\cNrm:c"
    "\\dNrm:d"
    "\\eNrm:e"
    "\\hNrm:h"
    "\\wNrm:w"
    "\\pNrm:p"
    "\\kNrm:k"
    "\\mNrm:m"
    "\\vNrm:v"
    "\\xNrm:x"
    "\\yNrm:y"
    "\\yNrm:y"
    "\\zNrm:z"
    "\\prd:t"
    "\\prdt:t"
    "\\prdT:t"
    "\\trmT:T"
)
unicode_rules=(
    "\\aLvl:𝐚"
    "\\bLvl:𝐛"
    "\\cLvl:𝐜"
    "\\hLvl:𝐡"
    "\\pLvl:𝐩"
    "\\kLvl:𝐤"
    "\\mLvl:𝐦"
    "\\vLvl:𝐯"
    "\\yLvl:𝐲"
    "\\Ex:𝔼"
    "\\PermGroFac:𝒢"
    "\\PermGroFacAdj: "
    "\\PermGroFacAdj: "
    "\\RNrm:ℛ"
    "\\vFunc:𝚟"
    "\\uFunc:𝚞"
    "\\cFunc:𝚌"
    "\\DiscFac:β"
    "\\std:σ"
    "\\CRRA:ρ"
    "\\Rfree:R"
    "\\Risky:𝐑"
    "\\Rport:ℜ"
    "\\Shr:Ϛ"
    "\\TranShkEmp:θ"
    "\\TranShkEmpDum:ϑ"
    "\\tranShkEmp:θ"
    "\\tranShkEmpDum:ϑ"
    "\\Nrml:𝒩"
    "\\arvl:←"
    "\\cntn:→"
    "\\BegMark:←"
    "\\EndMark:→"
    "\\wlthAftr:ẃ"
    "\\wlthBefr:w"
    "\\labor:ℓ"
    )

generate_prettify_rules() {
    output_file="symbols-mapping-prettify"
    > "$output_file"

    for rule in "${letter_rules[@]}"; do
	IFS=':' read -r command symbol <<< "$rule"
	echo "(\"$command\" . ?$symbol)" >> "$output_file"
    done

    for rule in "${unicode_rules[@]}"; do
	IFS=':' read -r command symbol <<< "$rule"
	echo "(\"$command\" . ?$symbol)" >> "$output_file"
    done

    echo "Prettify rules generated in $output_file"
}

generate_prettify_rules

generate_latex_commands() {
    output_file="symbols-mapping-latex.tex"
    > "$output_file"

    for rule in "${letter_rules[@]}"; do
	IFS=':' read -r command symbol <<< "$rule"
	echo "\\newcommand{$command}{$symbol}" >> "$output_file"
    done

    echo "LaTeX commands generated in $output_file"
}

generate_latex_commands
