'''
This module contains function(s) for producing LaTeX tables from simulated model results.
'''
import numpy as np

def mystr(number):
    if not np.isnan(number):
        out = "{:.3f}".format(number)
    else:
        out = ''
    return out


def mystr2(number):
    if not np.isnan(number):
        out = "{:.4f}".format(number)
    else:
        out = ''
    return out


def makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,Caption,filename):
    '''
    Make simulated results table function.
    '''
    output = '\\begin{table}\caption{' + Caption + '}\n'
    output += '\\begin{tabular}{cccccc}\n \hline \hline'
    output += '\multicolumn{6}{c}{$ \Delta \log \mathbf{C}_{t+1} = \\varsigma + \chi \Delta \log \mathbf{C}_t + \eta \mathbb{E}_t[\Delta \log \mathbf{Y}_{t+1}] + \\alpha A_t + \epsilon $ } \n'
    output += '\\\\ \multicolumn{3}{c}{Expectations : Dep Var} & OLS &  (2nd Stage) & $F~p$-val \n'
    output += '\\\\ \multicolumn{3}{c}{Independent Variables} & or IV & $ \\bar{R}^{2} $ & IV OID \n'
    output += '\\\\ \hline \multicolumn{3}{c}{Frictionless : $\Delta \log \mathbf{C}_{t+1}$} & & & \n'
    output += '\\\\ \multicolumn{1}{c}{$\Delta \log \mathbf{C}_{t} $} & \multicolumn{1}{c}{$\Delta \log \mathbf{Y}_{t+1}$}& \multicolumn{1}{c}{$ A_{t}  $} & & & \n'
    output += '\\\\ ' + mystr(Coeffs[0]) + ' & & & OLS & ' + mystr(Rsq[0]) + '& ' + mystr(Pvals[0]) +'\n'   
    output += '\\\\ (' + mystr(StdErrs[0]) + ') & & & & & \n'   
    output += '\\\\ & ' + mystr(Coeffs[1])  + ' & & IV & ' + mystr(Rsq[1]) + ' & ' + mystr(Pvals[1]) + '\n'    
    output += '\\\\ & (' + mystr(StdErrs[1]) + ') & & & & ' + mystr(OID[1]) + '\n'             
    output += '\\\\ & & ' + mystr2(Coeffs[2]) + ' & IV & ' + mystr(Rsq[2]) + ' & ' + mystr(Pvals[2]) + '\n'    
    output += '\\\\ & & (' + mystr2(StdErrs[2]) + ') & & & ' + mystr(OID[2]) + '\n'   
    output += '\\\\ ' + mystr(Coeffs[3]) + ' & ' + mystr(Coeffs[4]) + ' & ' + mystr2(Coeffs[5]) + ' & IV & ' + mystr(Rsq[3]) + ' & ' + mystr(Pvals[3]) +'\n'         
    output += '\\\\ (' + mystr(StdErrs[3]) + ') & (' + mystr(StdErrs[4]) + ') & (' + mystr2(StdErrs[5]) + ') & & & \n'
    output += '\\\\ & \multicolumn{4}{c}{Memo: For instruments $\mathbf{Z}_{t}$,  $\Delta \log \mathbf{C}_{t+1} = \mathbf{Z}_{t} \zeta,~~\\bar{R}^{2}=$ } ' + mystr(Counts[4]) + ' & \n'
    #output += '\\\\ \multicolumn{6}{c}{} \n'
    output += '\\\\ \hline \multicolumn{3}{c}{Sticky : $\Delta \log \mathbf{C}_{t+1}$} %NotOnSlide \n'
    output += '\\\\ \multicolumn{1}{c}{$\Delta \log {\mathbf{C}}_{t}$} & \multicolumn{1}{c}{$\Delta \log \mathbf{Y}_{t+1}$} & \multicolumn{1}{c}{$A_{t}$} \n'
    output += '\\\\  ' + mystr(Coeffs[6]) + ' & & & OLS & ' + mystr(Rsq[4]) + ' & ' + mystr(Pvals[4]) + '%NotOnSlide \n'
    output += '\\\\  (' + mystr(StdErrs[6]) + ') & & & & & %NotOnSlide \n'
    #output += '\\\\ \multicolumn{6}{c}{} \n'
    output += '\\\\ \hline \multicolumn{3}{c}{Sticky : $\Delta \log \widetilde{\mathbf{C}}_{t+1} $}%NotOnSlide \n'
    output += '\\\\ \multicolumn{1}{c}{$\Delta \log {\widetilde{\mathbf{C}}}_{t}$} & \multicolumn{1}{c}{$\Delta \log \mathbf{Y}_{t+1}$} & \multicolumn{1}{c}{$A_{t}$} \n'
    output += '\\\\ ' + mystr(Coeffs[7]) + ' & & & OLS & ' + mystr(Rsq[5]) + ' & ' + mystr(Pvals[5]) + '\n'   
    output += '\\\\ (' + mystr(StdErrs[7]) + ') & & & & & \n'   
    output += '\\\\ ' + mystr(Coeffs[8]) + ' & & & IV & ' + mystr(Rsq[6]) + ' & ' + mystr(Pvals[6]) + '\n'   
    output += '\\\\ (' + mystr(StdErrs[8]) + ') & & & & &' + mystr(OID[6]) + '\n'   
    output += '\\\\ & ' + mystr(Coeffs[9]) + ' & & IV & ' + mystr(Rsq[7]) + ' & ' + mystr(Pvals[7]) + '\n'     
    output += '\\\\ & (' + mystr(StdErrs[9]) + ') & & & &' + mystr(OID[7]) + '\n'    
    output += '\\\\ & & ' + mystr2(Coeffs[10]) + ' & IV & ' + mystr(Rsq[8]) + ' & ' + mystr(Pvals[8]) + '\n'   
    output += '\\\\ & & (' + mystr2(StdErrs[10]) + ') & & &' + mystr(OID[8]) + '\n'    
    output += '\\\\ ' + mystr(Coeffs[11]) + ' & ' + mystr(Coeffs[12]) + ' & ' + mystr2(Coeffs[13]) + ' & IV & ' + mystr(Rsq[9]) + ' & ' + mystr(Pvals[9]) + '\n'     
    output += '\\\\ (' + mystr(StdErrs[11]) + ') & (' + mystr(StdErrs[12]) + ') & (' + mystr2(StdErrs[13]) + ') & & & \n'
    output += '\\\\ & \multicolumn{4}{c}{Memo: For instruments $\mathbf{Z}_{t}$,  $\Delta \log \widetilde{\mathbf{C}}_{t+1} = \mathbf{Z}_{t} \zeta,~~\\bar{R}^{2}=$ } ' + mystr(Counts[3]) + ' & \n'
    output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on $\Delta \log \widetilde{\mathbf{C}}_t$ significant at 95\% level for ' + str(Counts[0]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'
    output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on $\mathbb{E}[\Delta \log \mathbf{Y}_{t+1}]$ significant at 95\% level for ' + str(Counts[1]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'
    output += '\\\\ \hline \hline \n'
    output += '\end{tabular} \n'
    output += '\end{table} \n'
    
    with open('./Tables/' + filename + '.txt','w') as f:
        f.write(output)
        f.close()
