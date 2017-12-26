

clear

global filename `1'
global interval_size `2'
global meas_err `3'
global instruments = "L(3/4).deltalogc L3.delta8logc L(3/4).deltalogy L3.delta8logy L(3/4).a"

*import data from file produced in Python
import delimited $filename, clear
tsset time_period
global num_regressions = floor(_N/$interval_size)

*use measurement error if required
if $meas_err==1 {
	drop deltalogc
	gen deltalogc = deltalogc_me
}

*Order: DeltaLogC_OLS, DeltaLogC_me_OLS, DeltaLogC_me_IV, DeltaLogY_IV, A_OLS, DeltaLogC_HR, DeltaLogY_HR, A_HR
matrix CoeffsArray = J(7,1,0)
matrix StdErrArray = J(7,1,0)
matrix RsqArray = J(5,1,0)
matrix PvalArray = J(5,1,0)
matrix OIDarray = J(5,1,0)
* ExtraInfo is 1) Number of L.deltalogc significant coeff, 2) Number of deltalogy significant coeff, 3) num_regressions, 4) Rsq of instruments on deltalogc
matrix ExtraInfo = J(4,1,0)


*OLS on consumption 
forvalues i = 1/$num_regressions {
	quietly regress deltalogc L.deltalogc if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix CoeffsArray[1,1] = CoeffsArray[1,1]+_b[L.deltalogc]/$num_regressions
	matrix StdErrArray[1,1] = StdErrArray[1,1]+_se[L.deltalogc]/$num_regressions
	matrix RsqArray[1,1] = RsqArray[1,1]+ e(r2_a)/$num_regressions
	matrix PvalArray[1,1] = PvalArray[1,1]+ e(idp)/$num_regressions
	matrix OIDarray[1,1] = OIDarray[1,1]+ e(jp)/$num_regressions
}

*IV on consumption 
forvalues i = 1/$num_regressions {
	quietly ivreg2 deltalogc (L.deltalogc = ${instruments}) if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix CoeffsArray[2,1] = CoeffsArray[2,1]+_b[L.deltalogc]/$num_regressions
	matrix StdErrArray[2,1] = StdErrArray[2,1]+_se[L.deltalogc]/$num_regressions
	matrix PvalArray[2,1] = PvalArray[2,1]+ e(idp)/$num_regressions
	matrix OIDarray[2,1] = OIDarray[2,1]+ e(jp)/$num_regressions
	*calc second stage R2
	qui reg L.deltalogc ${instruments} if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	cap drop first_stage_predict_c
	quietly predict first_stage_predict_c
	quietly reg deltalogc first_stage_predict_c if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix RsqArray[2,1] = RsqArray[2,1]+ e(r2_a)/$num_regressions
}

*IV on income 
forvalues i = 1/$num_regressions {
	quietly ivreg2 deltalogc (deltalogy = ${instruments}) if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix CoeffsArray[3,1] = CoeffsArray[3,1]+_b[deltalogy]/$num_regressions
	matrix StdErrArray[3,1] = StdErrArray[3,1]+_se[deltalogy]/$num_regressions
	matrix PvalArray[3,1] = PvalArray[3,1]+ e(idp)/$num_regressions
	matrix OIDarray[3,1] = OIDarray[3,1]+ e(jp)/$num_regressions
	*calc second stage R2
	quietly reg deltalogy ${instruments} if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	cap drop first_stage_predict_y
	quietly predict first_stage_predict_y
	quietly reg deltalogc first_stage_predict_y if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix RsqArray[3,1] = RsqArray[3,1]+ e(r2_a)/$num_regressions
}

*IV on assets 
forvalues i = 1/$num_regressions {
	quietly ivreg2 deltalogc (L.a = ${instruments}) if time_period >= ${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix CoeffsArray[4,1] = CoeffsArray[4,1]+_b[L.a]/$num_regressions
	matrix StdErrArray[4,1] = StdErrArray[4,1]+_se[L.a]/$num_regressions
	matrix PvalArray[4,1] = PvalArray[4,1]+ e(idp)/$num_regressions
	matrix OIDarray[4,1] = OIDarray[4,1]+ e(jp)/$num_regressions
	*calc second stage R2
	quietly reg L.a ${instruments} if time_period >= ${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	cap drop first_stage_predict_a
	quietly predict first_stage_predict_a
	quietly reg deltalogc first_stage_predict_a if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix RsqArray[4,1] = RsqArray[4,1]+ e(r2_a)/$num_regressions
}

*IV Horeserace 
forvalues i = 1/$num_regressions {
	quietly ivreg2 deltalogc (L.deltalogc deltalogy L.a = ${instruments}) if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix CoeffsArray[5,1] = CoeffsArray[5,1]+_b[L.deltalogc]/$num_regressions
	matrix CoeffsArray[6,1] = CoeffsArray[6,1]+_b[deltalogy]/$num_regressions
	matrix CoeffsArray[7,1] = CoeffsArray[7,1]+_b[L.a]/$num_regressions
	
	matrix StdErrArray[5,1] = StdErrArray[5,1]+_se[L.deltalogc]/$num_regressions
	matrix StdErrArray[6,1] = StdErrArray[6,1]+_se[deltalogy]/$num_regressions
	matrix StdErrArray[7,1] = StdErrArray[7,1]+_se[L.a]/$num_regressions
	
	matrix PvalArray[5,1] = PvalArray[5,1]+ e(idp)/$num_regressions
	matrix OIDarray[5,1] = OIDarray[5,1]+ e(jp)/$num_regressions
	if abs(_b[L.deltalogc]/_se[L.deltalogc])>1.96 {
		matrix ExtraInfo[1,1] = ExtraInfo[1,1] + 1
	}
	if abs(_b[deltalogy]/_se[deltalogy])>1.96 {
		matrix ExtraInfo[2,1] = ExtraInfo[2,1] + 1
	}
	
	*calc second stage R2
	quietly reg deltalogc first_stage_predict_c first_stage_predict_y first_stage_predict_a if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix RsqArray[5,1] = RsqArray[5,1]+ e(r2_a)/$num_regressions
	*calc R2 of deltalogc on instruments
	quietly reg deltalogc ${instruments}  if time_period >=${interval_size}*`i'-(${interval_size}-1) & time_period <=${interval_size}*`i', robust
	matrix ExtraInfo[4,1] = ExtraInfo[4,1] + e(r2_a)/$num_regressions
}
matrix ExtraInfo[3,1] = $num_regressions

*Hard code variance of measurement error - better to pass this in from the data file
*Can replace this once new data files are produced
qui sum deltalogc
matrix ExtraInfo[5,1] =  r(sd)*0.375

*Store results in a file
clear
svmat double CoeffsArray, name(CoeffsArray)
svmat double StdErrArray, name(StdErrArray)
svmat double RsqArray, name(RsqArray)
svmat double PvalArray, name(PvalArray)
svmat double OIDarray, name(OIDarray)
svmat double ExtraInfo, name(ExtraInfo)

rename CoeffsArray1 CoeffsArray
rename StdErrArray1 StdErrArray
rename RsqArray1 RsqArray
rename PvalArray1 PvalArray
rename OIDarray1 OIDarray
rename ExtraInfo1 ExtraInfo

export delimited "C:\Users\edmun\OneDrive\Documents\Research\HARK\cAndCwithStickyE\results\temp.txt", replace

exit, STATA clear



