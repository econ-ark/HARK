This example calculates the relationship between the degree of uncertainty
and the aggregate saving rate in a cstwMPC economy

This example includes three files: 
1. SetupParamsCSTWUncert.py: set value for all parameters except time discount 
factor and standard deviation of permanent and transitory shocks
2. PermShocksAggSaving.py: calculates the relationship between the standard 
deviation of the permanent income shock and the aggregate saving rate
3. TranShocksAggSaving.py: calculates the relationship between the standard 
deviation of the transitory income shock and the aggregate saving rate

Further Detail:
1. SetupParamsCSTWUncert.py
(1)SetupParamsCSTWUncert.py includes parameters for both infinite horizon model 
and life-cycle model. For this example, we only use the parameters for infinite 
horizon model. 
(2) In order to use the parameter value in SetupParamsCSTWUncert.py, you need to 
open cstwMPC.py inside HARK/cstwMPC folder, change line 20:
"import SetupParamsCSTW as Params" to be "import SetupParamsCSTWUncert as Params"
(3) "Rfree" passed over to the agent type from the SetupParamsCSTWUncert.py is the 
effective interest rate. In this example, it is 1.01/LivPrb_i[0]

2. PermShocksAggSaving.py
(1) First choose the directory you want to save the output graphs, type the
directory path into variable "Folder_path".
(2) Boolean variable "Params.do_param_dist" controls if the code is solving "beta
-point" or "beta-dist" model. If Params.do_param_dist is True, the code allow time
discount factor to be heterogeneous among agents, solves "beta-dist" model. If it
is False, the code solves "beta-point" model.
(3) Boolean variable "do_optimizing" controls if the code is using the beta 
estimation that it calucated before or reestimate beta. You should reestimate beta 
every time you change the parameters that are passed from SetupParamsCSTWUncert.py
(4) Set benchmark standard deviation for permanent and transitory shocks in 
"BaselineType.PermShkStd" and "BaselineType.TranShkStd"
(5)"center_pre" and "spread_pre" are the previously calculated mean and spread 
estimation for beta. Update these two variables after you reestimated beta.
(6) After estimation of beta, the code will calculate the growth impatience factor 
for different value of permanent shock standard deviation. 
Notice that the interest rate we use to calculate the growth impatience factor 
is the effective interest rate passed from SetupParamsCSTWUncert.py times the 
survivall rate. The growth impatience condition restrict the maximum standard deviation of permanent shocks we can choose.
(7) After calculate the growth impatience factor, the next part of programs 
calculate the aggregate saving rate for any given value of permanent shock standard 
deviation.
(8) The the program plots the relationship between the aggregate saving rate and 
the standard deviation of permanent shocks

3.TranShocksAggSaving.py
(1) TranShocksAggSaving.py solves the relationship between aggregate saving rate 
and the standard deviation of idiosyncratic transitory income shocks.
(2) The steps to run this program is the same as the steps to run 
PermShocksAggSaving.py. Notice now you are increasing the standard deviation of 
transitory income shocks from half of its benchmark value to twice the benchmark 
value.
(3) Changing the standard deviation of transitory income shock does not change 
growth impationce factor. The program still report the growth impatience factor 
value to make sure the benchmark calibration does not violate the GIC
