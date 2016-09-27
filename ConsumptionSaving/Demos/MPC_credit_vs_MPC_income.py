"""
This is a HARK demo.  

The application here is to examine the Marginal Propensity to Consume (MPC) out of an increase in
a credit limit, and to compare it to the MPC out of temporary income.

This demo is very heavily commented so that HARK newcomers can use it to figure out how HARK works.
It also does things, like import modules in the body of the code rather than at the top, that
are typically deprecated by Python programmers.  This is all to make the code easier to read
and understand.

There are many ways to use HARK, and this demo cannot show them all.  
This demo demonstrates one great way to use HARK: import and solve a model for different parameter
values, to see how parameters affect the solution.
"""


####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.
"""

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks.
from ConsIndShockModel import IndShockConsumerType

## Import the default parameter values
import ConsumerParameters as Params

## Now, create an instance of the consumer type using the default parameter values
## We create the instance of the consumer type by calling IndShockConsumerType()
## We use the default parameter values by passing **Params.init_idiosyncratic_shocks as an argument
BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)

## Note: we've created an instance of a very standard consumer type, and many assumptions go
## into making this kind of consumer.  As with any structural model, these assumptions matter.
## For example, this consumer pays the same interest rate on 
## debt as she earns on savings.  If instead we wanted to solve the problem of a consumer
## who pays a higher interest rate on debt than she earns on savings, this would be really easy,
## since this is a model that is also solved in HARK.  All we would have to do is import that model
## and instantiate an instance of that ConsumerType instead.  As a homework assignment, we leave it
## to you to uncomment the two lines of code below, and see how the results change!
#from ConsIndShockModel import KinkedRconsumerType
#BaselineExample = KinkedRconsumerType(**Params.init_kinked_R)



####################################################################################################
####################################################################################################

"""
The next step is to change the values of parameters as we want.

To see all the parameters used in the model, along with their default values, see
ConsumerParameters.py

Parameter values are stored as attributes of the ConsumerType the values are used for.
For example, the risk-free interest rate Rfree is stored as BaselineExample.Rfree.
Because we created BaselineExample using the default parameters values.
at the moment BaselineExample.Rfree is set to the default value of Rfree (which, at the time
this demo was written, was 1.03).  Therefore, to change the risk-free interest rate used in 
BaselineExample to (say) 1.02, all we need to do is:

BaselineExample.Rfree = 1.02
"""

## Change some parameter values
BaselineExample.Rfree       = 1.02 #change the risk-free interest rate
BaselineExample.CRRA        = 2.   # change  the coefficient of relative risk aversion
BaselineExample.BoroCnstArt = -.3  # change the artificial borrowing constraint
BaselineExample.DiscFac     = .5   #chosen so that target debt-to-permanent-income_ratio is about .1
                                   # i.e. BaselineExample.solution[0].cFunc(.9) ROUGHLY = 1.

## There is one more parameter value we need to change.  This one is more complicated than the rest.
## We could solve the problem for a consumer with an infinite horizon of periods that (ex-ante)
## are all identical.  We could also solve the problem for a consumer with a fininite lifecycle,
## or for a consumer who faces an infinite horizon of periods that cycle (e.g., a ski instructor
## facing an infinite series of winters, with lots of income, and summers, with very little income.)
## The way to differentiate is through the "cycles" attribute, which indicates how often the
## sequence of periods needs to be solved.  The default value is 1, for a consumer with a finite
## lifecycle that is only experienced 1 time.  A consumer who lived that life twice in a row, and
## then died, would have cycles = 2.  But neither is what we want.  Here, we need to set cycles = 0,
## to tell HARK that we are solving the model for an infinite horizon consumer.


## Note that another complication with the cycles attribute is that it does not come from 
## Params.init_idiosyncratic_shocks.  Instead it is a keyword argument to the  __init__() method of 
## IndShockConsumerType.
BaselineExample.cycles      = 0  


####################################################################################################
####################################################################################################

"""
Now, create another consumer to compare the BaselineExample to.
"""
# The easiest way to begin creating the comparison example is to just copy the baseline example.
# We can change the parameters we want to change later.
from copy import deepcopy
XtraCreditExample = deepcopy(BaselineExample)


# Now, change whatever parameters we want.
# Here, we want to see what happens if we give the consumer access to more credit.
# Remember, parameters are stored as attributes of the consumer they are used for.
# So, to give the consumer more credit, we just need to relax their borrowing constraint a bit.

# Declare how much we want to increase credit by
credit_change =  .001

# Now increase the consumer's credit limit.  
# We do this by decreasing the artificial borrowing constraint.
XtraCreditExample.BoroCnstArt = BaselineExample.BoroCnstArt - credit_change



####################################################################################################
"""
Now we are ready to solve the consumers' problems.
In HARK, this is done by calling the solve() method of the ConsumerType.
"""

### First solve the baseline example.
BaselineExample.solve()

### Now solve the comparison example of the consumer with a bit more credit
XtraCreditExample.solve()



####################################################################################################
"""
Now that we have the solutions to the 2 different problems, we can compare them
"""

## We are going to compare the consumption functions for the two different consumers.
## Policy functions (including consumption functions) in HARK are stored as attributes
## of the *solution* of the ConsumerType.  The solution, in turn, is a list, indexed by the time
## period the solution is for.  Since in this demo we are working with infinite-horizon models
## where every period is the same, there is only one time period and hence only one solution.
## e.g. BaselineExample.solution[0] is the solution for the BaselineExample.  If BaselineExample
## had 10 time periods, we could access the 5th with BaselineExample.solution[4] (remember, Python
## counts from 0!)  Therefore, the consumption function cFunc from the solution to the
## BaselineExample is BaselineExample.solution[0].cFunc


## First, declare useful functions to plot later

def FirstDiffMPC_Income(x):
    # Approximate the MPC out of income by giving the agent a tiny bit more income,
    # and plotting the proportion of the change that is reflected in increased consumption

    # First, declare how much we want to increase income by
    # Change income by the same amount we change credit, so that the two MPC
    # approximations are comparable
    income_change = credit_change

    # Now, calculate the approximate MPC out of income
    return (BaselineExample.solution[0].cFunc(x + income_change) - 
            BaselineExample.solution[0].cFunc(x)) / income_change


def FirstDiffMPC_Credit(x):
    # Approximate the MPC out of credit by plotting how much more of the increased credit the agent
    # with higher credit spends
    return (XtraCreditExample.solution[0].cFunc(x) - 
            BaselineExample.solution[0].cFunc(x)) / credit_change 



## Now, plot the functions we want

# Import a useful plotting function from HARKutilities
from HARKutilities import plotFuncs
import pylab as plt # We need this module to change the y-axis on the graphs


# Declare the upper limit for the graph
x_max = 10.


# Note that plotFuncs takes four arguments: (1) a list of the arguments to plot, 
# (2) the lower bound for the plots, (3) the upper bound for the plots, and (4) keywords to pass
# to the legend for the plot.

# Plot the consumption functions to compare them
print('Consumption functions:')
plotFuncs([BaselineExample.solution[0].cFunc,XtraCreditExample.solution[0].cFunc],
           BaselineExample.solution[0].mNrmMin,x_max,
           legend_kwds = {'loc': 'upper left', 'labels': ["Baseline","XtraCredit"]})


# Plot the MPCs to compare them
print('MPC out of Credit v MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs([FirstDiffMPC_Credit,FirstDiffMPC_Income],
          BaselineExample.solution[0].mNrmMin,x_max,
          legend_kwds = {'labels': ["MPC out of Credit","MPC out of Income"]})

