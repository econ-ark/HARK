Suggested Variable Naming Conventions in the Econ-ARK Toolkit
=============================================================

Sharing code is easier when different contributors
use similar names for similar objects.
While we will not enforce the recommendations below,
contributors can make their code more attractive to others
by using names consistent with our guidelines.

Principles
==========

Our aim has been to balance:

*  Brevity
*  Mnemonic quality
*  Uniqueness (enabling global search-and-replace)
*  Ubiquity (definitions useful for many projects)
*  Combinatoriality (which encourages mashups)

Variables
=========

Single-Letter
-------------

We strongly discourage the use of single-letter variable names.
Reasons are presented *ad nauseum* in introductory CS texts
(which, we know, few economists consult);
here we will note only that the extent to which your code will be influential
depends upon the extent to which someone else can easily read it, which
is harder if you have used variable names which could mean almost anything.
(We have made careful choices for the
'default' meanings of each Roman letter (see below);
hence, in the context of the toolkit,
use of a single-letter name for any meaning other than
the designated one will be doubly confusing).

It is only slightly less objectionable to name a variable after a
familiar letter in another commonly used alphabet (say, ``delta``).
Your future self (and other users) will not know which of
the many possible meanings of :math:`\delta` you had in mind.

But, because brevity is a virtue, a single letter
*in combination with a modifier or two*
('``hMin``' as the value of minimum human wealth, say)
is fine – so long as the reader has some reason to expect
that the lower-case letter ``h`` signifies human wealth
(as they will, if they consult :ref:`Table 1 <table:RomanLetters>`).

That is the spirit in which we offer preferred interpretations
for the Roman letters below.
The upper case version is an aggregated version of the variable
(at the level of the whole economy, say,
or of the whole market being studied),
while the lower case indicates the level of
an individual consumer or firm or other subaggregate entity.

A few exceptions to these rules are explicitly noted below the table.

When an alternative is needed with a meaning similar to,
but distinct from, the definitions below,
please use a multi-letter name to represent it.
For example, please do not use :math:`W` for wealth
(if some measure of wealth that differs from
:math:`A`, :math:`B`, :math:`\pmb{H}`, or :math:`N` is needed);
instead use, say, ``Wlth`` or ``Wealth``.
(Some examples follow in a subsequent section).

Finally, a few of the definitions below are actually prohibitions;
these are based on many years of experience
which have shown that use of the prohibited variable name
generates more confusion than clarity.

.. table:: **Table 1** Preferred Usages of Roman Letters
   :name: table:RomanLetters

   ===============   =========================================================
    Letter            Meaning
   ===============   =========================================================
   :math:`A`         *A*\ ssets *A*\ fter *A*\ ll *A*\ ctions
                     *A*\ re *A*\ ccomplished (end of period)
   :math:`B`         *B*\ eginning *B*\ ank *B*\ alances
                     *B*\ efore any *B*\ ehavior (*b*\ eginning-of-period)
   :math:`C`         *C*\ onsumption *C*\ hoice *C*\ onnects
                     :math:`B` to :math:`A`
   :math:`D`         *D*\ ebt
   :math:`E`         PROHIBITED: Too many possible meanings
                     (expectations, effort, expenses)
   :math:`F`         Production *F*\ unction
   :math:`G`         *G*\ rowth
   :math:`\pmb{H}`   *H*\ uman wealth
   :math:`I`         *I*\ nvestment
   :math:`J`         Ad\ *J*\ ustment costs (e.g., in a :math:`Q` model)
   :math:`K`         Capital or beginning of period nonhuman assets
   :math:`L`         PROHIBITED: Is it Labor or Leisure or Land or ...?
   :math:`M`         *M*\ arket resources
                     (the sum of capital, capital income, and labor income)
   :math:`N`         *N*\ et wealth including human wealth (:math:`=B + H`)
   :math:`O`         PROHIBITED: Too similar to the number 0;
                     too many possible meanings
   :math:`P`         PROHIBITED: Is it prices, permanent income,
                     present value, profits, ...?
   :math:`Q`         Hayashi/Abel :math:`Q` (or similar asset price)
   :math:`R`         *R*\ eturn (see the variants articulated below)
   :math:`S`         PROHIBITED: "saving" (flow)? "savings" (stock)?
                     or the "saving rate" (ratio)?
   :math:`T`         This is a tough one. See the discussion below.
   :math:`U`         *U*\ tility
   :math:`V`         *V*\ alue
   :math:`W`         *W*\ age
   :math:`X`         e\ *X*\ penditures
                     (as distinct from consumption; e.g., for durables)
   :math:`Y`         Noncapital income
                     (usually, the sum of transfer and labor income)
   :math:`Z`         Lei\ *Z*\ ure in consumption/leisure tradeoff
   ===============   =========================================================

Exceptions to the Rules
-----------------------

The letter :math:`T` is an exception to the rule that lower- and upper-case
versions of variables are individual and aggregate quantities.
We reserve the capital letter to designate the end of the horizon
(death, or the end of the economy, occurs at the end of period :math:`T`).
The lower case version :math:`t` is so ubiquitiously used
as the current time period that we follow that convention here.

Finally, the following are exempted from the prohibition on
single-letter variable names because they are used so frequently
that the prohibition would be more trouble than it is worth:
:math:`a`, :math:`b`, :math:`c`, :math:`m`.

Standard Variable Names
-----------------------

There are more objects that are likely to be used extensively
in ARK projects than there are Roman letters.
We present preferred usages for some of those commonly-needed variables here.

.. table:: **Table 2** String Variables
   :name: table:Standard-Variable-Names

   =========  ===============================================================
    Name          Description
   =========  ===============================================================
   ``CND``    Consumption of Nondurable Good
   ``CNS``    Consumption of Nondurables and Services
   ``Cst``    Cost of something
   ``Dgd``    Stock of durable good
   ``Dvd``    Dividends
   ``Hse``    Quantity of housing
              (*not* value, which is quantity :math:`\times` price)
   ``Inc``    Income
   ``Nlq``    Quantity that is **N**\ ot **l**\ i\ **q**\ uid
              (e.g., ``aNlq`` is illiquid financial)
   ``Lbr``    Quantity of labor (hours)
   ``Pop``    Size of population
   ``Sav``    Saving (=Income Minus Consumption)
   ``Tax``    Tax – should be modified by ``Rte`` or ``Amt``
              articulated below
   ``Perm``   Permanent (labor; or at least, noncapital) income
   ``Tran``   Transitory (labor; or at least, noncapital) income
   =========  ===============================================================

Factors and Rates
=================

When measuring change over time, lower-case variables reflect rates
while the corresponding upper-case variable connects adjacent discrete
periods. [1]_\ :sup:`,`\ [2]_ So, for example, if the time interval
is a year and the annual interest rate is
:math:`{\mathsf{r}}=0.03` or three percent, then the annual
interest factor is :math:`{\mathsf{R}}=1.03`. [3]_

.. table:: **Table 3** Factors and Rates
   :name: table:Factors

   ==========  ===============================  =============================
    Code        Output                           Description
   ==========  ===============================  =============================
   ``Rfree``   :math:`\mathsf{R}`               Riskfree interest factor
   ``rfree``   :math:`\mathsf{r}`               Riskfree interest rate
   ``Risky``   :math:`\mathbf{R}`               The return factor on
                                                a risky asset
   ``risky``   :math:`\mathbf{r}`               The return rate on
                                                a risky asset
   ``Rport``   :math:`\mathbb{R}`            The return factor on
                                                the entire portfolio
   ``rport``   :math:`\mathbb{r}`            The return rate on
                                                the entire portfolio
   ``RSave``   :math:`\underline{\mathsf{R}}`   Return factor earned on
                                                positive end-of-period assets
   ``rsave``   :math:`\underline{\mathsf{r}}`   Return rate earned on
                                                positive end-of-period assets
   ``RBoro``   :math:`\bar{\mathsf{R}}`         Return factor paid on debts
   ``rboro``   :math:`\bar{\mathsf{r}}`         Return rate paid on debts
   ==========  ===============================  =============================


We depart from the upper-lower case scheme when the natural letter to
use has an even more urgent use elsewhere in our scheme.
A particularly common example occurs in the case of models like
Blanchard (1985) [#blanchardFinite]_ in which individual agents
are subject to a Poisson probability of death.
Because death was common in the middle ages,
we use the archaic Gothic font for the death rate;
and the probability of survival is the cancellation
of the probability of death:

.. table:: **Table 4** Special Cases: Factors and Rates
   :name: table:SpecialFactors

   ===========  ============================  ========================
    Code         LaTeX                         Description
   ===========  ============================  ========================
   ``DiePrb``   :math:`\mathsf{D}`            Probabilty of death
   ``LivPrb``   :math:`\cancel{\mathsf{D}}`   Probability to not die
                                              :math:`=(1-\mathsf{D})`
   ===========  ============================  ========================

Parameters
==========

Some parameters are worth defining because they
are likely to be used in a high proportion of models;
others are subject to enough constraints when used
(such as the need for similar-looking
upper- and lower-case Greek representations),
as to be worth standardizing.

Programmers should use
the corresponding variable name without the backslash
as the name of the corresponding object in their code.
For example, the Coefficient of Relative Risk Aversion is
``\CRRA`` in a LaTeX document and ``CRRA`` in a software module.

.. list-table:: **Table 5** Parameters
   :name: table:Parameters
   :header-rows: 1

   * - Name           
     - LaTeX          
     - Description    
     - Illustration   

   * - ``\CARA``    
     - :math:`\alpha`
     - Coefficient of Absolute Risk Aversion
     - :math:`\mathrm{u}(\bullet)=-\alpha^{-1}e^{-\alpha\bullet}`

   * - ``\CRRA``    
     - :math:`\rho`
     - Coefficient of Relative Risk Aversion
     - :math:`\mathrm{u}(\bullet)=(1-\rho)^{-1}\bullet^{1-\rho}`

   * - ``\DiscFac``
     - :math:`\beta`
     - Time Discount Factor
     - :math:`\mathrm{u}^\prime(c_t)=\mathsf{R}\beta\mathrm{u}^\prime(c_{t+1})`

   * - ``\discRte``
     - :math:`\vartheta`
     - Time Discount rate
     - :math:`\vartheta=\beta^{-1}-1`

   * - ``\DeprFac``
     - :math:`\daleth`
     - Depreciation Factor (Hebrew ``daleth``)
     - :math:`{K}_{t+1}=\daleth{K}_t+I_t`

   * - ``\deprRte``
     - :math:`\delta`
     - Depreciation Rate
     - :math:`\daleth=1-\delta`

   * - ``\TranShkAgg``
     - :math:`\Theta`
     - Transitory shock (aggregate)
     - :math:`\mathbb{E}_t[\Theta_{t+n}]=1` if :math:`\Theta` iid

   * - ``\tranShkInd``
     - :math:`\theta`
     - Transitory shock (individual)
     - :math:`\mathbb{E}_t[\theta_{t+n}]=1` if :math:`\theta` iid

   * - ``\PermShkAgg``
     - :math:`\Psi`
     - Permanent shock (aggregate)
     - :math:`\mathbb{E}_t[\Psi_{t+n}]=1` if :math:`\Psi` iid

   * - ``\permShkInd``
     - :math:`\psi`
     - Permanent shock (individual)
     - :math:`\mathbb{E}_t[\psi_{t+n}]=1` if :math:`\psi` iid

   * - ``\PopGro``
     - :math:`\Xi`
     - Population Growth Factor
     - :math:`\mathtt{Pop}_{t+1}=\Xi\mathtt{Pop}_t`

   * - ``\popGro``
     - :math:`\xi`
     - Population Growth rate
     - :math:`\Xi=1+\xi`

   * - ``\PtyGro``
     - :math:`\Phi`
     - Productivity Growth Factor
     - :math:`G=\Phi\Xi`

   * - ``\ptyGro``
     - :math:`\phi`
     - Productivity Growth rate
     - :math:`\Phi=(1+\phi)`

   * - ``\leiShare``
     - :math:`\zeta`
     - Leisure share, Cobb-Douglas utility
     - :math:`\mathrm{u}(c,z)=(1-\rho)^{-1}(c^{1-\zeta}z^\zeta)^{1-\rho}`

   * - ``\MPC``
     - :math:`\kappa`
     - Marginal Propensity to Consume
     - :math:`\mathrm{c}^\prime(m)=\partial c/\partial m`

   * - ``\Pat``
     - :math:`\text{\pmb{\Thorn}}`
     - Absolute Patience Factor (``Thorn``)
     - :math:`\text{\pmb{\Thorn}}=(\mathsf{R}\beta)^{1/\rho}`

   * - ``\PatPGro``
     - :math:`\text{\pmb{\Thorn}}_\Gamma`
     - Growth Patience Factor (``Thorn``)
     - :math:`\text{\pmb{\Thorn}}=(\mathsf{R}\beta)^{1/\rho}/\Phi`

   * - ``\PatR``
     - :math:`\text{\pmb{\Thorn}}_\mathsf{R}`
     - Return Patience Factor (``Thorn``)
     - :math:`\text{\pmb{\Thorn}}=(\mathsf{R}\beta)^{1/\rho}/\mathsf{R}`

   * - ``\pat``
     - :math:`\text{\thorn}`
     - Absolute Patience rate (``thorn``)
     - :math:`\text{\thorn}=(\mathsf{R}\beta)^{1/\rho}-1 \approx \rho^{-1}(\mathsf{r}-\vartheta)`

   * - ``\patpGro``
     - :math:`\text{\thorn}_\gamma`
     - Growth Patience rate (``thorn``)
     - :math:`\text{\thorn}_\gamma=\text{\thorn}-\phi`

   * - ``\patr``
     - :math:`\text{\thorn}_\mathsf{r}`
     - Return Patience rate (``thorn``)
     - :math:`\text{\thorn}_\mathsf{r}=\text{\thorn}-\mathsf{r}`

   * - ``\riskyshare``
     - :math:`\varsigma`
     - Portfolio share in risky assets
     - :math:`\mathbb{R}_{t+1}=(1-\varsigma)\mathsf{R}+\varsigma\mathbf{R}_{t+1}`

Mnemonics:

*  Hebrew ``daleth`` is the fourth letter of the Hebrew alphabet
   (as d and :math:`\delta` are of the Roman and Greek)
   and is an etymological and linguistic cousin of those letters

*  :math:`\vartheta` is the lower case Greek letter ``omega``,
   because people say "OMG, I've got to think about the future."

*  You are invited to scrutinize :math:`\Xi` yourself to imagine reasons
   it could represent something to do with population growth.

*  The glorious letter :math:`\text{\pmb{\Thorn}}`
   (pronounced '`thorn <http://en.wikipedia.org/wiki/Thorn_(letter)>`__')
   enriched Old English, Gothic, and some other defunct alphabets;
   sadly, it remains in use today only in Iceland.
   It is useful because having to type the many symbols in the object
   :math:`(\mathsf{R}\beta)^{1/\rho}`
   over and over again is a *thorn* in the side of economists
   working with dynamic models!
   (It is the 'absolute patience factor' because if it is less than one
   the consumer wants to bring resources from the future to the present
   and is therefore absolutely impatient;
   for a fuller discussion of this terminology, see Carroll 2016
   [#carrollTractable]_.)

Operators
=========

A few operators are so universally used that it will be useful to define them.

.. list-table:: **Table 6** Operators
   :name: table:Operators

   * - Name     
     - LaTeX         
     - Code     
     - Description   
     - Illustration
   * - ``\Ex``  
     - :math:`\mathbb{E}`
     - ``Ex_``  
     - The expectation as of date :math:`t`          
     - :math:`\mathbb{E}_t[\mathrm{u}^\prime(c_{t+1})]`
   * - ``\PDV`` 
     - :math:`\mathbb{P}`
     - ``PDV_`` 
     - Present Discounted Value       
     - :math:`\mathbb{P}_t^T(y)` is human wealth

Modifiers
=========

.. table:: **Table 7** General Purpose Modifiers
   :name: table:General

   =====================  ====================================================
   *[object]*\ ``P``      "Prime" means derivative, e.g. ``vPP``
                          is the second derivative of value:
                          :math:`\mathrm{v}^{\prime\prime}`
   *[object]*\ ``Agg``    Value of something at the aggregate level
                          (as opposed to ``Ind``)
   *[object]*\ ``Ind``    Value of something at the level of an individual
                          (as opposed to ``Agg``)
   *[object]*\ ``Lvl``    Level                         
   *[object]*\ ``Rto``    Ratio                         
   *[object]*\ ``Bot``    Lower value in some range     
   *[object]*\ ``Top``    Upper value in some range     
   *[object]*\ ``Min``    Minimum possible value        
   *[object]*\ ``Max``    Maximum possible value        
   *[object]*\ ``Cnt``    Continuous-time value         
   *[object]*\ ``Dsc``    Discrete-time value           
   *[object]*\ ``Shk``    Shock                         
   *[object]*\ ``StE``    Steady-state Equilibrium value of a variable           
   *[object]*\ ``Trg``    The 'target' value of a variable                      
   *[object]*\ ``Rte``    A 'rate' variable like the discount rate
                          :math:`\vartheta`
   *[object]*\ ``Fac``    A factor variable like the discount factor
                          :math:`\beta`
   *[object]*\ ``Amt``    An amount, like ``TaxAmt`` which might be lump-sum       
   *[object]*\ ``Nrm``    A normalized quantity; ex:
                          ``RNrm``\ =\ :math:`\mathsf{R}/\Gamma`
   *[object]*\ ``Abve``   Range of points ABOvE some boundary                      
   *[object]*\ ``Belw``   Range of points BELoW some boundary                      
   *[object]*\ ``Grid``   Points to be used as a grid for interpolations
   *[object]*\ ``Xtra``   An "extra" set of points to be
                          added to some existing set
   =====================  ====================================================

Shocks will generally be represented by finite vectors of outcomes and
their probabilities. For example, permanent income is called ``Perm``
and shocks are designated ``PermShk``

.. table:: **Table 8** Probabilities
   :name: table:Probabilities

   =====================  ====================================================
   *[object]*\ ``Dstn``   Representation of a probability distribution
                          (includes both Prbs and Vals)
   *[object]*\ ``Prbs``   Probabilities of outcomes
                          (e.g. ``PermShkPrbs`` for permanent shocks)
   *[object]*\ ``Vals``   Values (e.g., for mean one shock 
                          ``PermShkVals`` . ``PermShkPrbs`` = 1)
   =====================  ====================================================

Timing can be confusing because there can be multiple ordered steps
within a 'period.' We will use ``Prev``, ``Curr``, ``Next`` to refer to
steps relative to the local moment within a period, and :math:`t`
variables to refer to succeeding periods:

.. table:: **Table 9** Timing
   :name: table:Timing

   =================  ========================================================
   *[object]*\ tmn    object in period :math:`t` minus :math:`n`
   *[object]*\ tm1    object in period :math:`t` minus 1
   *[object]*\ Now    object in period :math:`t`
   *[object]*\ t      object in period :math:`t` (alternative definition)
   *[object]*\ tp1    object in :math:`t` plus 1
   *[object]*\ tpn    object in :math:`t` plus :math:`n`
   *[object]*\ Prev   object in previous subperiod
   *[object]*\ Curr   object in current subperiod
   *[object]*\ Next   object in next subperiod
   =================  ========================================================

Model Imports
=============

A convention in python is that when a tool is imported it is given
a convenient short name, e.g. ``import numpy as np``.

Here are the preferred shortnames for some of our models:

.. code:: python

   import ConsIndShockModel as cisMdl

Footnotes and Citations
=======================

.. [1]
   This convention rarely conflicts with the usage we endorse elsewhere
   of indicating individual-level variables by the lower and aggregate
   variables by the upper case.

.. [2]
   If there is a need for the continuous-time representation, we endorse
   use of the discrete-time rate defined below. Any author who needs a
   continuous-time rate, a discrete-time rate, and a discrete-time
   factor is invited to invent their own notation.

.. [3]
   In the rare cases where it is necessary to distinguish between a
   continuous-time rate and a discrete-time rate – for example, when
   there is an analytical result available in continuous time – the
   variable in question can be modified by ``Cnt`` or ``Dsc``.

.. [#blanchardFinite]
    Blanchard, Olivier J. 1985. "Debt, Deficits, and Finite Horizons."
    Journal of Political Economy 93 (2): 223–47.
    `doi:10.1086/261312 <https://doi.org/10.1086/261312>`__.

.. [#carrollTractable]
    Carroll, Christopher D. 2016. "Lecture Notes: A Tractable Model
    of Buffer Stock Saving." Johns Hopkins University.
    At https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption
    URL: http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/TractableBufferStock.pdf
