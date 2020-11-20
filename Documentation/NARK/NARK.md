Suggested Variable Naming Conventions in the Econ-ARK Toolkit
-------------------------------------------------------------

November 20, 2020

 

  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Abstract  
Sharing of code is easier when diﬀerent contributors use similar names
for similar objects. While we will not enforce the recommendations
below, contributors can make their code more attractive to others by
using names consistent with our guidelines.   

            Keywords   
ARK, Variable, Function, Object, Operator, Naming, Guidelines,
Conventions

            JEL codes   
None

html version at
[https://econ-ark.github.io/HARK/Documentation/NARK](https://econ-ark.github.io/HARK/Documentation/NARK)

### 1  Principles

Our aim has been to balance:

-   Brevity
-   Mnemonic quality
-   Uniqueness (enabling global search-and-replace)
-   Ubiquity (deﬁnitions useful for many projects)
-   Combinatoriality (which encourages mashups)

### 2  Variables

#### 2.1  Single-Letter

We strongly discourage the use of single-letter variable names. Reasons
are presented ad nauseum in introductory CS texts (which, we know, few
economists consult); here we will note only that the extent to which
your code will be inﬂuential depends upon the extent to which someone
else can easily read it, which is harder if you have used variable names
which could mean almost anything. (We have made careful choices for the
‘default’ meanings of each Roman letter (see below); hence, in the
context of the toolkit, use of a single-letter name for any meaning
other than the designated one will be doubly confusing).

It is only slightly less objectionable to name a variable after a
familiar letter in another commonly used alphabet (say, delta). Your
future self (and other users) will not know which of the many possible
meanings of ![δ ](NARK0x.svg) you had in mind.

But, because brevity is a virtue, a single letter in combination with a
modiﬁer or two (‘hMin’ as the value of minimum human wealth, say) is ﬁne
– so long as the reader has some reason to expect that the lower-case
letter h signiﬁes human wealth (as they will, if they consult
Table [1](#x1-30011)).

That is the spirit in which we oﬀer preferred interpretations for the
Roman letters below. The upper case version is an aggregated version of
the variable (at the level of the whole economy, say, or of the whole
market being studied), while the lower case indicates the level of an
individual consumer or ﬁrm or other subaggregate entity.

A few exceptions to these rules are explicitly noted below the table.

When an alternative is needed with a meaning similar to, but distinct
from, the deﬁnitions below, please use a multi-letter name to represent
it. For example, please do not use ![W ](NARK1x.svg) for wealth (if some
measure of wealth that diﬀers from ![A ](NARK2x.svg), ![B ](NARK3x.svg),
![HHH ](NARK4x.svg), or ![N ](NARK5x.svg) is needed); instead use, say,
Wlth or Wealth. (Some examples follow in a subsequent section).

Finally, a few of the deﬁnitions below are actually prohibitions; these
are based on many years of experience which have shown that use of the
prohibited variable name generates more confusion than clarity.

![Table 1:Preferred Usages of Roman Letters](NARK6x.svg)

#### 2.2  Exceptions to the Rules

The letter ![T ](NARK7x.svg) is an exception to the rule that lower- and
upper-case versions of variables are individual and aggregate
quantities. We reserve the capital letter to designate the end of the
horizon (death, or the end of the economy, occurs at the end of period
![T ](NARK8x.svg)). The lower case version ![t ](NARK9x.svg) is so
ubiquitiously used as the current time period that we do not want to
resist the overwhelming force of tradition to prohibit its use in that
capacity.

Finally, the following are exempted from the prohibition on
single-letter variable names because they are used so frequently that
the prohibition would be more trouble than it is worth: ![a
](NARK10x.svg), ![b ](NARK11x.svg), ![c ](NARK12x.svg), ![m
](NARK13x.svg).

#### 2.3  Strings

There are more objects that are likely to be used extensively in ARK
projects than there are Roman letters. We present preferred usages for
some of those commonly-needed variables here.

![Table 2:String Variables](NARK14x.svg)

### 3  Factors and Rates

When measuring change over time, lower-case variables reﬂect rates while
the corresponding upper-case variable connects adjacent discrete
periods.[¹](NARK2.html#fn1x0) ![, ](NARK15x.svg)[²](NARK3.html#fn2x0)
So, for example, if the time interval is a year and the annual interest
rate is ![r = 0.03 ](NARK16x.svg) or three percent, then the annual
interest factor is ![R = 1.03 ](NARK17x.svg).[³](NARK4.html#fn3x0)

![Table 3:Factors and Rates](NARK18x.svg)

We depart from the upper-lower case scheme when the natural letter to
use has an even more urgent use elsewhere in our scheme. A particularly
common example occurs in the case of models like
[Blanchard](#XblanchardFinite) ([1985](#XblanchardFinite)) in which
individual agents are subject to a Poisson probability of death. Because
death was common in the middle ages, we use the archaic Gothic font for
the death rate; and the probability of survival is the cancellation of
the probability of death:

![Table 4:Special Cases: Factors and Rates](NARK19x.svg)

### 4  Parameters

Some parameters are worth deﬁning because they are likely to be used in
a high proportion of models; others are subject to enough constraints
when used (such as the need for similar-looking upper- and lower-case
Greek representations), as to be worth standardizing.

Programmers should use the corresponding variable name without the
backslash as the name of the corresponding object in their code. For
example, the Coeﬃcient of Relative Risk Aversion is ![\\CRRA
](NARK20x.svg) in a LATE Xdocument and CRRA in a software module.

![Table 5:Parameters](NARK21x.svg)

Mnemonics:

-   Hebrew daleth is the fourth letter of the Hebrew alphabet (as d and
    ![δ ](NARK22x.svg) are of the Roman and Greek) and is an
    etymological and linguistic cousin of those letters
-   ![𝜗 ](NARK23x.svg) is the lower case Greek letter omega, because
    people say “OMG, I’ve got to think about the future.”
-   You are invited to scrutinize ![Ξ ](NARK24x.svg) yourself to imagine
    reasons it could represent something to do with population growth.
-   The glorious letter ![ÞÞÞ ](NARK25x.svg) (pronounced
    ‘[thorn](http://en.wikipedia.org/wiki/Thorn_(letter))’) enriched Old
    English, Gothic, and some other defunct alphabets; sadly, it remains
    in use today only in Iceland. It is useful because having to type
    the many symbols in the object ![(Rβ )1∕ρ ](NARK26x.svg) over and
    over again is a thorn in the side of economists working with dynamic
    models! (It is the ‘absolute patience factor’ because if it is less
    than one the consumer wants to bring resources from the future to
    the present and is therefore absolutely impatient; for a fuller
    discussion of this terminology, see
    [Carroll](#XcarrollTractable) ([2016](#XcarrollTractable)).)

### 5  Operators

A few operators are so universally used that it will be useful to deﬁne
them.

![Table 6:Operators](NARK27x.svg)

### 6  Modiﬁers

![Table 7:General Purpose Modiﬁers](NARK28x.svg)

Shocks will generally be represented by ﬁnite vectors of outcomes and
their probabilities. For example, permanent income is called Perm and
shocks are designated PermShk

![Table 8:Probabilities](NARK29x.svg)

Timing can be confusing because there can be multiple ordered steps
within a ‘period.’ We will use Prev, Curr, Next to refer to steps
relative to the local moment within a period, and ![t ](NARK30x.svg)
variables to refer to succeeding periods:

![Table 9:Timing](NARK31x.svg)

### 7  Model Imports

A convention in python is that when a tool is imported it is given a
convenient short name, e.g. import numpy as np.

Here are the preferred shortnames for some of our models:

import ConsIndShockModel as cisMdl

### References

   Blanchard, Olivier J. (1985): “Debt, Deﬁcits, and Finite Horizons,”
Journal of Political Economy, 93(2), 223–247.

   Carroll, Christopher D. (2016): “Lecture Notes: A Tractable Model of
Buﬀer Stock Saving,” Discussion paper, Johns Hopkins University, At
[http://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption](http://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption).
