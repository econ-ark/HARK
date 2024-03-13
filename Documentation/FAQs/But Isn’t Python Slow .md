# But Isn't Python Slow?
It may seem like a disadvantage to write a library for doing computationally difficult tasks in a language that's got a reputation for being much slower than its competators. Python is an uncompiled language which doesn't allow it to use many of the optimizations that come with compiled languages. Raw python will always be slower than alternative languages like Java, Julia, and C.

However raw Python is rarely used for programming. Python is powerful because of its libraries which give it much more flexibility and speed. Python's libraries are typically written in C and many run just as fast. For sections of code that need to be written in python, there are specialized libraries like Cython and Numba which can dramatically improve the proformance of Python code.

The study "[A comparison of programming languages in macroeconomics (2015)](https://www.sciencedirect.com/science/article/abs/pii/S0165188915000883)" found that raw python was much slower on both Windows and Mac. Using Numba, a Just In Time compiler, the code ran faster than Julia and Java on both mac and windows. In their [2018 update](https://www.sas.upenn.edu/~jesusfv/Update_March_23_2018.pdf), Python with Numba was roughly as fast as the original Julia implimentation and Cython was as fast as their newly optimized version.

<div align="center">
<img src=".\images\table 2015.png" height="300" />
</p><b align = "center">2015 language comparison [Aruoba, S.B., and J. Fern·ndez-Villaverde]</b></p>
<img src=".\images\Table.png" height="400" />
</p><b align = "center">2018 language comparison [Aruoba, S.B., and J. Fern·ndez-Villaverde]</b></p>
</div>


In a [Nasa comparison](https://software.nasa.gov/software/GSC-18111-1), Python is slower for certain tasks, like string manipulations and recursive functions, while its roughly as fast, as Julia when working with arrays, especially when using Numba. This specific analysis has a number of problems with its benchmarking. Several of the Python solutions can be speed up dramatically with a small amount of tweaking, and others have bugs in them which make them appear like they're faster than they are. Occasionally the solution method will be different across languages. In the 2021 analysis, the R method of calculating fibonacci numbers recursively is much faster because it uses a different method than the other languages. If all of them used it their results would be much more similar. A Numba method for calculating the fibonacci numbers recursively isn't provided, but applying it to their raw python method, speeds it up by roughly 50x. More issues with this comparison are discussed on the issues section of the github page.


<div align="center">
<img src="./images\nasa comparison 2021-1.png" height="300" />
</p><p align = "center">2021 language comparison [Jules Kouatchou and Alexander Medema]</p>
</div>


The paper "[Matlab, Python, Julia: What to Choose in Economics?](https://link.springer.com/article/10.1007/s10614-020-09983-3#Sec8)" compares Matlab, Python, and Julia, for two specific economic models; A stripped down version of the RBC model, and a more complicated New Keynsian model. They optimized their solutions for all three languages, and employeed Numba for python, though there may still be room for improvement. They find that Python and Matlab are slower than Julia for the Value Function ideration methods, and the Endogenous grid method used heavily by this project, because their implimentation wasn't vectorized. Python arrays run faster the larger they are because the overhead is proportionally smaller. When using a vectorized method that allows Matlab and Python to take advantage of this, they are comparably fast. In the comparison for their solution to the new keynsian model, all three were similarly fast with Python only being moderately slower for the largest approximation, though the authors didn't speculate why. During their multicore processing test, Python and Julia ran similarly fast.

<div align="center">
<img src=".\images\RBC table-1.png" height="300" />
</p><b align = "center">Consumption function solution comparison [Coleman, C., Lyon, S., Maliar, L. et al.]</b></p>
<img src=".\images\NK table-1.png" height="300" />
</p><b align = "center">New Keynsian model comparison [Coleman, C., Lyon, S., Maliar, L. et al.]</b></p>
</div>


Overall Python is generally slower that Julia, C, and Fortran, though not by much. Even though raw python is much slower, the strength of Python comes from its libraries, which run at the speed of the languages they're written in with a usually insignificant amount of overhead. In situations where Python needs to be speeded up, tools like Numba and Cython can both bring it up to par.