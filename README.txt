Heterogeneous Agents Resources and toolKit (HARK)
alpha release - December 14, 2015

Table of Contents:

I.   Introduction
II.  Quick start guide
III. List of archived files
IV.  Warnings and disclaimers

-------------------------------------------------------------------

I. INTRODUCTION

Welcome to HARK!  We are tremendously excited you're here.  HARK is
very much a work in progress, but we hope you find it valuable.  We 
*really* hope you find it so valuable that you decide to contribute
to it yourself.  This document will tell you how to get HARK up and
running on your machine, and what you will find in HARK once you do.

If you have any comments on the code or documentation, we'd love to
hear from you!  Our email addresses are:

Chris Carroll: ccarroll@llorracc.org
Matthew White: mnwhite@gmail.com
Nathan Palmer: Nathan.Palmer@ofr.treasury.gov
David Low: David.Low@cfpb.gov

-------------------------------------------------------------------

II. QUICK START GUIDE

This is going to be easy, friend.  HARK is written in Python, specifically the
Anaconda distribution of Python.  Follow these easy steps to get HARK going:

1) Go to https://www.continuum.io/downloads and download Anaconda for your
operating system; be sure to get the version for Python 2.7

2) Install Anaconda, using the instructions provided on that page.

3) Copy the entire contents of alphaHARKive.zip into an empty directory.
Maybe call that directory /HARK ?  The choice is yours.

4) Open Spyder, an interactive development environment (IDE) for Python
(specifically, iPython).  On Windows, open a command prompt and type "spyder".
On Linux, open the command line and type "spyder".  On Mac, open the command
line and type "spyder".

5) Navigate to the directory where you put the HARK files.  This can be done
within Spyder by doing "import os" and then using os.chdir() to change directories.  
chdir works just like cd at a command prompt on most operating systems, except that
it takes a string as input: os.chdir('Music') moves to the Music subdirectory
of the current working directory.

6) Test one of HARK's sample modules with one of the following commands:

"run ConsumerExamples"
"run SolvingMicroDSOPs"
"run cstwMPC"
"run TBSexamples"

7) The Python environment can be cleared or reset with ctrl+.  Note that
this will also change the current working directory back to its default.
To change the default directory (the "global working directory"), see
Tools-->Preferences-->Global working directory; you might need to restart
Spyder for the change to take effect.

8) Read the more complete documentation in HARKdoc.pdf.

-------------------------------------------------------------------

III. LIST OF ARCHIVED FILES

Documentation:
- README.txt                   The file you are currently reading.
- HARKdoc.pdf                  Somewhat more complete description of the HARK project

Main HARK Python modules:
- HARKcore.py                  The microeconomic agent framework
- HARKutilities.py             Assorted useful tools
- HARKestimation.py            Model estimation methods, etc
- HARKsimulation.py            Tools for generating simulated data
- HARKinterpolation.py         Methods for representing interpolated functions

Consumption-saving model files:
- ConsumptionSavingModel.py    ConsumerType and related solvers
- SetupConsumerParameters.py   Load in a baseline specification for SolvingMicroDSOPs
- ConsumerExamples.py          Demonstrate a few simple consumption-saving models
- SolvingMicroDSOPs.py         Estimate a very simple lifecycle model
- SetupSCFdata.py              Imports SCF data for use by SolvingMicroDSOPs
- SCFdata.scv                  SCF data for SolvingMicroDSOPs
- SMMcontour.png               Contour plot of objective function for SolvingMicroDSOPs

CSTWmpc files:
- cstwMPC.py                   The CSTWmpc model and estimation
- SetupParamsCSTW.py           Loads in model parameters for CSTW, chooses specification
- MakeCSTWfigs.py              Makes figures for CSTWmpc paper (needs results files)
- MakeCSTWfigsForSlides.py     Makes figures for CSTWmpc slides (doesn't need results)
- SCFwealthDataReduced.txt     SCF data with just net worth and data weights
- USactuarial.txt              U.S. mortality data from the Social Security Admin
- EducMortAdj.txt              Education-age mortality adjusters

Tractable buffer stock files:
- TractableBufferStock.py      TractableConsumerType and its solver
- TBSexamples.py               An example of the TBS model in action

Empty directories:
- ./Figures                    Holds figures produced by CSTWmpc
- ./Results                    Holds results files produced by CSTWmpc

-------------------------------------------------------------------

IV. WARNINGS AND DISCLAIMERS

This is an early "alpha" version of HARK.  The code has not been
extensively tested as it should be.  We hope it is useful, but 
there are absolutely no guarantees (expressed or implied) that 
it works or will do what you want.  Use at your own risk.  And 
please, let us know if you find bugs!

-------------------------------------------------------------------
