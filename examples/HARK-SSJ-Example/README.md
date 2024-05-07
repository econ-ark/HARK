# To run notebook

- conda env create -f binder\environment.yml

- cond activate KS-HARK-SSJ-Example

change directory to this repo

- jupyter lab

NOTE** 

KS-HARK-presentation-notebook.ipynb only works with the HARK folder in this directory.  It does NOT work with any version of HARK that can be pip installed. 

econ-ark == 0.13.0 is the closest version to the local HARK folder but 0.13.0 will produce bugs to the Heterogeneous agent jacobians. 

