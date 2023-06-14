# Aiyagari Model - Deep Learning based Algorithm

1. We solve Aiyagri model with Poisson income process with a neural network.  (It takes around 4 minutes to finish training)

For results only, run the following commands
1. Set current directory as working directory with 
* cd ./examples/Aiyagari-Deep/
2. Install necessary packages with 
* pip install -r ./binder/requirement.txt
3. Execute training process and generate the graphs with 
* python Aiyagari-Deep_Poisson.py


Or simply execute 

* bash ./reproduce.sh


2. The following graph contains main files and shows how this folder is structured.

   ```mermaid
   graph LR;
      Parent-->Aiyagari-Deep_Poisson.ipynb
      Parent-->Tex
      Parent-->Aiyagari-Deep_Poisson.py
      Parent-->README.md
      Parent-->reproduce.sh
     
      Tex--> Figures
   
   ```