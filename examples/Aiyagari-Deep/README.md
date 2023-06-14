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

   Where:
   
   * "Aiyagari-Deep_Poisson.ipynb" is the Jupyter Notebook file, for key features of the paper and step-by-step python codes implementing to solve the HJB equation;
   * "Tex" is a folder where the .tex file is located. It includes figures.
    * "Aiyagari-Deep_Poisson.py"  is the file which contains all codes for solving the model, generating figures.
* "reproduce.sh" is a bash shell script which calls "Aiyagari-Deep_Poisson.py" to run. In other words, it runs everything in Linux System. 
   
   