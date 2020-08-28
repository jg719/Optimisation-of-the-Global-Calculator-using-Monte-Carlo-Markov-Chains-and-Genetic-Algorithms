Optimisation of the Global Calculator using Monte Carlo Markov Chains and Genetic Algorithms 
============================

**Independent Research Project for MSc Applied Computational Science and Engineering <br>
Jorge Garcia (GitHub: acse-jg719, CID: 01300431, email: jg719@ic.ac.uk)** <br>

This repository contains the code that optimises the Global Calculator using Monte Carlo Markov Chains and Genetic Algorithms. <br>

Documentation and user guide 
----------------------------
[Click here to access a detailed documentation of all the methods used, two usage examples and the code’s license and installation requirements.](https://jg719.github.io/IRP-acse-jg719-documentation/) <br>

The code contained in this repository has been tested inside their respective Jupyter notebook, as mentioned in the final report.  <br>

Repository structure
--------------------
There are **four Jupyter notebooks** in this repository. The two key ones for this investigation are "*MCMC_Analysis.ipynb*" and "*GA_Analysis.ipynb*". The other two ("*CMAES_Analysis.ipynb*" and "*ANN_Analysis.ipynb*") correspond to failed approaches but are still included for reference. <br>

These four notebooks depend on **XLSX files** contained in "*excel_files*". <br>

**Total's final sustainable pathway** and its corresponding 2050 forecast are contained in "*Total's_optimal_pathway.xlsm*". <br>

The folder "*docs*" contains all the **documentation files** generated using Sphinx. <br>

Lastly, the **requirements and license files** are included as "*requirements.txt*" and "*license.txt*".

Summary
-------
The Global Calculator is a model used to forecast the world's energy, food and land systems to 2050. <br>

The aim of this investigation is to perform a constrained multiobjective optimisation of its input parameter space to yield alternative pathways to sustainability. <br>

This is achieved using Monte Carlo Markov Chains and Genetic Algorithms.  <br>

**Genetic algorithms:**


The optimiser takes user-specified input and output constraints and finds lever combinations that satisfy them whilst guaranteeing a minimum climate impact and maximum economic output. <br>

The figure below shows how the generations evolve towards an optimal solution. <br>
<p align="center">
 <img src="https://github.com/acse-2019/irp-acse-jg719/blob/master/Code/excel_files/GA_evolution.gif">
</p>


**Monte Carlo Markov Chains:**

This method is used to derive probability distributions of the Global Calculator's inputs that maximise the likelihood of meeting user-specified constraints - In this case, climate impact minimisation and economic output maximisation. <br>

The figure below shows such distributions. <br> 

![](https://github.com/acse-2019/irp-acse-jg719/blob/master/Code/excel_files/input_levers.png)


License
-------
MIT <br>

