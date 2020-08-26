Optimisation of the Global Calculator using Monte Carlo Markov Chain and Genetic Algorithms
===========================================================================================

Page structure
##############

This page contains documentation of all the methods used, two usage examples and the code's license and installation requirements. 

To install Jupiter notebooks, go to: https://jupyter.readthedocs.io/en/latest/install.html

To install any missing packages via conda, follow: https://docs.conda.io/projects/conda/en/latest/commands/install.html 

To open the notebooks, follow these instructions: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html   

Introduction
############

This project includes four self-contained Jupyter notebooks with detailed step by step guide. 

They are located in the root directory and can be executed from their first to their last cell. They are: 

- Markov Chain Monte Carlo analysis (*MCMC_Analysis.ipynb*) [**Successful**]
- Genetic Algorithms analysis (*GA_Analysis.ipynb*) [**Successful**]
- Covariance Matrix Adaption Evolutionary Strategy analysis (*CMAES_Analysis.ipynb*) [**Unsuccessful**]
- Artificial Neural Network analysis (*ANN_Analysis.ipynb*) [**Unsuccessful**]

Setting the constraints and running the optimiser
#################################################


To change the optimisation constraints, the cell "**Defining the optimisation constraints**" in *GA_Analysis.ipynb* must be edited accordingly, as shown below. 

        .. toctree::
           :maxdepth: 2

           Sample optimisation constraints <optimisation_constraints/constraints>

The constraint names are listed here: http://tool.globalcalculator.org

The input constraint values must be in the range [1, 4]. Each output constraint value has its own units and scale. 

*Note that the optimiser will minimise GHG emissions per capita and maximise economic viability by default.*

The 2nd notebook "*GA_Analysis.ipynb*" can be run to perform the optimisation of the Global Calculator. 

User documentation
##################
  
MCMC
****

    .. toctree::
       :maxdepth: 2

       MCMC_analysis

GA
**

    .. toctree::
       :maxdepth: 2

       GA_Analysis

CMA-ES
******

    .. toctree::
       :maxdepth: 2

       CMAES_analysis

ANN
******

    .. toctree::
       :maxdepth: 2

       ANN_analysis.rst

Usage examples
##############

Genetic algorithms optimisation
*******************************

    .. toctree::
       :maxdepth: 2

       Genetic algorithms <examples/GA/GA_Analysis>

Monte Carlo Markov Chain analysis
*******************************

    .. toctree::
       :maxdepth: 2

       Genetic algorithms <examples/MCMC/MCMC_Analysis>

Housekeeping
############

    .. toctree::
       :maxdepth: 2

       License <license/license>
       Requirements <license/requirements>


