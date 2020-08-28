import time 
import string
import math
import random
import csv   
from functools import reduce
from openpyxl import load_workbook

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools

import selenium
from selenium import webdriver
from selenium.common.exceptions import ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import optimize
from scipy.stats import multivariate_normal

from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf
driver = 0.1
current_url = ''
std_devs = 3
dfs = pd.read_excel("./Output_map.xlsx") # file mapping output lever names to xpaths 
dfs_3 = pd.read_excel("./Input_map.xlsx") # file mapping input names to xpaths 
#for i in range(len(dfs)): # generate html lever addresses and put them in the dataframe#
#    dfs.iloc[i, 2] = '/html/body/table[1]/tbody/tr/td/table/tbody/tr[2]/td[1]/div[13]/div/table/tbody/tr[' + str(dfs.iloc[i, 1]).strip("%") + ']/td[5]/div/font'
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D']


def CMAES_iterate(iterations = 5, population_size = 30, 
            constraint = False, constraint_levers = [], constraint_values = [], 
            output_constraint = False, output_constraint_names = [], output_constraints = [],
            threshold = False,  threshold_names  = [], thresholds = [],   
            std_devs = [] ):
    """
    Given a set of constraints performs CMA-ES iteration(s) on the Global Calculator.
    
    **Args**:

    - iterations (*int*): Number of CMA-ES iterations. 
    - population_size (*int*): Number of chromosomes per iteration. 
    - constraint (*boolean*): Flag to decide whether to fix input levers or not. 
    - constraint_levers (*list of strings*): Contains the names of the levers to be fixed. 
    - constraint_Values (*list of floats*): Contains the values of the levers to be fixed. 
    - output_constraint (*boolean*): Flag to decide whether to fix outputs or not. 
    - output_constraint_names (*list of strings*): Contains the names of the output to be fixed. 
    - output_constraints (*list of floats*): Contains the values of the output to be fixed. 
    - threshold (*boolean*): Flag to decide whether to bound levers within a range or not. 
    - threshold_names (*list of strings*): Contains the names of the levers to be bounded within a range. 
    - thresholds (*list of list of floats*): Contains the upper and lower threshold to bound the specified levers. 
    
    **Returns**:
    Total fitness value of each generation and lever values of all the chromosomes from the last generation.
    """
    legends = []
    all_fitness = []; lever_values = []; results = []; output_values = []
    lever_names = list(dfs_3.iloc[:, 0].to_numpy()) # Create list with all lever names
    
    # Initialise population randomly
    for i in range(population_size): # Population size
        
        # Generate chromosome and read associated temperature, cost and other output values
        lever_values_temp, temperature_cost_temp = generate_chromosome(constraint = constraint, constraint_levers = constraint_levers, constraint_values = constraint_values, threshold = threshold, threshold_names  = threshold_names, thresholds = thresholds)  # Generate chromosome
        lever_values.append(lever_values_temp)
        results.append(temperature_cost_temp) # Calculate temperature and cost values associated to chromosome
        if output_constraint == True: # If output constraint set, read output
            output_values.append(read_outputs())      
        
    # Perform iterations of GA
    for j in range(iterations): # Fixed number of iterations (replace by tolerance later on)
        plt.figure(figsize = (16, 9))
        if plot == True:
            c = next(palette)
            # Plotting routine
            for pair in results:
                plt.plot(pair[0], pair[1], '.',  color=c) # color='darkorange'
            plt.xlim(0, 6000)
            plt.ylim(-12, 2)
            plt.xlabel("Temperature values", fontsize = 15)
            plt.ylabel("Cost values", fontsize = 15)
            plt.title("Population evolution", fontsize = 20)
            legends.append("Generation " + str(j))
            #plt.show()
        
        # Evaluate fitness
        fitness_values = []
        for i in range(len(results)):
            fitness_values.append(fitness(results[i], output_constraint = output_constraint, current_output_values = output_values[i], output_constraint_names =  output_constraint_names, output_constraints = output_constraints, std_devs = std_devs)) 
        all_fitness.append(sum(fitness_values)) # Find sum of fitness
        
        # Find fittest candidates <=> Parents  
        fittest_index = sorted(range(len(fitness_values)), key = lambda sub: fitness_values[sub])[:2] # Find the fittest 2
        parent_1 = lever_values[fittest_index [0]] # Find lever combination of parent 1
        parent_2 = lever_values[fittest_index [1]] # Lever combination of aprent 2
        
        # Printing routine
        print("Generation: ", j+1, "; Fitness is: ", sum(fitness_values))
        print("Temperature and cost values: ", results[fittest_index[0]], "; ", results[fittest_index[1]])
        print("Parents:")
        print(parent_1)
        print(parent_2, "\n")
        
        # Crossover and mutation
        for i in range(len(lever_values)): # Perform crossover by mating parents using uniform crossover (high mutation prob)
            
            # If some inputs are bounded within thresholds, take into account when mating
            if lever_names[i] in threshold_names:
                th = thresholds[threshold_names.index(lever_names[i])] # Temporary variable containing current threshold
                lever_values[i] = mate(parent_1, parent_2, threshold = True, threshold_value = th) # Generates full new set of lever combination
            
            # Otherwise mate right away
            else:
                lever_values[i] = mate(parent_1, parent_2) # Generates full new set of lever combinations
        
        results = []; output_values = []
        
        # Calculate temperature and cost of each lever combination and overwrite lever values according to constraints
        for lever_combination in lever_values: # For each chromosome
            lever_combination_temp = lever_combination # Store it in a temp variable
            # Overwrite lever values with constraints. If current lever is not constrained, it does nothing 
            lever_names, lever_combination = overwrite_lever_values(lever_names, lever_combination, constraint_levers, constraint_values)
            lever_values[lever_values.index(lever_combination_temp)] = lever_combination # Set current lever values after constraint
            # Read temperature and cost values for given lever combination (w or w/o constraint)
            results.append(move_lever(lever_names, lever_combination, costs = True, constraint = constraint,  constraint_levers = constraint_levers, constraint_values = constraint_values))
            # Read outher output values for current lever combination
            if output_constraint == True:
                output_values.append(read_outputs())
    plt.legend(legends)
    plt.show()
    return all_fitness, lever_values

def lever_step(lever_value, thresholds = [1, 3.9], threshold = False, threshold_name = "", threshold_value = "", p = [0.5, 0.5, 0, 0, 0, 0]):
    """Mutate gene by randomly moving a lever up or down by 0.1. Returns the mutated gene (the new lever value)"""
    move = -0.
    prob = random.randint(0, 100)/100 # Generate random gene
    if prob < p[0]: 
        #print("Lever down 1")
        move = -0.1 # Move lever down
    elif prob < p[0] + p[1]: 
        #print("Lever up 1")
        move = 0.1 # Move lever up
    elif prob < p[0] + p[1] + p[2]: 
        #print("Lever down 2")
        move = -0.2 # Move lever down
    elif prob < p[0] + p[1] + p[2] + p[3]: 
        #print("Lever up 2")
        move = 0.2 # Move lever up
    elif prob < p[0] + p[1] + p[2] + p[3] + p[4]: 
        #print("Lever down 3")
        move = -0.3 # Move lever down
    elif prob < p[0] + p[1] + p[2] + p[3] + p[4] + p[5]: 
        move = 0.3 # Move lever up
        #print("Lever up 3")
    # If the lever value is out of bounds, reverse direction of step (taking specified threshold into account)
    if threshold == True:
        if (lever_value + move < threshold_value[0]) or (lever_value + move > threshold_value[1]):
            move = -move
    else:
        if (lever_value + move < thresholds[0]) or (lever_value + move > thresholds[1]):
            move = -move
    return round(lever_value + move, 3)

def generate_chromosome(constraint = False, constraint_levers = [], constraint_values = [], 
                        threshold = False, threshold_names = [], thresholds = []):
    """
    Initialises a chromosome and returns its corresponding lever values, and temperature and cost. 

    **Args**:

    - constraint (*boolean*): Flag to select whether any inputs have been fixed. 
    - constraint_levers (*list of strings*): Contains the name of levers to be fixed.  
    - constraint_values (*list of floats*): Contains the values to fix the selected levers to. 
    - threshold (*boolean*): Flag to select whether any inputs have to be bounded within a range. 
    - threshold_names (*list of strings*): Contains the name of the levers to be bounded within a range. 
    - thresholds (*list of lists of floats*): Contains the upper and lower bound for each specified lever. 

    **Returns**:
    Lever values corresponding to generated chromosome and cost values corresponding to the current chromosome. 
    """
    lever_names = list(dfs_3.iloc[:, 0].to_numpy()) # Create list with all lever names
    # Generate random lever combination
    random_lever_values = new_lever_combination(threshold = threshold, threshold_names = threshold_names, thresholds = thresholds) 
    # Fix specified input levers
    if constraint == True:
        lever_names, random_lever_values = overwrite_lever_values(lever_names, random_lever_values, constraint_levers,  constraint_values)
    result = move_lever(lever_names, random_lever_values, costs = True, constraint = constraint, constraint_levers = constraint_levers, constraint_values = constraint_values) # Move lever accordingly and read temperature and cost valuesw
    return random_lever_values, result

def mate(parent_1, parent_2, threshold = False, threshold_name = "", threshold_value = ""): 
        ''' Takes a couple of parents, performs crossover, and returns resulting child. '''
        child_chromosome = [] 
        for p1, p2 in zip(parent_1, parent_2):     
            prob = random.random()  # Generate random value
            if prob < 0.4: # Select gene from 1st parent
                child_chromosome.append(p1) 
            elif prob < 0.8: # Select gene from 2nd parent
                child_chromosome.append(p2) 
            elif prob < 0.9: 
                child_chromosome.append(mutated_genes(p1, threshold = threshold, threshold_name = threshold_name, threshold_value = threshold_value))  # Mutate gene from 1st parent
            else:
                child_chromosome.append(mutated_genes(p2, threshold = threshold, threshold_name = threshold_name, threshold_value = threshold_value)) # Mutate gene from 2nd parent
        return child_chromosome

