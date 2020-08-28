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
def GA_iterate(iterations = 5, population_size = 30, 
            constraint = False, constraint_levers = [], constraint_values = [], 
            output_constraint = False, output_constraint_names = [], output_constraints = [],
            threshold = False,  threshold_names  = [], thresholds = [],   
            std_devs = std_devs ):
    """
    Given a set of constraints performs GA iteration(s) on the Global Calculator.

    **Args**:

    - iterations (*int*): Number of GA iterations. 
    - population_size (*int*): Number of chromosomes per iteration. 
    - constraint (*boolean*): Flag to decide whether to fix input levers or not. 
    - constraint_levers (*list*): Contains the names of the levers to be fixed. 
    - constraint_Values (*list*): Contains the values of the levers to be fixed. 
    - output_constraint (*boolean*): Flag to decide whether to fix outputs or not. 
    - output_constraint_names (*list*): Contains the names of the output to be fixed. 
    - output_constraints (*list*): Contains the values of the output to be fixed. 
    - threshold (*boolean*): Flag to decide whether to bound levers within a range or not. 
    - threshold_names (*list*): Contains the names of the levers to be bounded within a range. 
    - thresholds (*list*): Contains the upper and lower threshold to bound the specified levers. 
    
    **Returns**:
        
    Total fitness value of each generation and lever values of all the chromosomes from the last generation.
    """
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
        
        # Plotting routine
        plt.figure(figsize = (12, 6))
        for pair in results:
            plt.plot(pair[0], pair[1], '.',  color='darkorange')
        plt.xlim(0, 6000)
        plt.ylim(-12, 2)
        plt.xlabel("Temperature values")
        plt.ylabel("Cost values")
        plt.title("Current population")
        plt.show()
        
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
                
    return all_fitness, lever_values

    
def moving_average(a, n=3) :
    """Simple moving average filter"""
    ret = np.cumsum(a, dtype=float) # Cumulative sum of input values
    ret[n:] = ret[n:] - ret[:-n] # Differences given by window length n
    return ret[n - 1:] / n # Divide by window length

def move_lever(lever, value, costs = False,  address = current_url,  
               constraint = False, constraint_levers = [], constraint_values = []): 
    """

    Sets a lever to a given value. Reads corresponding temperature and, if selected, cost values. 
    
    **Args**:

    - lever (*list*): Contains the names of the levers to be moved. 
    - value (*list*): Contains the value of the levers to be moved - Automatically matched to lever names. 
    - costs (*boolean*): Flag to decide whether to read cost values or not. 
    - address (*string*): URL address corresponding to given lever combination. 
    - constraint (*boolean*): Flag to decide whether to set exact input constraints or not. 
    - constraint_levers (*list*): Contains the names of the levers to be fixed. 
    - constraint_values (*list): Contains the values to fix the specified levers.
    
    **Returns**:
    
    List of temperature and cost values for the new lever position
    """
    # Update URL address with input lever names and values, one at a time 
    for i in range(len(lever)):
        address = new_URL(lever[i], value[i], address = address)  
    
    # Overwrite for constraint
    if constraint == True: 
        address = overwrite(constraint_levers, constraint_values, address = address)
     
    # Open website corresponding to the input values
    driver.get(address) 
    
    ########################################## IMPORTANT ####################################################
    # All of the lines below are in charge of webscraping the temperature and, if selected, the cost values. 
    # The Global Calculator is a hard to webscrape website (sometimes, it results in bugs or uncoherent
    # temperature and cost values). The code below ensures that, no matter what, the values will be read. 
    # To do so it performs different actions based on the current state of the website and the output values. 
    #########################################################################################################
    
    time.sleep(0.2)
    id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
    id_box.click()
    time.sleep(1)
    
    # Read temperature values
    try:
        output = int(read_CO2()[:4]) # Read output CO2
    except: # Problem reading output CO2? The code below sorts it
        time.sleep(1)
        open_lever_menus() # Open lever menus
        move_lever([lever[0]],[1.3], costs = False) # Move lever to an arbitrary value
        driver.get(address) # Open website back 
        time.sleep(0.2)
        id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
        id_box.click()
        output = int(read_CO2()[:4]) # Read output CO2
        
    # Read cost values   
    if costs == True:   
        driver.find_element_by_xpath('//*[@id="mn-6"]').click() # Move to compare tab 
        time.sleep(0.2)
        userid_element = driver.find_element_by_xpath('//*[@id="container_costs_vs_counterfactual"]/div/div[11]') # Read GDP
        cost_output = userid_element.text
        try:
            cost_output = float(cost_output[:4].rstrip("%")) # Convert GDP from string to float
        except: # Problem converting GDP? The code below sorts it
            cost_output = float(cost_output[:3].rstrip("%"))
        
        # Reload the page and bypass start
        driver.refresh() # Refresh
        time.sleep(1)
        id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
        id_box.click()
        userid_element = driver.find_element_by_xpath('//*[@id="container_costs_vs_counterfactual"]/div/div[12]') # Read text below GDP value
        cost_flag = userid_element.text   
        
        # Find sign of GDP (less expensive => increase; more expensive => decrease)
        if cost_flag == 'less expensive': 
            cost_output = -cost_output # Reverse sign
            
        # Go back to the overview section
        try:
            driver.find_element_by_xpath('//*[@id="mn-1"]').click() 
        except: # Problem going back to the overview section? The code below sorts it
            time.sleep(0.2)
            id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
            id_box.click()
        output = [output, cost_output] # Output temperature and cost values
    return  output



def generate_chromosome(constraint = False, constraint_levers = [], constraint_values = [], 
                        threshold = False, threshold_names = [], thresholds = []):
    """
    Initialises a chromosome and returns its corresponding lever values, and temperature and cost. 

    **Args**:

    - constraint (*boolean*): Flag to select whether any inputs have been fixed. 
    - constraint_levers (*list*): Contains the name of levers to be fixed.  
    - constraint_values (*list*): Contains the values to fix the selected levers to. 
    - threshold (*boolean*): Flag to select whether any inputs have to be bounded within a range. 
    - threshold_names (*list*): Contains the name of the levers to be bounded within a range. 
    - thresholds (*list*): Contains the upper and lower bound for each specified lever. 

    **Returns**:
    
    Lever values corresponding to generated chromosome and temperature-cost values corresponding to the current chromosome. 
    """
    lever_names = list(dfs_3.iloc[:, 0].to_numpy()) # Create list with all lever names
    # Generate random lever combination
    random_lever_values = new_lever_combination(threshold = threshold, threshold_names = threshold_names, thresholds = thresholds) 
    # Fix specified input levers
    if constraint == True:
        lever_names, random_lever_values = overwrite_lever_values(lever_names, random_lever_values, constraint_levers,  constraint_values)
    result = move_lever(lever_names, random_lever_values, costs = True, constraint = constraint, constraint_levers = constraint_levers, constraint_values = constraint_values) # Move lever accordingly and read temperature and cost valuesw
    return random_lever_values, result

def fitness(chromosome, target_temperature = 3000, target_cost = 0, output_constraint = False, current_output_values = [], output_constraint_names = [], output_constraints = [], std_devs = []):
    """Need to apply some sort of normalisation. Divide by standard deviation"""
    total_cost = 0 # Initialise fitness
    lever_names = list(dfs_3.iloc[:, 0].to_numpy()) # Create list with all lever names
    output_names = list(dfs.iloc[:, 0].to_numpy()) # Create list with all output names
    if output_constraint == True: # If output constraints have been set, take into account in fitness function
        for i in range(len(output_constraint_names)): # Iterate through output constraints
            if output_constraint_names[i] in  output_names: # Ensure name is correct, otherwise ignore constraint
                # Equality constraint of specified output value. Normalised by diving it by an estimate of its standard deviation. 
                total_cost += (abs(current_output_values[output_names.index(output_constraint_names[i])] - output_constraints[i]))/std_devs[output_names.index(output_constraint_names[i])]
    # Equality constraint for temperature value and inequality constraint for cost value (normalised with their approxiamte std dev)
    total_cost += (abs(chromosome[0] - target_temperature)/18204.97) + ((chromosome[1] - target_cost)/34.59)
    return  total_cost


def selection(fitness_values):
    """Given an array of fitness values, find the indeces and the values of the 2 fittest candidates -> Parents"""
    fittest_index = sorted(range(len(fitness_values)), key = lambda sub: fitness_values[sub])[:2] # Find 2 fittest candidates
    parent_1, parent_2 = lever_values[fittest_index [0]], lever_values[fittest_index [1]] # Find values of parents
    return parent_1, parent_2, fittest_index

def mutated_genes(lever_value, thresholds = [1, 3.9], threshold = False, threshold_name = "", threshold_value = ""):
    """Mutate gene by randomly moving a lever up or down by 0.1. Returns the mutated gene (the new lever value)"""
    move = -0.
    prob = random.randint(0, 100)/100 # Generate random gene
    if prob < 0.5: move = -0.1 # Move lever down
    else: move = 0.1 # Move lever up
    # If the lever value is out of bounds, reverse direction of step (taking specified threshold into account)
    if threshold == True:
        if (lever_value + move < threshold_value[0]) or (lever_value + move > threshold_value[1]):
            move = -move
    else:
        if (lever_value + move < thresholds[0]) or (lever_value + move > thresholds[1]):
            move = -move
    return round(lever_value + move, 3)

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
    
def overwrite(levers, values, address = current_url):
    """Given a set of lever names and corresponding values, overwrite specified lever combination"""
    for i in range(len(levers)): # Overwrite 1 value at a time
        address = new_URL(levers[i], values[i], address = address)  
    return address

def overwrite_lever_values(levers, values, constraint_levers, constraint_values):
    """
    Given a set of lever names and corresponding values, and a set of 
    constrained lever names and their values, return the corresponding lever combination and its values. 
    """
    for i in range(len(levers)): # Iterate through all levers
        if levers[i] in constraint_levers: # If current lever is constrained
            values[i] = constraint_values[constraint_levers.index(levers[i])] # Update
    return levers, values

def read_outputs():
    """Reads all outputs and returns them as a list (empirical scraping)"""
    time.sleep(0.2) 
    compare_box = driver.find_element_by_xpath('//*[@id="mp-nav-compare"]') # Move to the "Compare" section
    time.sleep(0.1)
    try: 
        compare_box.click()
    except:
        id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
        id_box.click()
        time.sleep(0.1)
        compare_box.click()
    out_vals = []
    for i in range(len(dfs)): 
        userid_element = driver.find_element_by_xpath(dfs.iloc[i, 2])
        out_vals.append(float(userid_element.text.rstrip("%")))
    time.sleep(0.1)   
    try:
        driver.find_element_by_xpath('//*[@id="mn-1"]').click() 
    except: # Problem going back to the overview section? The code below sorts it
        time.sleep(0.2)
        id_box = driver.find_element_by_id('lets-start') # Bypass "Start" screen
        id_box.click()
    return out_vals