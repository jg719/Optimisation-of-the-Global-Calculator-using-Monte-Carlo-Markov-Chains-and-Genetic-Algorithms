Defining the optimisation constraints
-------------------------------------

.. code:: ipython3

    # Specify input levers to fix
    constraint_lever_names = ['CCS electricity', 
    'GGR1', 
    'GGR2', 
    'GGR3', 
    'GGR4']
    constraint_lever_values = [1, 1, 1, 1, 1]
    
    # Specify input levers to bound within a threshold
    threshold_names  = [ 'Solar', 
                        'Wind',
                        'Global population',
     'Electric & hydrogen',
     'CCS manufacturing',
    'Nuclear',
    'Calories consumed',
     'Quantitiy of meat',
      'Type of meat',
     'Livestock grains/residues fed', 
    'Land-use efficiency']
    thresholds = [[2.6, 3.2], [2.5, 3.0], [1.6, 2.0], [2.8, 3.1], [1, 2], [1.5, 2], [2, 3], [2, 3], [2, 3], [1.8, 2.2], [1.8, 2.2]]
    
    # Specify output constraints
    output_constraint_names = ['Total energy demand (EJ / year)', 'Forest area (native and commercial, millions of hectares']
    output_constraints = [420, 4100]
