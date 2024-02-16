import os
import numpy as np
import ray
import warnings


###############################################################################
# Local functions
###############################################################################
def scaling(X,mediumsize,fixedmediumvalue):
    """
    Transform X from experiemental concentration to 
    upper bound intake flux for better fitting with cobra
    """
    if isinstance(X,np.ndarray) != True:
        X = np.float32(X.to_numpy())
#    X = X[:,:mediumsize]
    medium_X = X*fixedmediumvalue
#    medium_X[medium_X < 1.0e-4] = 0
    return medium_X


def knock_out(model, i ,gene,genename, rebound = False, esp = 1e-1):
    """
    Find the reactions with 0 fluxes and
    eliminate them in cobra model

    If rebound is True, find the reactions with 0 fluxes (KO) and
    rebound them in cobra model, used to set a very small upper bound for KO reactions
    """
    if isinstance(genename,np.ndarray) != True:
        genename = np.array(genename)
    KO = genename[gene[i,]==0]

    if rebound: 
        bound = (0,esp)
        for react in KO:
            model.reactions.get_by_id(react).bounds = bound 
    else:        
        for react in KO:
            model.reactions.get_by_id(react).knock_out()
    return model 


def set_medium(cobramodel,medium, exact = False):
    """
    Update the medium of a CobraModel with the provided medium dictionary.
    If a medium component exists in both the original CobraModel's medium and the provided
    medium dictionary, its value will be replaced with the value from the dictionary.
    If a medium component is not present in the original medium, it will be added.

    IF exact = True, set reaction upper bound = lower bound = fluxes-in
    Parameters:
        cobramodel: The CobraModel to update.
        medium (dict): A dictionary of medium components and their values.

    Returns:
        cobramodel: The updated Cobramodel with the new medium components.
    """
    if not exact:
        medium_ori = cobramodel.medium
        medium_ori = {key: 0 for key in medium_ori}

        for key, value in medium.items():
            medium_ori[key] = value

        cobramodel.medium = medium_ori
    else:
        for key, value in medium.items():
            bound = (value,value)
            cobramodel.reactions.get_by_id(key).bounds = bound
    return cobramodel


def fit_cobra(i, medium_X, mediumname, cobramodel, gene, genename , objective):
    """
    Fit a COBRA model with A SINGLE data point.

    Parameters:
    - i (int): position in data array.

    Returns:
    - y_cobra (float or None): The optimized value of the specified objective function
    - rows_with_warnings (list): A list containing indices of rows where warnings were generated during optimization.
    """
    if i > medium_X.shape[0]: 
        return

    medium = dict(zip(mediumname, medium_X[i,:]))
    rows_with_warnings = []

    with warnings.catch_warnings(record=True) as w, cobramodel:
        cobramodel = set_medium(cobramodel,medium, exact = True)
        if len(gene) > 0:
            cobramodel = knock_out(cobramodel,i,gene,genename, rebound = True, esp = 0)
        solution = cobramodel.optimize()
        y_cobra = solution.fluxes[objective]
        if w:
            rows_with_warnings.append(i)
    
    return y_cobra, rows_with_warnings


def unload(result, batch_size):
    """
    For parallelization, unload lists of results into separated 'y' and 'warn'.

    Args:
    - result (list): A list of results, where each result is a sublist of shape (batch_size, 2).
    - batch_size (int): The size of each batch in the results.

    Returns:
    - y (list): A list containing 'y' prediction values from the results.
    - warn (list): A list containing indexes where solver cannot solve input medium and throw warning.
    """
    y, warn = [], []
    for i in range(len(result)):
        for j in range(batch_size):
            y.append(result[i][j][0])
            warn.append(result[i][j][1])
    return y, warn


@ray.remote
def cobra_one_medium(start, end, medium_X, mediumname, cobramodel, gene, genename , objective):
    """
    Fit COBRA model with a BATCH of X

    Parameters:
    - start (int): The start index of the batch in X array.
    - end (int): The end index of the batch in X array.
    - medium_X (numpy.ndarray): A 2D array contains SCALED medium conditions 

    Returns:
    - results (list): A list of results, where each result is a tuple containing 
        both y_cobra and row indices where there was warning.
    """
    return [fit_cobra(i, medium_X, mediumname, cobramodel, gene, genename , objective) for i in range(start, end)]


###############################################################################
# Callable function in other scripts
###############################################################################
def run_cobra_parallel(X, cobramodel, mediumsize, mediumname, fixedmediumvalue,  genename, objective):
    """
    Run parallel cobra predictions with a different batch on each CPU core

    Parameters:
    - X (numpy.ndarray): Input dataset for predictions.
    - cobramodel (cobra.Model): A COBRApy model to perform optimizations on.
    - mediumsize (int): The size of the medium.
    - mediumname (list): reaction names.
    - fixedmediumvalue (float): The scaling value.
    - genename (list): A list of gene names corresponding to the genes to be knocked out.
    - objective (str): The objective function to optimize in the COBRA model.

    Returns:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
#    cobramodel.solver.configuration = optlang.glpk_interface.Configuration(timeout=5, presolve='auto', lp_method='simplex')
    ray.init()
    X = scaling(X, mediumsize, fixedmediumvalue)
    medium = X[:,:mediumsize]
    gene = X[:,mediumsize:]
    size = medium.shape[0]
    batch_size = round(size/os.cpu_count())
    result_id = []

    for x in range(os.cpu_count()):
        result_id.append(cobra_one_medium.remote(x*batch_size,
                                        (x+1)*batch_size,
                                        medium, 
                                        mediumname, 
                                        cobramodel, 
                                        gene, 
                                        genename , 
                                        objective))

    result = ray.get(result_id)
    ray.shutdown()

    y_cobra, warning = unload(result, batch_size)
    y_cobra = np.array(y_cobra).ravel()
    return y_cobra, warning


def run_cobra_slow(X, cobramodel, mediumsize, mediumname, fixedmediumvalue,  genename, objective):
    """
    Run parallel cobra predictions without parallel

    Parameters:
    - X (numpy.ndarray): Input dataset for predictions.
    - cobramodel (cobra.Model): A COBRApy model to perform optimizations on.
    - mediumsize (int): The size of the medium.
    - mediumname (list): reaction names.
    - fixedmediumvalue (float): The scaling value.
    - genename (list): A list of gene names corresponding to the genes to be knocked out.
    - objective (str): The objective function to optimize in the COBRA model.

    Returns:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
    X = scaling(X, mediumsize, fixedmediumvalue)
    medium_X = X[:,:mediumsize]
    gene = X[:,mediumsize:]
    size = medium_X.shape[0]
    y_cobra = []
    rows_with_warnings = []
    for i in range(size):
        medium = dict(zip(mediumname, medium_X[i,:]))

        with warnings.catch_warnings(record=True) as w, cobramodel:
            cobramodel = set_medium(cobramodel,medium, exact = False)
            if len(gene) > 0:
                cobramodel = knock_out(cobramodel,i,gene,genename, rebound = True, esp = 0)
            solution = cobramodel.optimize()
            y = solution.fluxes[objective]
            y_cobra.append(y)
            if w:
                rows_with_warnings.append(i)
                
    y_cobra = np.array(y_cobra).ravel()
    
    return y_cobra, rows_with_warnings


def run_cobra(XX, parameter):
    """
    Choose running cobra parallel for big dataset (biolog/biolog_mini) or without
    Return:
    - y_cobra (numpy.ndarray): Predicted values (Y_cobra) from the COBRA model.
    - warning (list): A list containing indices of rows with warnings during predictions.
    """
    if parameter.parallel == True:
        return run_cobra_parallel(XX, 
                                    parameter.cobramodel, 
                                    parameter.mediumsize, 
                                    parameter.medium, 
                                    parameter.fixedmediumvalue, 
                                    parameter.genename, 
                                    parameter.objective)
    else:
        return run_cobra_slow(XX, 
                                    parameter.cobramodel, 
                                    parameter.mediumsize, 
                                    parameter.medium, 
                                    parameter.fixedmediumvalue, 
                                    parameter.genename, 
                                    parameter.objective)
        