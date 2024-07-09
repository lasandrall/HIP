#!~/Python/env/bin/python3

### Version for Ranking

# all_functions.py
# Contains the following functions
# (from main_functions.py, helper_functions.py, and adagrad_functions.py):

## Main functions:
#   generate_data
#   standardize_dat
#   select_K_simple
#   select_lambda_CV
#   optimize_torch
#   train_mse
#   test_mse
#   train_class
#   test_class
#   train_pois
#   test_pois
#   train_zip
#   test_zip
#   + helper and adagrad functions
#     not meant to be called by the user


# Author: Jessica Butts
# Last Updated: August 2023

# Using Python 3.8.3

# Imports and Set-up
#----------------------------------------------------------------------------------
import torch
import math
import numpy as np
import time
import random # used inside optimize_torch_cont
import pickle
import itertools
import copy

from scipy import special

# to implement parallel processing
from joblib import Parallel, delayed, cpu_count

# needed for plotting
import matplotlib.pyplot as plt

# Set default tensor type
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

#----------------------------------------------------------------------------------
# generate_data: Generate simulated data to use with HIP
# Required Inputs:
#   seed - int or string - set a seed for replicability
#   n - int vector - number of subjects desired in each subgroup; should be length S
#   p - int vector - number of covariates desired in each data set; should be length D
#   K - int - number of latent components to use in generating the data
#   D - int - number of data sets to generate
#   S - int - number of subgroups
#   nonzero - int - number of important variables for each subgroup; same for all subgroups
#   offset - int - how many variables to offset the important variables between subgroups; same for all subgroups
#   sigma_x - double - factor by which the errors added to X are scaled by
#   sigma_y - double - factor by which the errors added to Y are scaled by; not applicable when family = 'poisson' or 'zip'
#   family - string - determines what type of outcome is generated;
#        - 'gaussian' - q should also be specified or q will default to 1
#        - 'multiclass' - m should also be specified or m will default to 2
#        - 'poisson'
#        - 'zip'
# Optional Inputs:
#   m - int - number of classes in multiclass outcome; can be >= 2; default = 2
#   q - int - number of continuous outcomes; can be >= 1; default = 1
#   theta_init - double matrix - matrix to use for theta; if None, then a theta matrix will be generated from a U(0,1) distribution
#   B - matrix list - B^d,s matrices used to generate the data; will randomly generate if not provided
#   Z - matrix list - Z^s matrices used to generate the data; if not providedm, will generate as N(mu = z_mean, sigma = z_sd)
#   y_mean - double - Value for mean of Y; default = 0
#   y_sd - double - Value for sd of Y; default = 1
#   z_mean - double - Value for mean of Y; default = 0
#   z_sd - double - Value for sd of Y; default = 1
#   beta - double - Value of intercept term; default = 0
# Outputs:
#   dict with the following elements:
#       X - matrix list - Contains all X^d,s matrices; access as X[d][s]
#       Y - matrix list - Contains all Y^s matrices; access as Y[s]
#       Z - matrix list - Contains all Z^s matrices used to generate the data; access as Z[s]
#       B - matrix list - Contains all B^d,s matrices used to generate the data; access as B[d][s]
#       theta - matrix - Contains the theta matrix used in generating the data
#       beta - double - Value for intercept
#       tau - double - If family = 'zip', proportion of observations in zero state with default = 0.25; None ow
#       y_mean - double - Value for mean of Y
#       y_sd - double - Value for sd of Y
#       z_mean - double - Value for mean of Y
#       z_sd - double - Value for sd of Y
#       seed - int or string - Value of seed used to generate the data
def generate_data(seed, n, p, K, D, S, nonzero, offset, sigma_x, sigma_y, family, q = 1, m = 2, theta_init = None, B = None, Z = None, y_mean=0, y_sd=1, z_mean=0, z_sd=1, beta=0, tau=None, t_dist = False, t_df = 20):

    # Check inputs make sense
    if len(n) != S:
        raise Exception("The length of n does not match the number of subgroups.")
    if len(p) != D:
        raise Exception("The length of p does not match the number of data views.")
    if sigma_x < 0 or sigma_y < 0:
        raise Exception("Sigma_x and sigma_y must be non-negative.")

    # set seed for reproducibility
    torch.manual_seed(seed)
    
    nonzero_vec = [nonzero for d in range(D)]
    offset_vec = [offset for d in range(D)]

    if family == 'gaussian':
    
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != q):
            raise Exception("The dimensions of theta are incorrect. It must be K x q.")
    
        # set true theta value depending on K and q
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, q).uniform_(0, 1)
        if B == None:
            B = [[torch.linalg.qr(torch.cat((torch.zeros(offset_vec[d]*s, K),
                                             torch.Tensor(nonzero_vec[d], K).uniform_(0.5, 1.0)*(-1)**torch.Tensor(nonzero_vec[d], K).bernoulli_(0.5),
                                             torch.zeros(p[d]-nonzero_vec[d]-offset_vec[d]*s, K)), dim=0),
                                  mode='reduced').Q for s in range(S)] for d in range(D)]
        if Z == None:
            Z = [torch.normal(mean=z_mean, std=z_sd, size=(n[s], K)) for s in range(S)]
            
        if t_dist == False:
            E = [[torch.normal(0, sigma_x, (n[s], p[d])) for s in range(S)] for d in range(D)]
            
        else:
            # torch.manual_seed(seed)                # uncomment if results are inconsistent (may work since set earlier in code)
            print("using t-distribution for E^d,s")

            # use the fact that for Z ~ N(0,1), V ~ chisq(DF), Z/sqrt(V/DF)
            
            # generate values from standard normal and chi-square distributions
            # std_normal = [[torch.normal(mean=0, std=1, size=(n[s], K)) for s in range(S)] for d in range(D)]
            # chisq = [[torch.chisquare(df = t_df, size=(n[s], K)) for s in range(S)] for d in range(D)]
            
            t_dist = torch.distributions.studentT.StudentT(df = t_df)
            #TODO: need to do a bit more work in typecasting since this seems to work
            # but returns each object as a list of length 1, so need to get rid of the list
            # wrapping I think
            E  = [[t_dist.sample(sample_shape=((n[s], p[d]))) for s in range(S)] for d in range(D)] 
        
        
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        Y = [beta + torch.matmul(Z[s], theta_init) + torch.normal(y_mean, sigma_y*y_sd, (n[s], q)) for s in range(S)]

    elif family == 'multiclass':
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != m):
            raise Exception("The dimensions of theta are incorrect. It must be K x m.")
            
        # set true theta value depending on K and q
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, m).uniform_(0, 1)
        if B == None:
            B = [[torch.linalg.qr(torch.cat((torch.zeros(offset_vec[d]*s, K),
                                             torch.Tensor(nonzero_vec[d], K).uniform_(0.5, 1.0)*(-1)**torch.Tensor(nonzero_vec[d], K).bernoulli_(0.5),
                                             torch.zeros(p[d]-nonzero_vec[d]-offset_vec[d]*s, K)), dim=0),
                                  mode='reduced').Q for s in range(S)] for d in range(D)]
        if Z == None:
            Z = [torch.normal(mean=z_mean, std=z_sd, size=(n[s], K)) for s in range(S)]
        E = [[torch.normal(0, sigma_x, (n[s], p[d])) for s in range(S)] for d in range(D)]
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        Y = [torch.empty((n[s],m)) for s in range(S)]
        for s in range(S):
            P = calc_probs(beta + Z[s].matmul(theta_init) + torch.normal(mean = 0, std = sigma_y, size = (n[s], m)))
            Y[s] = class_matrix(torch.max(P, dim=1).indices, S = 1)
    
    elif family == 'poisson':
        # Check dimension of theta
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != 1):
            raise Exception("The dimensions of theta are incorrect. It must be K x 1 for a poisson outcome.")
    
        # set true theta value depending on K and q
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, q).uniform_(0, 1)
        
        if B == None:
            B = [[torch.linalg.qr(torch.cat((torch.zeros(offset_vec[d]*s, K),
                                             torch.Tensor(nonzero_vec[d], K).uniform_(0.5, 1.0)*(-1)**torch.Tensor(nonzero_vec[d], K).bernoulli_(0.5),
                                             torch.zeros(p[d]-nonzero_vec[d]-offset_vec[d]*s, K)), dim=0),
                                  mode='reduced').Q for s in range(S)] for d in range(D)]
        
        if Z == None:
            Z = [torch.normal(mean=z_mean, std=z_sd, size=(n[s], K)) for s in range(S)]
        
        E = [[torch.normal(0, sigma_x,(n[s], p[d])) for s in range(S)] for d in range(D)]
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        
        # to prevent extremely large Y values, standardize Z first
        Z_std = standardize_dat(Y = Z, standardize = 'subgroup', std_type = 'scale_center')['Y']
        Y_out = [torch.poisson(torch.exp(beta + torch.matmul(Z_std[s], theta_init)))  for s in range(S)]

        # add column of ones for offset
        Y = [torch.cat((Y_out[s], torch.ones_like(Y_out[s])), dim = 1) for s in range(S)]
        
    elif family == 'zip':
        # Check dimension of theta
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != 1):
            raise Exception("The dimensions of theta are incorrect. It must be K x 1 for a poisson outcome.")
        
        # check range of tau
        if tau != None and (tau < 0 or tau > 1):
            raise Exception("Invalid tau value; tau must be in [0,1].")
    
        # set true theta value depending on K and q
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, q).uniform_(0, 1)
            
        # set default tau = 0.25 if tau not specified
        if tau == None:
            tau = 0.25
        
        if B == None:
            B = [[torch.linalg.qr(torch.cat((torch.zeros(offset_vec[d]*s, K),
                                             torch.Tensor(nonzero_vec[d], K).uniform_(0.5, 1.0)*(-1)**torch.Tensor(nonzero_vec[d], K).bernoulli_(0.5),
                                             torch.zeros(p[d]-nonzero_vec[d]-offset_vec[d]*s, K)), dim=0),
                                  mode='reduced').Q for s in range(S)] for d in range(D)]
        
        if Z == None:
            Z = [torch.normal(mean=z_mean, std=z_sd, size=(n[s], K)) for s in range(S)]
        
        E = [[torch.normal(0, sigma_x,(n[s], p[d])) for s in range(S)] for d in range(D)]
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        
        # Standard Poisson
        Z_std = standardize_dat(Y = Z, standardize = 'subgroup', std_type = 'scale_center')['Y']
        Y_out = [torch.poisson(torch.exp(beta + torch.matmul(Z_std[s], theta_init)))  for s in range(S)]
        # But need to have them be 0 with prob tau independent of linear predictor
        mask = [torch.bernoulli((1-tau)*torch.ones_like(Y_out[s])) for s in range(S)]
        
        # add column of ones for offset
        Y = [torch.cat((Y_out[s]*mask[s], torch.ones_like(Y_out[s])), dim = 1) for s in range(S)]
            
    
    else:
        print("Invalid family specification.")

    # tau only relevant for 'zip'
    return {'X':X, 
            'Y':Y, 
            'Z':Z, 
            'B':B,
            'theta':theta_init,
            'beta':beta,
            'tau': tau,
            'y_mean': y_mean,
            'y_sd': y_sd,
            'z_mean': z_mean,
            'z_sd': z_sd,
            'seed': seed
           }

#----------------------------------------------------------------------------------
# standardize_dat: Standardizes input data
# Required Inputs:
#   standardize - string - One of "all", "subgroup", or "none"
#   std_type - string - One of "scale_center", "center", or "norm"
# Optional Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Y - matrix list - list of Y^s matrices containing outcomes
#   X_train - matrix list - list of X^d,s matrices whose mean and sd will be used to standardize X
#   Y_train - matrix list - list of Y^s matrices whose mean andd sd will be used to standardize Y
#   std_x - boolean - indicates whether to standardize X; default = True
#   std_y - boolean - indicates whether to standardize Y; default = True
# Outputs:
#   dict with the following elements:
#       X - matrix list - standardized X^d,s if standardization was requested
#       Y - matrix list - standardized Y^s if standardization was requested
def standardize_dat(standardize, std_type, X=None, Y=None, X_train=None, Y_train=None, std_x=True, std_y=True, verbose=True):
    X_copy = copy.deepcopy(X)
    Y_copy = copy.deepcopy(Y)
    
    # If X is provided, standardize it
    if X != None and std_x:
        D = len(X)
        S = len(X[0])
    
        if X_train == None:
            X_train = copy.deepcopy(X)
        
        # Standardize across all subgroups
        if standardize == 'all':
            if std_type == 'norm':
                for d in range(D):
                    x_norm = torch.linalg.matrix_norm(torch.cat([X_train[d][s] for s in range(S)], dim=0), "fro")
                    for s in range(S):
                        X_copy[d][s] = torch.div(X_copy[d][s], x_norm)
                                
            elif std_type == 'scale_center':
                for d in range(D):
                    x_mean = torch.mean(torch.cat([X_train[d][s] for s in range(S)], dim=0), dim=0)
                    x_std = torch.std(torch.cat([X_train[d][s] for s in range(S)], dim=0), dim=0)
                    for s in range(S):
                        X_copy[d][s] = torch.div((X_copy[d][s] - x_mean), x_std)
                    
            elif std_type == 'center':
                for d in range(D):
                    x_mean = torch.mean(torch.cat([X_train[d][s] for s in range(S)], dim=0), dim=0)
                    for s in range(S):
                        X_copy[d][s] = X_copy[d][s] - x_mean
            else:
                raise Exception("Invalid std_type specification.")
            
            if verbose:
                print('X standardized across subgroups')
    
        # Option to standardize by subgroup
        elif standardize == 'subgroup':
            if std_type == 'norm':
                for d in range(D):
                    for s in range(S):
                        x_norm = torch.linalg.matrix_norm(X_train[d][s], "fro")
                        X_copy[d][s] = torch.div(X_copy[d][s], x_norm)
            
            elif std_type == 'scale_center':
                for d in range(D):
                    for s in range(S):
                        x_mean = torch.mean(X_train[d][s], dim=0)
                        x_std = torch.std(X_train[d][s], dim=0)
                        X_copy[d][s] = torch.div((X_copy[d][s] - x_mean), x_std)
        
            elif std_type == 'center':
                for d in range(D):
                    for s in range(S):
                        x_mean = torch.mean(X_train[d][s], dim=0)
                        X_copy[d][s] = X_copy[d][s] - x_mean
            else:
                raise Exception("Invalid std_type specification.")
            
            if verbose:
                print('X standardized within subgroups')
   
        else:
            if verbose:
                print('X not standardized')
    
    # If Y is provided, standardize it
    if Y != None and std_y:
        S = len(Y)
        
        if Y_train == None:
            Y_train = copy.deepcopy(Y)
        
        if standardize == 'all':
            if std_type == 'norm':
                y_norm = torch.linalg.matrix_norm(torch.cat(Y_train), "fro")
                for s in range(S):
                    Y_copy[s] = torch.div(Y_copy[s], y_norm)
                                          
            elif std_type == 'scale_center':
                y_mean = torch.mean(torch.cat(Y_train), dim=0)
                y_std = torch.std(torch.cat(Y_train), dim=0)
                for s in range(S):
                    Y_copy[s] = torch.div((Y_copy[s] - y_mean), y_std)
            
            elif std_type == 'center':
                y_mean = torch.mean(torch.cat(Y_train), dim=0)
                for s in range(S):
                    Y_copy[s] = Y_copy[s] - y_mean
            
            else:
                raise Exception("Invalid std_type specification.")
            
            if verbose:
                print("Y standardized across subgroups")
        
        elif standardize == 'subgroup':
            if std_type == 'norm':
                for s in range(S):
                    y_norm = torch.linalg.matrix_norm(Y_train[s], 'fro')
                    Y_copy[s] = torch.div(Y_copy[s], y_norm)
            
            elif std_type == 'scale_center':
                for s in range(S):
                    y_mean = torch.mean(Y_train[s], dim=0)
                    y_std = torch.std(Y_train[s], dim=0)
                    Y_copy[s] = torch.div((Y_copy[s] - y_mean), y_std)
                    
            elif std_type == 'center':
                for s in range(S):
                    y_mean = torch.mean(Y_train[s], dim=0)
                    Y_copy[s] = Y_copy[s] - y_mean
            
            else:
                raise Exception("Invalid std_type specification.")
            
            if verbose:
                print("Y standardized within subgroups")
        
        else:
            if verbose:
                print('Y not standardized')  
    
    # will return NULL if argument not provided
    return {'X': X_copy, 
            'Y': Y_copy}

#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# select_K_simple: Uses the simple approach to select a value for K
# Required Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
# Optional Inputs:
#   threshold - double - value to use as the threshold in the simple approach; default = 0.2
#   verbose - boolean - whether to print information about the K selected; default = True
#   cat - boolean - whether to concatenate X before calculation; default = True
#   vals - string - whether to use singular values or eigenvalues; default = 'eigen'
# Outputs:
#   dict with following elements:
#       kchoose - int - suggested value for K
#       values - list<double> - either singular values or eigenvalues depending on `vals`
#       calc - list<double> - calculated values
#       data - tensor<double> - tensor of data used in calculation
def select_K_simple(X, threshold = 0.2, verbose=True, cat = True, vals = "eigen"):

    if cat:
        D = len(X)

        # Form concatenated data matrix
        dat = torch.cat([torch.cat(X[d]) for d in range(D)], dim=1)
    
    else:
        dat = copy.deepcopy(X)
        
    if vals == 'singular':
        # Perform SVD to get singular values
        Z_u, Z_d, Z_vt = torch.svd(dat)
        
    elif vals == 'eigen':
        Z_d = torch.pow(torch.linalg.svdvals(dat),2)/(dat.shape[0] - 1)
        
    else:
        raise Exception("Invalid vals argument in select_K_simple.")
        
    calc = list()
    
    for j in range(1, len(Z_d)):
        val = Z_d[j]/torch.sum(Z_d[0:j])
        calc.append(val)
        if val < threshold:
            kchoose = j
            break
    
    if verbose:
        print("K Based on simple approach using", threshold, "as cut-off:", kchoose)
    
    return {'kchoose': kchoose,
            'values': Z_d,
            'calc': calc,
            'data': dat
            }
        
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# select_lambda_CV: function to tune lambda parameter using cross-validation
# Required Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Y - matrix list - list of Y^s matrices containing outcomes
#   gamma - list<double> - indicators of whether to penalize each data view; should be length D
#   family - string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#   ncore - int - number of cores to use in parallel processing
#   topn - int or list<int> - number of variables to retain; different values may be specified for each view using a list of length D
# Optional Inputs:
#   K - int - number of latent components; will use 'select_K_simple' to choose if not provided
#   k_thresh - double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#   update_thresh - double - threshold to use for suboptimization convergence criterion; default = 10**-5
#   epsilon - double - threshold to use for overall convergence criterion; default = 10**-4
#   max_iter - int - maximum number of outer loop iterations; default = 50
#   folds - int - number of CV folds; default = 5
#   search - string - what type of search to perform for lambda parameter; default = "random"
#       - "random" - tries a random selection of rand_prop*num_steps lambda values in grid
#       - "grid" - tries all lambda values in grid
#   xi_range - list<double> - minimum  and maximum values to consider for lambda_xi; default = [0.0, 2.0]
#   g_range - list<double> - minimum  and maximum values to consider for lambda_g; default = [0.0, 2.0]
#   num_steps - dict - number of steps to use in lambda grid
#       - 'Xi' - int - number of steps for lambda_xi; default = 8
#       - 'G' - int - number of steps for lambda_g; default = 8
#   rand_prop - double - proportion of grid points to search when search = "random"; must be > 0 and < 1; default = 0.20
#   standardize - string - One of "all", "subgroup", or "none"; default = "subgroup"
#   std_type - string - One of "scale_center", "center", or "norm"; default = "scale_center"
#   std_x - boolean - indicates whether to standardize X; default = True
#   std_y - boolean - indicates whether to standardize Y; default = True
#   verbose - boolean - whether to print additional information during optimization; default = False
# Outputs:
#   dict with the following elements:
#       search_results - list - list with a dict of results returned from optimize_torch for each lambda value tried
#       total_time - double - time to complete entire search in seconds
#       xi_range - list<double> - minimum and maximum values considered for lambda_xi
#       g_range - list<double> - minimum and maximum values considered for lambda_g
#       num_steps - dict - number of steps used in lambda grid
#       search - string - type of search performed for selecting lambda parameters
def select_lambda_CV(Y, X, gamma, family, ncore, topn,
                     K = None, k_thresh = 0.2,
                     update_thresh = 10**-5, epsilon = 10**-4, max_iter = 50,
                     folds = 5, search = 'random', xi_range = [0.0, 2.0], g_range = [0.0, 2.0], num_steps = {'Xi': 8, 'G': 8}, rand_prop = 0.20,
                     standardize = 'subgroup', std_type = 'scale_center', std_x = True, std_y = True,
                     verbose = False):
                     
    #Get n, p, d, s from the X provided from user
    D = len(X)
    S = len(X[0])
    n = [X[0][s].shape[0] for s in range(S)]
    p = [X[d][0].shape[1] for d in range(D)]
    q = Y[0].shape[1]
    
    # Run basic data validation checks - will throw error if any check fails
    validate_data(X = X, Y = Y, gamma = gamma, xi_range = xi_range, g_range = g_range, rand_prop = rand_prop)
    
    if verbose:
        print("D =", D)
        print("S =", S)
        print("n =", n)
        print("p =", p)
        print("q =", q)
    
    # define test error function based on family.
    global test_error
    
    if family == 'gaussian':
        test_error = test_mse
    elif family == 'multiclass':
        test_error = test_class
        std_y = False #should not standardize Y ever when multiclass outcome.
    elif family == 'poisson':
        test_error = test_pois
        std_y = False #should not standardize Y ever when poisson outcome.
    elif family == 'zip':
        test_error = test_zip
        std_y = False #should not standardize Y ever when zip outcome.
    else:
        raise Exception("Invalid family specification.")
    
    search_tic = time.perf_counter()
    
    if K == None:
        K = select_K_simple(X=X, threshold=k_thresh, verbose=True)

    # Search over Xi; fix lam_g = 1.0
    # Allow for fixing one of the lambda values (i.e., upper  and lower are the same)
    test = []
    #  Xi values.
    if xi_range[0] == xi_range[1]:
        test.append([xi_range[0]])
    else:
        test.append([j*(xi_range[1]-xi_range[0])/(num_steps['Xi']) + xi_range[0] for j in  range(0, num_steps['Xi']+1)][1:])
    #  G values.
    if g_range[0] == g_range[1]:
        test.append([g_range[0]])
    else:
        test.append([j*(g_range[1]-g_range[0])/(num_steps['G']) + g_range[0] for j in  range(0, num_steps['G']+1)][1:])
    
    # All possible combinations of lambda datasets
    combos = list(itertools.product(*test))
    
    # Random or Grid search
    if search == 'random':
        random.seed(7)
        selection = random.sample(range(len(combos)), math.ceil(rand_prop*(len(combos))))
    elif search == 'grid':
        selection = range(len(combos))  
    
    # set seed for reproducability
    random.seed(777)
    result = [[] for f in range(folds)]
    test_err_dict = {''.join(str(combos[val])): {'full': torch.empty(folds), 'subset': torch.empty(folds)} for val in selection}

    # Create set of indices for each fold
    rows = [list() for s in range(S)]
    mods = [n[s]%folds for s in range(S)]
    per_fold = [[n[s]//folds + 1 if f < mods[s] else n[s]//folds for f in range(folds)] for s in range(S)]
    
    for s in range(S):
        shuffle = random.sample(range(n[s]), k = n[s])
        start = 0
        for f in range(folds):
            rows[s].append(shuffle[start:start+per_fold[s][f]])
            start += per_fold[s][f]

    # For each fold
    for f in range(folds):
        # Set the test and training data
        
        # keep given fold out as test set
        keep = [[False if l in rows[s][f] else True for l in range(n[s])] for s in range(S)]
        
        X_test = [[X[d][s][rows[s][f]].clone().detach() for s in range(S)] for d in range(D)]
        X_train = [[X[d][s][keep[s]].clone().detach() for s in range(S)] for d in range(D)]
        
        Y_test = [Y[s][rows[s][f]].clone().detach() for s in range(S)]
        Y_train = [Y[s][keep[s]].clone().detach() for s in range(S)]
        
        dat_test_std = standardize_dat(standardize=standardize, std_type=std_type, 
                                       X=X_test, Y=Y_test, X_train=X_train, Y_train=Y_train, std_y=std_y)
        
        # result is now going to have length = f with each entry a list of all lambda values for that fold.
        result[f] = Parallel(n_jobs=ncore,prefer="processes",verbose=100, pre_dispatch='1.5*n_jobs')(delayed(optimize_ranking)(Y = Y_train, X = X_train, family = family, topn = topn,
                                                                   lambda_xi = combos[val][0],
                                                                   lambda_g = combos[val][1],
                                                                   gamma = gamma, K = K, max_iter = max_iter,
                                                                   update_thresh = update_thresh, epsilon = epsilon,
                                                                   standardize = standardize, std_type = std_type, std_x = std_x, std_y = std_y,
                                                                   verbose = verbose) for val in selection)
        
        for r in result[f]:
            cv_err_full = test_error(Y_test = dat_test_std['Y'], X_test = dat_test_std['X'], B = r['full']['B'],
                                     theta_dict = {'theta': r['full']['theta'],
                                                   'beta': r['full']['beta'],
                                                   'tau':r['full']['tau']})['comp_val']
            test_err_dict[''.join(str(r['full']['Lambda']))]['full'][f] = cv_err_full
            
            Xsub_test = [[dat_test_std['X'][d][s][:, r['include'][d].eq(1.)]  for s in range(S)] for d in range(D)]
            cv_err_sub = test_error(Y_test=dat_test_std['Y'], X_test=Xsub_test, B=r['subset']['B'],
                                    theta_dict = {'theta': r['subset']['theta'],
                                                  'beta': r['subset']['beta'],
                                                  'tau': r['subset']['tau']})['comp_val']
            test_err_dict[''.join(str(r['full']['Lambda']))]['subset'][f] = cv_err_sub
    
    full_fit = Parallel(n_jobs=ncore,prefer="processes",verbose=100, pre_dispatch='1.5*n_jobs')(delayed(optimize_ranking)(Y = Y, X = X, family = family, topn = topn,
                                                              lambda_xi = combos[val][0],
                                                              lambda_g = combos[val][1],
                                                              gamma = gamma, K = K, max_iter = max_iter,
                                                              update_thresh = update_thresh, epsilon = epsilon,
                                                              standardize = standardize, std_type = std_type, std_x = std_x, std_y = std_y,
                                                              verbose = verbose) for val in selection)
    
    for r in full_fit:
        r['cv_error'] = {'full': torch.nanmean(test_err_dict[''.join(str(r['full']['Lambda']))]['full']),
                         'subset': torch.nanmean(test_err_dict[''.join(str(r['full']['Lambda']))]['subset'])}
        r['fold_errors'] = test_err_dict[''.join(str(r['full']['Lambda']))]
    
    search_toc = time.perf_counter()
    
    return {'search_results': full_fit,
            'total_time': search_toc-search_tic,
            'xi_range': xi_range,
            'g_range': g_range,
            'num_steps': num_steps,
            'search': search
            }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# select_lambda: function to tune lambda parameter using some information criterion
# Required Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Y - matrix list - list of Y^s matrices containing outcomes
#   gamma - list<double> - indicators of whether to penalize each data view; should be length D
#   family - string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#   ncore - int - number of cores to use in parallel processing
#   topn - int or list<int> - number of variables to retain; different values may be specified for each view using a list of length D
# Optional Inputs:
#   K - int - number of latent components; will use 'select_K_simple' to choose if not provided
#   k_thresh - double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#   update_thresh - double - threshold to use for suboptimization convergence criterion; default = 10**-5
#   epsilon - double - threshold to use for overall convergence criterion; default = 10**-4
#   max_iter - int - maximum number of outer loop iterations; default = 50
#   search - string - what type of search to perform for lambda parameter; default = "random"
#       - "random" - tries a random selection of rand_prop*num_steps lambda values in grid
#       - "grid" - tries all lambda values in grid
#   xi_range - list<double> - minimum  and maximum values to consider for lambda_xi; default = [0.0, 2.0]
#   g_range - list<double> - minimum  and maximum values to consider for lambda_g; default = [0.0, 2.0]
#   num_steps - dict - number of steps to use in lambda grid
#       - 'Xi' - int - number of steps for lambda_xi; default = 8
#       - 'G' - int - number of steps for lambda_g; default = 8
#   rand_prop - double - proportion of grid points to search when search = "random"; must be > 0 and < 1; default = 0.20
#   standardize - string - One of "all", "subgroup", or "none"; default = "subgroup"
#   std_type - string - One of "scale_center", "center", or "norm"; default = "scale_center"
#   std_x - boolean - indicates whether to standardize X; default = True
#   std_y - boolean - indicates whether to standardize Y; default = True
#   verbose - boolean - whether to print additional information during optimization; default = False
# Outputs:
#   dict with the following elements:
#       search_results - list - list with a dict of results returned from optimize_torch for each lambda value tried
#       total_time - double - time to complete entire search in seconds
#       xi_range - list<double> - minimum and maximum values considered for lambda_xi
#       g_range - list<double> - minimum and maximum values considered for lambda_g
#       num_steps - dict - number of steps used in lambda grid
#       search - string - type of search performed for selecting lambda parameters
def select_lambda(Y, X, gamma, family, ncore, topn,
                  K = None, k_thresh = 0.2,
                  update_thresh = 10**-5, epsilon = 10**-4, max_iter = 50,
                  search = 'random', xi_range = [0.0, 2.0], g_range = [0.0, 2.0], num_steps = {'Xi': 8, 'G': 8}, rand_prop = 0.20,
                  standardize = 'subgroup', std_type = 'scale_center', std_x = True, std_y = True, verbose = False):
    
    # Run basic data validation checks - will throw error if any check fails
    validate_data(X = X, Y = Y, gamma = gamma, xi_range = xi_range, g_range = g_range, rand_prop = rand_prop)
    
    #Get n, p, d, s from the X provided from user
    D = len(X)
    S = len(X[0])
    n = [X[0][s].shape[0] for s in range(S)]
    p = [X[d][0].shape[1] for d in range(D)]
    q = Y[0].shape[1]
    
    if verbose:
        print("D =", D)
        print("S =", S)
        print("n =", n)
        print("p =", p)
        print("q =", q)
    
    # define test error function based on family.
    global test_error
    
    if family == 'gaussian':
        test_error = test_mse
    elif family == 'multiclass':
        test_error = test_class
        std_y = False #should not standardize Y ever when multiclass outcome.
    elif family == 'poisson':
        test_error = test_pois
        std_y = False #should not standardize Y ever when poisson outcome.
    elif family == 'zip':
        test_error = test_zip
        std_y = False #should not standardize Y ever when zip outcome.
    else:
        raise Exception("Invalid family specification.")
    
    search_tic = time.perf_counter()
    
    if K == None:
        K = select_K_simple(X=X, threshold=k_thresh, verbose=True)

    # Search over Xi; fix lam_g = 1.0
    # Allow for fixing one of the lambda values (i.e., upper  and lower are the same)
    test = []
    #  Xi values.
    if xi_range[0] == xi_range[1]:
        test.append([xi_range[0]])
    else:
        test.append([j*(xi_range[1]-xi_range[0])/(num_steps['Xi']) + xi_range[0] for j in  range(0, num_steps['Xi']+1)][1:])
    #  G values.
    if g_range[0] == g_range[1]:
        test.append([g_range[0]])
    else:
        test.append([j*(g_range[1]-g_range[0])/(num_steps['G']) + g_range[0] for j in  range(0, num_steps['G']+1)][1:])
    
    # All possible combinations of lambda datasets
    combos = list(itertools.product(*test))
    
    # Random or Grid search
    # NOTE: because of this seed, the random search will select the same lambda combinations each time
    if search == 'random':
        random.seed(7)
        selection = random.sample(range(len(combos)), math.ceil(rand_prop*(len(combos))))
    elif search == 'grid':
        selection = range(len(combos))
        
    result = Parallel(n_jobs=ncore,prefer="processes",verbose=100, pre_dispatch='1.5*n_jobs')(delayed(optimize_ranking)(Y = Y, X = X, family = family, topn = topn,
                                                              lambda_xi = combos[val][0],
                                                              lambda_g = combos[val][1],
                                                              gamma = gamma, K = K, max_iter = max_iter,
                                                              update_thresh = update_thresh, epsilon = epsilon,
                                                              standardize = standardize, std_type = std_type, std_x = std_x, std_y = std_y,
                                                              verbose = verbose) for val in selection)
    
    search_toc = time.perf_counter()
    
    return {'search_results': result,
            'total_time': search_toc-search_tic,
            'xi_range': xi_range,
            'g_range': g_range,
            'num_steps': num_steps,
            'search': search
            }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# optimize_ranking: fit model with all variables, select topn, refit with those topn variables only
# Required Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Y - matrix list - list of Y^s matrices containing outcomes
#   lambda_xi - double - value of lambda_xi in penalty term
#   lambda_g - double - value of lambda_g in penalty term
#   gamma - list<double> - indicators of whether to penalize each data view; should be length D
#   family - string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#   topn - int or list<int> - number of variables to retain; different values may be specified for each view using a list of length D
# Optional Inputs:
#   K - int - number of latent components; will use 'select_K_simple' to choose if not provided
#   k_thresh - double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#   update_thresh - double - threshold to use for suboptimization convergence criterion; default = 10**-5
#   epsilon - double - threshold to use for overall convergence criterion; default = 10**-4
#   max_iter - int - maximum number of outer loop iterations; default = 50
#   standardize - string - One of "all", "subgroup", or "none"; default = "subgroup"
#   std_type - string - One of "scale_center", "center", or "norm"; default = "scale_center"
#   std_x - boolean - indicates whether to standardize X; default = True
#   std_y - boolean - indicates whether to standardize Y; default = True
#   verbose - boolean - whether to print additional information during optimization
# Outputs:
#   dict with the following elements:
#       full - dict - results returned from optimize_torch on full data
#       include - list<tensor> - list of length D with a 1 indicating the variable was included in subset fit and 0 indicating not included in subset fit
#       subset - dict - results returned from optimize_torch on subset of variables 
def optimize_ranking(Y, X, lambda_xi, lambda_g, gamma, family, topn,
                     K = None, k_thresh = .2,
                     update_thresh = 10**-5, epsilon = 10**-4, max_iter = 50,
                     standardize = 'subgroup', std_type = 'scale_center', std_x = True, std_y = True, verbose = False):
                     
    D = len(X)
    S = len(X[0])
    p = [X[d][0].shape[1] for d in range(D)]
    n = [X[0][s].shape[0] for s in range(S)]
    
    # If user gives single integer, select same number of variables for each view
    if type(topn) == int:
        topn = [topn for d in range(D)]
    else:
        # if topn is a vector, check that same length as D
        # NOTE: will be an error if topn entries are not integers.
        if len(topn) != D:
            raise Exception("topn must be a single integer or a list of length D containing integers")
    
    # Fit full alg with given  lambda values
    opt_res = optimize_torch(Y = Y, X = X, family = family, K = K,
                             lambda_xi = lambda_xi, lambda_g = lambda_g, gamma = gamma,
                             update_thresh = update_thresh, epsilon = epsilon, max_iter = max_iter,
                             verbose = verbose, standardize = standardize, std_type = std_type, std_x = std_x, std_y = std_y)
                             
    # calculate tau(S_j) for eBIC
    # use log properties to avoid overflow: log(ab) = log(a) + log(b)
    tauj = 0.0
    for d in range(D):
        tauj_d = 0
        for w in range(topn[d], S*topn[d] + 1):
            tauj_d += special.comb(N = p[d], k = w)
        tauj += math.log(tauj_d)
        
    # Find top X loadings
    opt_res['top'] = top_loadings(B = opt_res['B'], top_n = topn, print = False, plot = None)
    # Keep the variables in the top 50 for at least 1 subgroup
    include = [torch.zeros(p[d]) for d in range(D)]
    for d in range(D):
        for s in range(S):
            include[d][opt_res['top']['selected'][d][s].eq(1)] = 1.
    Xsub = [[X[d][s][:, include[d].eq(1.)]  for s in range(S)] for d in range(D)]
    
    # Extended BIC (Chen & Chen, 2008)
    # -2*ll(theta_s) + nu(s) log(n) + 2*gamma*log(tau(S_j))
    #  pred is the -ll already, so just multiply by 2
    #  tau(S_j) is the cardinality of models with size j
    ebic = 0
    for s in range(S):
        ebic += math.log(n[s])*sum([sum(include[d]) for d in range(D)]).item()
        
    opt_res['eBIC_0'] = 2*opt_res['pred'] + ebic
    opt_res['eBIC_5'] = opt_res['eBIC_0'] + tauj # no additional log needed here because already in tauj
    opt_res['eBIC_1'] = opt_res['eBIC_0'] + 2*tauj # no additional log needed here because already in tauj
    
    opt_sub = optimize_torch(Y = Y, X = Xsub, family = family, K = K,
                             lambda_xi = 0.0, lambda_g = 0.0, gamma = [0.0 for d in range(D)],
                             update_thresh = update_thresh, epsilon = epsilon,
                             verbose = verbose, standardize = standardize, std_type = std_type, std_x = std_x, std_y = std_y,
                             max_iter = max_iter)
                             
    opt_sub['eBIC_0'] = 2*opt_sub['pred'] + ebic
    opt_sub['eBIC_5'] = opt_sub['eBIC_0'] + tauj # no additional log needed here because already in tauj
    opt_sub['eBIC_1'] = opt_sub['eBIC_0'] + 2*tauj # no additional log needed here because already in tauj
    
    return {'full':opt_res, 'include':include, 'subset':opt_sub}


#----------------------------------------------------------------------------------
# optimize_torch: Main optimization function for continuous outcomes
# Required Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Y - matrix list - list of Y^s matrices containing outcomes
#   lambda_xi - double - value of lambda_xi in penalty term
#   lambda_g - double - value of lambda_g in penalty term
#   gamma - list<double> - indicators of whether to penalize each data view; should be length D
#   family - string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
# Optional Inputs:
#   K - int - number of latent components; will use 'select_K_simple' to choose if not provided
#   k_thresh - double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#   update_thresh - double - criteria for convergence in Z, G, Xi, and theta optimization functions; default = 10**-5
#   epsilon - double - criteria for outer loop convergence; default = 10**-4
#   max_iter - int - maximum number of outer loop iterations allowed; default = 50
#   standardize - string - One of "all", "subgroup", or "none"; default = "subgroup"
#   std_type - string - One of "scale_center", "center", or "norm"; default = "scale_center"
#   std_x - boolean - indicates whether to standardize X; default = True
#   std_y - boolean - indicates whether to standardize Y; default = True
#   verbose - boolean - indicates whether to print additional info during run; default = False
# Outputs:
#   dict with the possible following elements based on status:
#       theta - tensor<double> - estimate of theta
#       beta - tensor<double> - estimate of beta
#       B - list<tensor<double>> - estimates of each B^d,s
#       G - list<tensor<double>> - estimates of each G^d
#       Xi - list<tensor<double>> - estimates of each Xi^d,s
#       Z - list<tensor<double>> - estimates of each Z^s
#       Lambda - tuple - values of lambda_xi and lambda_g used
#       BIC - double - calculated BIC
#       AIC - double - calculated AIC
#       pred - double - prediction loss evaluated at final estimates
#       train_err - dict - dict returned from function to calculate training error
#       message - string - message with the status of the result;
#           - "Converged" indicates the algorithm converged successfully
#           - "MAX ITERS" indicates the algorithm reached max_iter without converging
#       paths - dict - history of losses until convergence
#       iter - int - number of iterations to converge
#       iter_time - double - time to find solution in seconds
#       conv_criterion - double -  value of last convergence criterion
#       std_x - boolean - whether X was standardized
#       std_y - boolean - whether Y was standardized
def optimize_torch(Y, X, lambda_xi, lambda_g, gamma, family,
                   K = None, k_thresh = .2,
                   update_thresh = 10**-5, epsilon = 10**-4, max_iter = 50,
                   standardize = 'subgroup', std_type = 'scale_center', std_x = True, std_y = True, verbose = False):
    
    # torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)
    
    # Get n, p, D, S from the X  provided from user
    D = len(X)
    S = len(X[0])
    n = [X[0][s].shape[0] for s in range(S)]
    p = [X[d][0].shape[1] for d in range(D)]
    N = sum(n)
    
    if family == "poisson" or family == "zip":
        q = 1
    else:
        q = Y[0].shape[1]

    if verbose:
        print("D =", D)
        print("S =", S)
        print("n =", n)
        print("p =", p)
        if family == "multiclass":
            print("m =", q)
        else:
            print("q =", q)
    
    # must define as global to allow optimization functions to access
    global pred_loss
    global calc_err
    
    # Assign appropriate functions for the given family
    # Also ensure q defined correctly (so that beta_new is also defined correctly)
    if family == 'gaussian':
        pred_loss = pred_gauss
        calc_error = train_mse
    
    elif family == 'multiclass':
        pred_loss = pred_class
        calc_error = train_class
        std_y = False # should never standardize Y when multiclass
    
    elif family == 'poisson':
        pred_loss = pred_pois
        calc_error = train_pois
        std_y = False # should never standardize Y when poisson
        # if not offset, add column of 1s as offset
        if Y[0].shape[1] == 1:
            Y = [torch.cat((Y[s], torch.ones_like(Y[s])), dim = 1) for s in range(S)] 
            
    elif family == 'zip':
        pred_loss = pred_zip
        calc_error = train_zip
        std_y = False # should never standardize Y when ZIP
        # if not offset, add column of 1s as offset
        if Y[0].shape[1] == 1:
            Y = [torch.cat((Y[s], torch.ones_like(Y[s])), dim = 1) for s in range(S)] 
    
    else:
        raise Exception("Invalid family specification in optimize_torch.")
    
    tic = time.perf_counter()
    paths = {'assoc': [],
             'pred': [],
             'total_loss': [],
             'obj': []
            }
    
    # Standardize data
    dat_std = standardize_dat(standardize = standardize, std_type = std_type, X = X, Y = Y, std_x = std_x, std_y = std_y)
    X_train = dat_std['X']
    Y_train = dat_std['Y']
    
    # Run K selection on standardized data if no K specified.
    if K == None:
        K = select_K_simple(X=X_train, threshold=k_thresh, verbose=verbose)['kchoose']

    # Initialize matrices for algorithm
    torch.manual_seed(7)
    Z_new = [torch.DoubleTensor(n[s], K).uniform_(0.9, 1.1) for s in range(S)]
    theta_new = torch.ones((K, q))
    beta_new = torch.ones((1, q))
    # tau only needed for ZIP
    theta_dict = {'theta': theta_new,
                  'beta': beta_new,
                  'tau': None}
    if family == 'zip':
        theta_dict['tau'] = init_tau(Y=Y_train, Z=Z_new, theta=theta_new, beta=beta_new)
    
    G_new = [torch.ones((p[d], K)) for d in range(D)]
    Xi_new = [[torch.matmul(torch.inverse(torch.matmul(Z_new[s].t(), Z_new[s])), torch.matmul(Z_new[s].t(), X_train[d][s])).t() for s in range(S)] for d in range(D)]

    lambda_xi_gam = [gamma[d]*lambda_xi for d in range(D)]
    lambda_g_gam = [gamma[d]*lambda_g for d in range(D)]
    
    pred = pred_loss(Y=Y_train, Z=Z_new, theta_dict=theta_dict)
    assoc = assoc_loss(X=X_train, Z=Z_new, G = G_new, Xi = Xi_new)
    total = assoc + pred
    obj = total + penalty(G = G_new,  Xi = Xi_new, lam_g = lambda_g_gam, lam_xi = lambda_xi_gam)

    # Initial loss values
    paths['pred'].append(pred)
    paths['assoc'].append(assoc)
    paths['total_loss'].append(total)
    paths['obj'].append(obj)
    
    selected = {'Xi': [[[] for s in range(S)] for d in range(D)],
                'G': [[] for d in range(D)],
                'B': [[[] for s in range(S)] for d in range(D)]
                }
    
    if verbose:
        print("Lambda: [", lambda_xi, ", ", lambda_g, "]")
        print("G - L2,1 norm", [torch.pow(G_new[d], 2).sum(dim = 1).sqrt().sum() for d in range(D)])
        print("Xi - L2,1 norm", [[torch.pow(Xi_new[d][s], 2).sum(dim = 1).sqrt().sum() for s in range(S)] for d in range(D)])
        print("Initial loss values")
        print("Pred Loss:", pred)
        print("Assoc Loss:", assoc)
        print("Total Loss:", total)
        print("Objective:", obj)
    
    hit_max = True

    for i in range(1, max_iter):
        print("Iteration", i)
        
        # using functions to solve Z so converge fully
        for s in range(S):
            Z_obj = fista_z(Y = Y_train[s],
                            X = [X_train[d][s] for d in range(D)],
                            Z = Z_new[s],
                            G = G_new,
                            Xi = [Xi_new[d][s] for d in range(D)],
                            theta_dict=theta_dict,
                            update_thresh = update_thresh,
                            verbose = verbose)
                            
            Z_temp = Z_obj['val'].clone().detach()
            zmean = torch.mean(Z_temp, dim = 0)
            zsd = torch.std(Z_temp, dim = 0)
            Z_new[s] = torch.div(Z_temp-zmean, zsd)

        # Xi and G Optimization
        for d in range(D):
            G_temp = opt_g(X = X_train[d],
                             Z = Z_new,
                             G = G_new[d],
                             Xi = Xi_new[d],
                             lam_g = lambda_g_gam[d],
                             update_thresh = update_thresh,
                             verbose = verbose)
            
            # Update G^d before updates to Xi
            G_new[d] = G_temp['val'].clone().detach()

            for s in range(S):
                Xi_temp = opt_xi(X = X_train[d][s],
                                   Z = Z_new[s],
                                   G = G_new[d],
                                   Xi = Xi_new[d][s],
                                   lam_xi = lambda_xi_gam[d],
                                   update_thresh = update_thresh,
                                   verbose = verbose)
                
                Xi_new[d][s] = Xi_temp['val'].clone().detach()
            
        B_new = [[Xi_new[d][s]*G_new[d] for s in range(S)] for d in range(D)]
        
        # using functions to solve theta/beta so converge fully
        temp = ista_theta(Y = Y_train,
                          Z = Z_new,
                          theta_dict=theta_dict,
                          update_thresh = update_thresh,
                          verbose = verbose)
        theta_dict = temp['val']
        
        if theta_dict['tau'] != None:
            theta_dict['tau'] = init_tau(Y=Y_train, Z=Z_new, theta=theta_dict['theta'], beta=theta_dict['beta'])

        # Check for convergence
        with torch.no_grad():
            
            assoc = assoc_loss(X=X_train, Z=Z_new, G=G_new, Xi=Xi_new)
            pred = pred_loss(Y=Y_train, Z=Z_new, theta_dict=theta_dict) #theta=theta_new, beta=beta_new)
            tot_pen = penalty(G = G_new, Xi = Xi_new, lam_g = lambda_g_gam, lam_xi = lambda_xi_gam)
            
            paths['assoc'].append(assoc)
            paths['pred'].append(pred)
            paths['total_loss'].append(assoc+pred)
            paths['obj'].append(assoc+pred+tot_pen)
            
            rel_loss = abs(paths['total_loss'][i] - paths['total_loss'][i-1])/abs(paths['total_loss'][i-1])
        
            if verbose:
                print("Convergence calculations")
                print("Pred Loss:", paths['pred'][i])
                print("Assoc Loss:", paths['assoc'][i])
                print("Total Loss:", paths['total_loss'][i])
                print("Relative Change in Loss", rel_loss)
                print("Objective:", paths['obj'][i])
           
            if i > 1 and rel_loss < epsilon:
                print("Stopped at iteration", i)
                print("Convergence status: Converged")
                message = 'Converged'
                hit_max = False
                break
    
    if hit_max:
        print("Convergence status: MAX ITERS")
        message = 'MAX ITERS'
    
    # Model selection info
    pred = pred_loss(Y = Y_train, Z = Z_new, theta_dict=theta_dict)
    assoc = assoc_loss(X = X_train, Z = Z_new, G = G_new, Xi = Xi_new)
    
    # no variables will be exactly 0 in the ranking
    num_selected = sum(p)*S
    
    # Updated version from Sandra
    BIC = N*math.log((pred + assoc)/sum(n)) + math.log(N)*num_selected
    AIC = N*math.log((pred + assoc)/sum(n)) + 2.0*num_selected
                
    # calculate training mse for each column, and overall
    err = calc_error(Y = Y_train, Z = Z_new, theta_dict=theta_dict)
    toc = time.perf_counter()
                  
    if verbose:
        print("The algorithm completed in", toc-tic, "seconds.")
        print('Train Error =', err['comp_val'])
        print('Summary of Variable Selection - B')
        common = summary_B(B_new)
        for d in range(D):
            print("Data View", d)
            print(common[d])
                    
        plot_paths(paths)
    
    return {'theta': theta_dict['theta'],
            'beta': theta_dict['beta'],
            'tau': theta_dict['tau'],
            'B':B_new,
            'G':G_new,
            'Xi':Xi_new,
            'Z':Z_new,
            'Lambda':(lambda_xi, lambda_g),
            'BIC': BIC,
            'AIC': AIC,
            'pred': pred,
            'train_err': err,
            'message': message,
            'paths': paths,
            'iter': i,
            'conv_criterion': rel_loss,
            'iter_time': toc-tic,
            'std_x': std_x,
            'std_y': std_y
            }
    
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# train_mse: calculate the training MSE on fitted model
# Required inputs:
#   Y - list<tensor<double>> - list of outcomes
#   Z - list<tensor<double>> - estimates of Z^s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       each - tensor<double> - tensor of MSEs for each of the q outcomes
#       comp_val - tensor<double> - MSE calculated across all q outcomes
#       pred - list<tensor<double>> - predicted values from fitted model
def train_mse(Y, Z, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    q = Y[0].shape[1]
    S = len(Y)

    Y_pred = [beta + torch.matmul(Z[s], theta) for s in range(S)]
    
    mydiff = torch.cat(Y_pred) - torch.cat(Y)
    mse_all = torch.mean(mydiff**2)
    mse_each = torch.mean((mydiff)**2, dim = 0)
    
    return {'each': mse_each, 
            'comp_val': mse_all,
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_mse: calculate the test MSE using fitted model
# Required inputs:
#   Y_test - list<tensor<double>> - outcomes from test data
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       each - tensor - tensor of MSEs for each of the q outcomes
#       all - tensor - MSE calculated across all q outcomes
#       pred - list of tensors - predicted values from fitted model
def test_mse(Y_test, X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    D = len(X_test)
    S = len(X_test[0])
    q = Y_test[0].shape[1]
    
    
    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)

        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = beta + torch.matmul(Z_pred[s], theta)
    
    mydiff = torch.cat(Y_pred) - torch.cat(Y_test)
    mse_all = torch.mean(mydiff**2)
    mse_each = torch.mean((mydiff)**2, dim = 0)

    return {'each': mse_each, 
            'comp_val': mse_all,
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_mse: make predictions from given X values for continuous outcome
# Required inputs:
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   Y_pred = list<tensor<double>> - predicted values from fitted model

def pred_mse(X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    D = len(X_test)
    S = len(X_test[0])
    
    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)

        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = beta + torch.matmul(Z_pred[s], theta)
    
    return(Y_pred)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# train_class: Calculates training classification accuracy for multiclass outcome
# Required inputs:
#   Y - list<tensor<double>> - list of outcomes
#   Z - list<tensor<double>> - estimates of Z^s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - classification accuracy
#       pred - list<tensor<double>> - predicted values from fitted model; single integer to represent class, starting with 0
def train_class(Y, Z, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    S = len(Z)
    Y_pred = [calc_probs(beta + Z[s].matmul(theta)).max(dim = 1, keepdim = True).indices for s in range(S)]
    Y_true = torch.cat(Y, dim = 0).max(dim = 1, keepdim = True).indices
    
    return {'comp_val': torch.sum(Y_true == torch.cat(Y_pred))/Y_true.shape[0],
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_class: Calculates test classification accuracy for multiclass outcome
# Required inputs:
#   Y_test - list<tensor<double>> - outcomes from test data
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - classification accuracy
#       pred - list<tensor<double>> - predicted classes of test data; single integer to represent class, starting with 0
def test_class(Y_test, X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = calc_probs(beta + torch.matmul(Z_pred[s], theta)).max(dim = 1, keepdim = True).indices
    
    Y_true = torch.cat(Y_test, dim = 0).max(dim = 1, keepdim = True).indices

    return {'comp_val': torch.sum(Y_true == torch.cat(Y_pred))/Y_true.shape[0],
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_class: make predictions from given X values for multiclass outcome
# Required inputs:
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   Y_pred = list<tensor<double>> - predicted classes of test data; single integer to represent class, starting with 0

def pred_multiclass(X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = calc_probs(beta + torch.matmul(Z_pred[s], theta)).max(dim = 1, keepdim = True).indices
        
    return(Y_pred)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# train_pois: returns deviance info and predicted means on training data
# Required inputs:
#   Y - list<tensor<double>> - list of outcomes
#   Z - list<tensor<double>> - estimates of Z^s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - fraction of deviance explained using training data
#       dev_sum - dict with more detailed deviance information
#       pred - list<tensor<double>> - predicted means from fitted model
def train_pois(Y, Z, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    S = len(Z)
    Y_pred = [torch.exp(beta + torch.matmul(Z[s], theta)) for s in range(S)]
    
    dev = deviance_pois(Y = torch.cat(Y, dim = 0), Y_pred = torch.cat(Y_pred, dim = 0))
    
    # D^2 = [D_null - D_opt]/[D_null]
    return {'comp_val': 1 - (dev['dev'].item()/dev['null_dev'].item()),
            'dev_sum': dev,
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_pois: returns deviance info and predicted means on test data
# Required inputs:
#   Y_test - list<tensor<double>> - outcomes from test data
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - fraction of deviance explained using training data
#       dev_sum - dict with more detailed deviance information
#       pred - list<tensor<double>> - predicted means from fitted model
def test_pois(Y_test, X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = torch.exp(beta + torch.matmul(Z_pred[s], theta))
    
    dev = deviance_pois(Y = torch.cat(Y_test, dim = 0), Y_pred = torch.cat(Y_pred, dim = 0))
    
    return {'comp_val': 1 - (dev['dev'].item()/dev['null_dev'].item()),
            'dev_sum': dev,
            'pred': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_pois: make predictions from given X values for Poisson outcome
# Required inputs:
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor - estimate of theta from fitted model
#       beta - tensor - estimate of beta from fitted model
# Outputs:
#   Y_pred = list<tensor<double>> - predicted means from fitted model

def pred_poisson(X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = torch.exp(beta + torch.matmul(Z_pred[s], theta))
        
    return(Y_pred)
#----------------------------------------------------------------------------------

#### Zero-Inflated Poisson

#----------------------------------------------------------------------------------
# train_zip: returns deviance info and predicted means on training data
# Required inputs:
#   Y - list<tensor<double>> - list of outcomes
#   Z - list<tensor<double>> - estimates of Z^s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
#       tau - tensor<double> - estimate of tau from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - fraction of deviance explained using training data
#       dev_sum - dict with more detailed deviance information
#       pred - list<tensor<double>> - predicted means from fitted model
def train_zip(Y, Z, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    tau = theta_dict['tau']

    S = len(Z)
    Y_pred = [torch.exp(beta + torch.matmul(Z[s], theta)) for s in range(S)]
    dev = deviance_zip(Y = torch.cat(Y, dim = 0), Y_pred = torch.cat(Y_pred, dim = 0), tau=tau)
    
    # D^2 = [D_null - D_opt]/[D_null]
    return {'comp_val': 1 - (dev['dev'].item()/dev['null_dev'].item()),
            'dev_sum': dev,
            'pred': [(1-tau)*Y_pred[s] for s in range(S)],
            'pred_notau': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_zip: returns deviance info and predicted means on test data
# Required inputs:
#   Y_test - list<tensor<double>> - outcomes from test data
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
#       tau - tensor<double> - estimate of tau from fitted model
# Outputs:
#   dict with following elements
#       comp_val - tensor<double> - fraction of deviance explained using training data
#       dev_sum - dict with more detailed deviance information
#       pred - list<tensor<double>> - predicted means from fitted model
def test_zip(Y_test, X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    tau = theta_dict['tau']
    
    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = torch.exp(beta + torch.matmul(Z_pred[s], theta))
    
    dev = deviance_zip(Y = torch.cat(Y_test, dim = 0), Y_pred = torch.cat(Y_pred, dim = 0), tau=tau)
    
    # D^2 = [D_null - D_opt]/[D_null]
    return {'comp_val': 1 - (dev['dev'].item()/dev['null_dev'].item()),
            'dev_sum': dev,
            'pred': [(1-tau)*Y_pred[s] for s in range(S)], 
            'pred_notau': Y_pred
           }
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_zip: make predictions from given X values for ZIP outcome
# Required inputs:
#   X_test - list<tensor<double>> - covariates from test data
#   B - list<tensor<double>> - estimates of B^d,s from fitted model
#   theta_dict - dict with following elements
#       theta - tensor<double> - estimate of theta from fitted model
#       beta - tensor<double> - estimate of beta from fitted model
#       tau - tensor<double> - estimate of tau from fitted model
# Outputs:
#   Y_pred = list<tensor<double>> - predicted means from fitted model

def pred_zero_poisson(X_test, B, theta_dict):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    tau = theta_dict['tau']
    
    D = len(X_test)
    S = len(X_test[0])

    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    
    for s in range(S):
        Xcat = torch.cat([X_test[d][s] for d in range(D)], dim=1)
        Bcat = torch.cat([B[d][s] for d in range(D)], dim=0)
        
        Z_pred[s] = torch.matmul(Xcat, torch.matmul(Bcat, torch.inverse(torch.matmul(torch.t(Bcat), Bcat) + (0.0001*torch.eye(Bcat.shape[1])))))
        Y_pred[s] = torch.exp(beta + torch.matmul(Z_pred[s], theta))
        
    return(Y_pred)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# assoc_loss: calculates the association term loss (no penalty)
def assoc_loss(X, Z, G, Xi, type = 'all'):
    
    # sum across all D, S
    if type == 'all':
        D = len(X)
        S = len(X[0])
        n = [len(X[0][s]) for s in range(S)]
        N = sum(n)
            
        loss = 0
        for d in range(D):
            for s in range(S):
                loss += torch.pow(X[d][s] - torch.matmul(Z[s], (G[d]*Xi[d][s]).t()), 2).sum()
        
    # sum across S for fixed d - called in G optimizationss
    elif type == 'view':
        S = len(X)
        loss = 0
        for s in range(S):
            loss += torch.pow(X[s] - torch.matmul(Z[s], (G*Xi[s]).t()), 2).sum()
    
    # for a single X^{d,s} - called in Xi optimizations
    elif type == 'sub':
        D = len(X)
        loss = 0
        for d in range(D):
            loss += torch.pow(X[d] - torch.matmul(Z, (G[d]*Xi[d]).t()), 2).sum()
    
    # for a single X^{d,s} - called in Xi optimizations
    elif type == 'view_sub':
        loss = torch.pow(X - torch.matmul(Z, (G*Xi).t()), 2).sum()
    
    else:
        raise Exception("Invalid type specification in assoc_loss call.")
        
    return loss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_gauss: calculates the prediction term loss for gaussian outcome
def pred_gauss(Y, Z, theta_dict, single = False):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    
    # will be called by loss_z for single subgroup
    if single:
        loss = torch.pow(Y - (beta + torch.matmul(Z, theta)), 2).sum()
    
    # to calculate across all subgroups
    else:
        S = len(Z)
        
        loss = 0
        for s in range(S):
            loss += torch.pow(Y[s] - (beta + torch.matmul(Z[s], theta)), 2).sum()
        
    return loss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_class: Calculates the loss for prediction term with multiclass outcome
# Y is expected to be in multi-column format
def pred_class(Y, Z, theta_dict, single = False):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    if single:
        Prob_mat = calc_probs(beta + Z.matmul(theta))
        loss = torch.sum(Y*torch.log(Prob_mat))
    
    else:
        S = len(Z)
        
        loss = 0
        for s in range(S):
            Prob_mat = calc_probs(beta + Z[s].matmul(theta))
            loss += torch.sum(Y[s]*torch.log(Prob_mat))

    return -1*loss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_pois: calculates the prediction term loss for Poisson outcome
#  Y is expected to be N x 2 with first column = outcome value and second column = offset
# NOTE: not tested using offset other than 1
def pred_pois(Y, Z, theta_dict, single = False):
    theta = theta_dict['theta']
    beta = theta_dict['beta']

    if single:
        factorial_tensor = torch.tensor(special.factorial(Y[:,0].reshape(-1,1)))
        loss = torch.sum(-1*Y[:,0].reshape(-1,1)*(torch.log(Y[:,1]).reshape(-1,1) + (beta + Z.matmul(theta))) + Y[:,1].reshape(-1,1)*torch.exp(beta + Z.matmul(theta)) + torch.log(factorial_tensor))
    
    else:
        loss = 0
        S = len(Y)
        for s in range(S):
            factorial_tensor = torch.tensor(special.factorial(Y[s][:,0].reshape(-1,1)))
            loss += torch.sum(-1*Y[s][:,0].reshape(-1,1)*(torch.log(Y[s][:,1]).reshape(-1,1) + (beta + Z[s].matmul(theta))) + Y[s][:,1].reshape(-1, 1)*torch.exp(beta + Z[s].matmul(theta)) + torch.log(factorial_tensor))
    # already accounts for -1, so just return loss
    return loss
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# pred_zip: calculates the prediction term loss for Zero-Inflated Poisson outcome
#  Y is expected to be N x 2 with first column = outcome value and second column = offset
#  tau is an additional parameter relating to the probability of being in zero state
# NOTE: not tested using offset other than 1
def pred_zip(Y, Z, theta_dict, single = False):
    theta = theta_dict['theta']
    beta = theta_dict['beta']
    tau = theta_dict['tau']

    if single:
        ll = ll_zip(Y = Y, Y_pred = torch.exp(beta + Z.matmul(theta)), tau = tau)
        
    
    else:
        ll = 0
        S = len(Y)
        for s in range(S):
            ll += ll_zip(Y = Y[s], Y_pred = torch.exp(beta + Z[s].matmul(theta)), tau = tau)

    return -1*ll
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# penalty: calculates the total penalty for G and Xi
# lambdas should include the gamma indicator (i.e., should be length sD)
def penalty(G, Xi, lam_g, lam_xi):
    D = len(Xi)
    S = len(Xi[0])
    
    tot = 0
    for d in range(D):
        tot += lam_g[d]*torch.pow(G[d], 2).sum(dim = 1).sqrt().sum()
        for s in range(S):
            tot += lam_xi[d]*torch.pow(Xi[d][s], 2).sum(dim = 1).sqrt().sum()
    
    return tot
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# lam_B: returns the number of non_zero rows in a matrix
def lam_B(mat):
    return sum(torch.sum(torch.abs(mat), 1) !=0)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# summary_B(B): give number of common and subgroup specific variables for each view
# Diagonals give the number of variables selected in that subgroup
# Off-diagonals give the number of variables that overlap between subgroup i and j
def summary_B(B):
    D = len(B)
    S = len(B[0])
    selected = [[torch.sum(torch.abs(B[d][s]), 1) != 0 for s in range(S)] for d in range(D)]
    common = [np.zeros((S,S)) for d in range(D)]
    for d in range(D):
        for s in range(S):
            for s2 in range(s, S):
                match=0
                for j in range(B[d][0].shape[0]):
                    match += selected[d][s][j] & selected[d][s2][j]
                common[d][s][s2] = match
    return common
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# plot_paths: plot losses over outer loop iterations
def plot_paths(paths):
    
    plt.figure()
    plt.subplot(141)
    plt.plot(paths['pred'], color='red')
    plt.title('Pred Loss')

    plt.subplot(142)
    plt.plot(paths['assoc'], color='blue')
    plt.title('Assoc Loss')

    plt.subplot(143)
    plt.plot(paths['total_loss'], color='purple')
    plt.title('Total Loss')
                    
    plt.subplot(144)
    plt.plot(paths['obj'], color='green')
    plt.title('Objective')
    plt.show()
    
    return None
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# validate_data:
def validate_data(X, Y, gamma, xi_range, g_range, rand_prop):
    D = len(X)
    S = len(X[0])
    
    # Check that each subgroup has the same number of variables within a view
    for d in range(D):
        pd = [X[d][s].shape[1] for s in range(S)]
        if len(np.unique(pd)) != 1:
            raise Exception(' '.join(("Subgroups in view", str(d+1), "have differing numbers of variables.")))
    
    # Check that each data view has the same number of subjects within a subgroup
    for s in range(S):
        ns = [X[d][s].shape[0] for d in range(D)]
        if len(np.unique(ns)) != 1:
            raise Exception(' '.join(("Subgroup", str(s+1), "has differing numbers of observations across the data views.")))
    
    # Check that Y is a list, or there will be errors
    if type(Y) != list:
        raise Exception("Y must be a list.")
    # If Y is a list, check that it has S subgroups
    elif len(Y) != S:
        raise Exception("Y and X have differing numbers of subgroups.")
   
    # Check that gamma is length D
    if len(X) != len(gamma):
        raise Exception("Gamma must be length D.")
    
    # Check specification of xi_range
    if xi_range[0] < 0 or xi_range[1] < xi_range[0]:
        raise Exception("lambda_xi parameters must be non-negative, and the maximum must be greater than or equal to the minimum.")
    
    # Check specification of g_range
    if g_range[0] < 0 or g_range[1] < g_range[0]:
        raise Exception("lambda_g parameters must be non-negative, and the maximum must be greater than or equal to the minimum.")
    
    # Check 0 < rand_prop < 1
    if rand_prop <= 0 or rand_prop >= 1:
        raise Exception("rand_prop must be between 0 and 1 (non-inclusive).")
    
    # return value doesn't really matter
    return "All Checks Passed"
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# get_folds: return indices to use as test data in CV
# designed to be called from R, not Python; select_lambda_CV does not rely on this function
def get_folds(n, folds = 5):
    S = len(n)
    
    # set seed for reproducability
    random.seed(777)
    result = [[] for f in range(folds)]

    # Create set of indices for each fold
    rows = [list() for s in range(S)]
    mods = [n[s]%folds for s in range(S)]
    per_fold = [[n[s]//folds + 1 if f < mods[s] else n[s]//folds for f in range(folds)] for s in range(S)]
    
    for s in range(S):
        # have to start at one because calling from R
        shuffle = random.sample(range(1, n[s]+1), k = n[s])
        start = 0
        for f in range(folds):
            rows[s].append(shuffle[start:start+per_fold[s][f]])
            start += per_fold[s][f]
    
    return(rows)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
def get_true(p, nonzero, offset):
    
    mat = [
    # First View
    [torch.cat((torch.ones(nonzero), torch.zeros(p[0] - nonzero))), # Subgroup 1
     torch.cat((torch.zeros(offset), torch.ones(nonzero), torch.zeros(p[0] - nonzero - offset)))], # Subgroup 2
    # Second View
    [torch.cat((torch.ones(nonzero), torch.zeros(p[1] - nonzero))), # Subgroup 1
    torch.cat((torch.zeros(offset), torch.ones(nonzero), torch.zeros(p[1] - nonzero - offset)))] # Subgroup 2
    ]
    
    return mat
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
def calc_varselect(est_vec, truth):
    D = len(truth)
    S = len(truth[0])

    # tpr: true positives (i.e. both est and true are nonzero)/total actual true
    tpr = [[torch.logical_and(est_vec[d][s].ne(0), truth[d][s].eq(1)).sum()/truth[d][s].sum() for s in range(S)] for d in range(D)]
    
    # fpr: False positives (est > 0 and truth  == 0)/total number of true zeros
    fpr = [[torch.logical_and(est_vec[d][s].ne(0), truth[d][s].eq(0)).sum()/truth[d][s].eq(0).sum() for s in range(S)] for d in range(D)]
    
    # recall = tpr
    # precision: True Positive/Predicted positive
    prec = [[torch.logical_and(est_vec[d][s].ne(0), truth[d][s].eq(1)).sum()/est_vec[d][s].ne(0).sum() for s in range(S)] for d in range(D)]
    
    f1 = [[2/((1/tpr[d][s]) + (1/prec[d][s])) for s in range(S)] for d in range(D)]
    
    return {'tpr': tpr, 'fpr':fpr,  'f1': f1}
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# top_loadings: print variables with largest loadings
# now assuming top_n is a list of length D
def top_loadings(B, top_n, print = False, plot = None):
    D = len(B)
    S = len(B[0])
    p = [B[d][0].size()[0] for d in range(D)]
    if type(top_n) == int:
        top_n = [top_n for d in range(D)]
 
    B_vec = [[torch.pow(B[d][s], 2).sum(dim=1).sqrt() for s in range(S)] for d in range(D)]
    B_sel = [[torch.zeros(p[d]) for s in range(S)] for d in range(D)]
    val_list = [[torch.zeros(p[d]) for s in range(S)] for d in range(D)]
    order_list = [[torch.zeros(p[d]) for s in range(S)] for d in range(D)]
        
    for d in range(D):
        for s in range(S):
            val, order = torch.sort(B_vec[d][s], descending=True)
            val_list[d][s] = val
            order_list[d][s] = order
            B_sel[d][s][order[0:top_n[d]]] = 1
            
            if print:
                out_mat = torch.cat((order[0:top_n[d]].reshape(top_n[d], 1), B[d][s][order[0:top_n[d]], :]), dim=1)
                print("Var".ljust(5), "Loadings")
                for row in out_mat:
                    print(format(row[0], "n").ljust(5), format(row[1], "2.3f").rjust(7), format(row[2], "2.3f").rjust(7))
                print()
    
    if plot != None:
        nonzero = plot['nonzero']
        offset = plot['offset']
        nonzero_vec = [nonzero for d in range(D)]
        offset_vec = [offset for d in range(D)]
        y_max = []
        for d in range(D):
            for s in range(S):
                y_max.append(max(B_vec[d][s]))
        y_lim_upper = max(y_max)*1.05
    
        plt.figure(tight_layout=True, figsize=[10.0, 6.0])
        plt.suptitle("B Loadings")
        ax = [plt.axes(ylim=(0, y_lim_upper), xlim=(0, p[d] + 1), label = str(d)) for d in range(D)]
        for d in range(D):
            x_index = np.arange(start= 1, stop = p[d]+1, step = 1)
            x_color = np.array([1 if (x_i <= (nonzero - offset) or (x_i > nonzero and x_i <= nonzero+offset)) else 2 if (x_i <= nonzero and x_i >= offset) else 0 for x_i in x_index])
            for s in range(S):
                x_start = offset_vec[d]*s
                x_end = x_start + nonzero_vec[d] # - 1
                lab = "T:" + str(sum(B_sel[d][s][x_start:x_end] != 0).item())
                plt.subplot(D, S, (s*D)+d+1, title= 'D='+str(d)+" S="+str(s), sharey=ax[d], sharex=ax[d])
                # range of 0 to p[d]-1
                plt.plot(x_index[torch.eq(B_sel[d][s], 1)], B_vec[d][s][torch.eq(B_sel[d][s], 1)], marker="o", color = 'r', linestyle='', label=lab)
                plt.plot(x_index[torch.eq(B_sel[d][s], 0)], B_vec[d][s][torch.eq(B_sel[d][s], 0)], marker="x", color = 'gray', linestyle='')
                plt.fill_between(x_index, 0, y_lim_upper, where=x_color == 1, facecolor='yellow', alpha=0.5)
                plt.fill_between(x_index, 0, y_lim_upper, where=x_color == 2, facecolor='orange', alpha=0.5)
                plt.legend(markerscale = 0.0)
                plt.show()
    
    return {'values': val_list,
            'order': order_list,
            'selected': B_sel}
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# get_best: return best result
# pass in ['search_results'] list
def get_best(res, criterion, which = 'full'):
    l = len(res)
    vec = torch.empty(l)
    if criterion == 'cv_sub':
        for i in range(l):
            vec[i] = res[i]['cv_error']['subset']
    else:
        for i in range(l):
            vec[i] = res[i][which][criterion]
    index = vec.min(dim=0).indices
    
    return {'index': index, 'res': res[index]}
#----------------------------------------------------------------------------------

# ADDED FOR MULTICLASS

#----------------------------------------------------------------------------------
# calc_probs: calculates the probability of being in each of m classes;
#   returns a matrix of probabilities of being in each class for each observation
def calc_probs(W):

    W_exp = torch.exp(W)
    row_sums = torch.sum(W_exp, dim = 1, keepdim=True)
    probs = torch.true_divide(W_exp, row_sums)
    
    return probs
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# class_matrix: takes an  nx1 vector of integer classes and converts to an nxm indicator matrix where m is the number of unique classes
def class_matrix(Y, S):
    classes = [list() for s in range(S)]
    column = [list() for s in range(S)]
    if S > 1:
        for s in range(S):
            classes[s], column[s] = torch.unique(Y[s], return_inverse=True)
        nclass  = max(len(classes[s]) for s in range(S))
        indicator_mat = [torch.zeros((Y[s].shape[0], nclass)) for s in range(S)]
        for s in range(S):
            for i in range(len(Y[s])):
                indicator_mat[s][i][column[s][i]] = 1
    else:
        classes, column = torch.unique(Y, return_inverse=True)
        nclass = len(classes)
        indicator_mat = torch.zeros(Y.shape[0], nclass)
        for i in range(len(Y)):
            indicator_mat[i][column[i]] = 1

    return indicator_mat
#----------------------------------------------------------------------------------

# ADDED FOR POISSON

#----------------------------------------------------------------------------------
# ll_pois: poisson log-likelihood
# assume Ypred has already had exponential applied to it.
def ll_pois(Y, Ypred):
    factorial_tensor = torch.tensor(special.factorial(Y[:,0].reshape(-1,1)))
    return torch.sum(Y[:,0].reshape(-1,1)*(torch.log(Y[:,1]).reshape(-1,1) + torch.log(Ypred)) -
                     Y[:,1].reshape(-1,1)*Ypred - 
                     torch.log(factorial_tensor))
    
    # without offset
    #return(torch.sum(Y*torch.log(Ypred) - Ypred - torch.log(special.factorial(Y))))
#----------------------------------------------------------------------------------
    
#----------------------------------------------------------------------------------
# deviance: calculates the deviance
# Must concatenate outcomes before passing to this function.
# Y is still N x 2, but Y_pred is N x 1
def deviance_pois(Y, Y_pred):
   
    null_ll = ll_pois(Y, torch.mean(Y[:,0]))
    sat_ll = ll_pois(Y, Y[:,0].reshape(-1,1)+0.000001)
    model_ll = ll_pois(Y, Y_pred)
    
    return {'ll': model_ll,
            'null_ll': null_ll,
            'sat_ll': sat_ll,
            'dev': 2*(sat_ll - model_ll),
            'null_dev': 2*(sat_ll - null_ll)}
#----------------------------------------------------------------------------------


# ADDED FOR ZERO-INFLATED POISSON

#----------------------------------------------------------------------------------
# ll_zip: zip log-likelihood
# Y is the observed response
# Y_pred is exp(lp)
# lp is the linear predictor beta + Z^s theta
# tau is P(Y_i = 0)
def ll_zip(Y, Y_pred, tau):
    # 1 if y_i > 0
    mask_gt = Y[:,0].gt(0).reshape(-1,1)
    
    # 1 if y_i = 0
    mask_0 = Y[:,0].eq(0).reshape(-1,1)

    # if y_i = 0
    y_zero = mask_0*torch.log(torch.exp(tau) + torch.exp(-1*Y[:,1].reshape(-1,1)*Y_pred))
        
    # if y_i > 0
    y_gt = mask_gt*(Y[:,0].reshape(-1,1)*(torch.log(Y[:,1]).reshape(-1,1) + torch.log(Y_pred)) - Y[:,1].reshape(-1,1)*Y_pred)
        
    # For all y_i [if Y = 0, log(Y!) = log(1) = 0]
    factorial_tensor = torch.tensor(special.factorial(Y[:,0].reshape(-1,1)))
    y_all = torch.log(factorial_tensor) + torch.log(1 + torch.exp(tau))
    
    return torch.sum(y_zero + y_gt - y_all)
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# deviance_zip: deviance for zip outcome
def deviance_zip(Y, Y_pred, tau):

    null_ll = ll_zip(Y, torch.mean(Y[:,0]), tau)
    sat_ll = ll_zip(Y, Y[:,0].reshape(-1,1)+0.000001, tau)
    model_ll = ll_zip(Y, Y_pred, tau)
    
    return {'ll': model_ll,
            'null_ll': null_ll,
            'sat_ll': sat_ll,
            'dev': 2*(sat_ll - model_ll),
            'null_dev': 2*(sat_ll - null_ll)}
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# init_tau: estimate proportion of excess zeros based on Lambert (1992)
def init_tau(Y, Z, theta, beta):
    lp = beta + torch.cat(Z, dim=0).matmul(theta)
    Y_cat = torch.cat(Y, dim=0)
    return (Y_cat[:,0].eq(0).sum() - torch.exp(-torch.exp(lp)).sum())/Y_cat.shape[0]
#----------------------------------------------------------------------------------

#-------------------------------------------------------------
# plot_loss: plot losses within sub-optimizations
def plot_loss(hist, title):
    plt.figure()
    plt.subplot(121)
    plt.plot(hist['loss'], color='blue')
    plt.title("Loss")
    plt.subplot(122)
    plt.plot(hist['losspen'], color='purple')
    plt.title("Loss + Penalty")
    plt.show()
    
    return None

#----------------------------------------------------------------------------------
# functions to check if step-size is appropriate.
def Q_L_xi(w, y, X, Z, G, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=Z, G=G, Xi=y, type='view_sub') + torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
    
def Q_L_g(w, y, X, Z, Xi, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=Z, G=y, Xi=Xi, type='view') +  torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
    
def Q_L_z(w, y, Y, X, G, Xi, theta_dict, Lbar):
    diff = w - y
    return assoc_loss(X=X, Z=y, G=G, Xi=Xi, type='sub') + pred_loss(Y=Y, Z=y, theta_dict=theta_dict, single=True) + torch.trace(diff.t().matmul(y.grad)) + (Lbar/2.0)*(torch.pow(diff, 2).sum())

def Q_L_theta(w_t, w_b, y_t, y_b, Y, Z, tau, Lbar):
    diff = torch.cat((w_b, w_t)) - torch.cat((y_b, y_t))
    return pred_loss(Y=Y, Z=Z, theta_dict={'theta':y_t, 'beta':y_b, 'tau':tau}) + torch.trace(diff.t().matmul(torch.cat((y_b.grad, y_t.grad)))) + (Lbar/2.0)*(torch.pow(diff, 2).sum())
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# opt_xi: optimize Xi^d,s using Adagrad in PyTorch
def opt_xi(X, Z, G, Xi, lam_xi, max_iter = 5000, update_thresh = 10**-5, verbose=False):
    
    xi_opt = Xi.clone().detach()
    
    hist = {'loss': [],
            'losspen': []}
    
    old_loss = 1
    old_losspen = 1
    hit_max = True
    
    xi_opt.requires_grad = True
    optimizer = torch.optim.Adagrad(params=[xi_opt])
    
    for c in range(max_iter):
    
        # Zero gradients
        optimizer.zero_grad()
        # Calculate current gradient
        xi_obj = assoc_loss(X=X, Z=Z, G=G, Xi=xi_opt, type = 'view_sub') + lam_xi*torch.pow(xi_opt,  2).sum(dim = 1).sqrt().sum()
        xi_obj.backward()
        # update estimates
        optimizer.step()
            
        # check for convergence
        with torch.no_grad():
        
            cur_loss = assoc_loss(X=X, Z=Z, G=G, Xi=xi_opt, type = 'view_sub')
            cur_losspen =  cur_loss + lam_xi*torch.pow(xi_opt,  2).sum(dim = 1).sqrt().sum()
            hist['loss'].append(cur_loss)
            hist['losspen'].append(cur_losspen)
            relloss = abs(cur_loss - old_loss)/old_loss
            rellosspen = abs(cur_losspen - old_losspen)/old_losspen

            if verbose and (c % 10 == 0):
                print("Xi: c =", c, "; loss =", cur_loss, "; losspen", cur_losspen, "; rellosspen =", rellosspen)
            
            if c > 1  and rellosspen < update_thresh:
                print("Xi converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break

        # update
        with torch.no_grad():
            old_loss = cur_loss
            old_losspen = cur_losspen
    
    if hit_max:
        print("Xi failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'Xi Loss')
    
    return {'val': xi_opt,
            'message': message
           }

#----------------------------------------------------------------------------------
# opt_g: optimize G^d using Adagrad in PyTorch
def opt_g(X, Z, G, Xi, lam_g, max_iter = 5000, update_thresh = 10**-5, verbose = False):
    
    g_opt = G.clone().detach()
    
    hist = {'loss': [],
            'losspen': []}
    
    old_loss = 1
    old_losspen = 1
    hit_max = True
    
    g_opt.requires_grad = True
    optimizer = torch.optim.Adagrad(params=[g_opt])
    
    for c in range(max_iter):
    
        # Zero gradients
        optimizer.zero_grad()
        # Calculate current gradient
        g_obj = assoc_loss(X=X, Z=Z, G=g_opt, Xi=Xi, type = 'view') + lam_g*torch.pow(g_opt,  2).sum(dim = 1).sqrt().sum()
        g_obj.backward()
        # update estimates
        optimizer.step()
            
        # check for convergence
        with torch.no_grad():
        
            cur_loss = assoc_loss(X=X, Z=Z, G=g_opt, Xi=Xi, type = 'view')
            cur_losspen = cur_loss + lam_g*torch.pow(g_opt,  2).sum(dim = 1).sqrt().sum()
            hist['loss'].append(cur_loss)
            hist['losspen'].append(cur_losspen)
            relloss = abs(cur_loss - old_loss)/old_loss
            rellosspen = abs(cur_losspen - old_losspen)/old_losspen

            if verbose and (c % 10 == 0):
                print("G: c =", c, "; loss =", cur_loss, "; losspen =", cur_losspen, "; rellosspen =", rellosspen)
            
            if c > 1  and rellosspen < update_thresh:
                print("G converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break

        # update
        with torch.no_grad():
            old_loss = cur_loss
            old_losspen = cur_losspen
    
    if hit_max:
        print("G failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'G Loss')
    
    return {'val': g_opt,
            'message': message
           }
           
#----------------------------------------------------------------------------------
# fista_z: optimize Z^s using FISTA with backtracking
def fista_z(Y, X, Z, G, Xi, theta_dict, max_iter = 500, update_thresh = 10**-5, verbose = False, L0=1, eta=2, max_iter_inner=30):
    
    x0 = Z.clone().detach()
    y_opt = Z.clone().detach()
    
    hist = {'loss':[], 'losspen': []} # losspen not used but included to prevent key error
    hist['loss'].append(assoc_loss(X=X, Z=y_opt, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=y_opt, theta_dict=theta_dict, single=True))

    L = L0
    t_old = 1
    
    hit_max = True
    
    # i starts at 1 now; so i = k
    for c in range(1, max_iter):
        
        # Calculate the gradient at current estimate using PyTorch
        y_opt.requires_grad = True
        y_obj = assoc_loss(X=X, Z=y_opt, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=y_opt, theta_dict=theta_dict, single=True)
        y_obj.backward()
        
        hit_max_inner = True
        
        # Find appropriate step size - backtracking
        for j in range(max_iter_inner):
            Lbar = L*(eta**j)

            with torch.no_grad():

                xk = y_opt - (1.0/Lbar)*y_opt.grad
                
                F_val = assoc_loss(X=X, Z=xk, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=xk, theta_dict=theta_dict, single=True)
                Q_val = Q_L_z(w=xk, y=y_opt, Y=Y, X=X, G=G, Xi=Xi, Lbar=Lbar, theta_dict=theta_dict)
                
                if F_val <= Q_val:
                    L = Lbar
                    hit_max_inner = False
                    break
        
        if hit_max_inner:
            print("Learning rate issue; returning previous estimate")
            return {'val': x0,
                    'message': "L satisfying condition not found"
                    }
        
        # check for convergence
        with torch.no_grad():
            
            hist['loss'].append(assoc_loss(X=X, Z=xk, G=G, Xi=Xi, type = 'sub') + pred_loss(Y=Y, Z=xk, theta_dict=theta_dict, single=True))
            relloss = abs(hist['loss'][c-1] - hist['loss'][c])/hist['loss'][c-1]
            max_diff = torch.max(torch.abs(xk - x0))
            
            if verbose:
                print("Z: c =", c, "; loss =", hist['loss'][c], "; relloss =", relloss, "; max diff =", max_diff)
            
            if c > 1 and relloss < update_thresh:
                print("Z converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break
            
        # update
        with torch.no_grad():
            t_new = (1 + math.sqrt(1 + 4*t_old**2))/2.0
            y_opt = xk + ((t_old - 1)/t_new)*(xk - x0)
            
            x0 = xk
            t_old = t_new
    
    if hit_max:
        message = 'MAX ITERS'
        print("Z failed to converge in", c, "iterations.")
    
    if verbose:
        plot_loss(hist, 'Z Loss')
        
    return {'val': xk,
            'message': message
           }

#----------------------------------------------------------------------------------
# ista_theta: optimize theta/beta_0 using ISTA with backtracking
def ista_theta(Y, Z, theta_dict, max_iter = 500, update_thresh = 10**-5, verbose = False, L0=1, eta=2, max_iter_inner=30):
    
    t_opt = theta_dict['theta'].clone().detach()
    b_opt = theta_dict['beta'].clone().detach()
    
    hist = {'loss':[], 'losspen': []} # losspen not used but included to prevent key error
    
    hist['loss'].append(pred_loss(Y=Y, Z=Z, theta_dict={'theta':t_opt, 'beta':b_opt, 'tau':theta_dict['tau']}))
    
    L = L0
    hit_max = True
    
    # i starts at 1 now; so i = k
    for c in range(1, max_iter):
        
        # Calculate gradient at current estimate
        t_opt.requires_grad = True
        b_opt.requires_grad = True
        obj = pred_loss(Y=Y, Z=Z, theta_dict={'theta':t_opt, 'beta':b_opt, 'tau':theta_dict['tau']})
        obj.backward()
        
        hit_max_inner = True
        
        # Find appropriate step size - backtracking
        for j in range(max_iter_inner):
            Lbar = L*(eta**j)

            with torch.no_grad():
                tk = t_opt - (1.0/Lbar)*t_opt.grad
                bk = b_opt - (1.0/Lbar)*b_opt.grad
                
                F_val = pred_loss(Y=Y, Z=Z, theta_dict={'theta':tk, 'beta':bk, 'tau':theta_dict['tau']})
                Q_val = Q_L_theta(w_t=tk, w_b=bk, y_t=t_opt, y_b=b_opt, Y=Y, Z=Z, tau=theta_dict['tau'], Lbar=Lbar)
                
                if F_val <= Q_val:
                    L = Lbar
                    hit_max_inner = False
                    break
        
        if hit_max_inner:
            print("Learning rate issue; returning previous estimate")
            return {'val_theta': t_opt,
                    'val_beta': b_opt,
                    'message': "L satisfying condition not found"
                    }
        
        with torch.no_grad():
            # check for convergence
            hist['loss'].append(pred_loss(Y=Y, Z=Z, theta_dict={'theta':tk, 'beta':bk, 'tau':theta_dict['tau']}))
            
            relloss = abs(hist['loss'][c-1] - hist['loss'][c])/hist['loss'][c-1]
            max_diff = torch.max(torch.abs(tk - t_opt).max(), torch.abs(bk - b_opt).max())
            
            if verbose:
                print("theta: c =", c, "; loss =", hist['loss'][c], "; relloss =", relloss, "; max diff =", max_diff)
            
            if c > 1 and (relloss < update_thresh):
                print("Theta converged in", c, "iterations.")
                message = 'Converged'
                hit_max = False
                break
            
            # update
            t_opt = tk
            b_opt = bk
   
    if hit_max:
        print("Theta failed to converge in", c, "iterations.")
        message = 'MAX ITERS'
    
    if verbose:
        plot_loss(hist, 'Theta/Beta Loss')
   
    return {'val': {'theta': tk.clone().detach(), 'beta': bk.clone().detach(), 'tau': theta_dict['tau']},
            'message': message
           }

# extra functions    
def compute_MSE_all(Y_pred, Y_actual):
    mydiff = torch.cat(Y_pred) - torch.cat(Y_actual)
    return(torch.mean(mydiff**2))
    
    
def compute_MSE_single(Y_pred, Y_actual):
    mydiff = Y_pred - Y_actual
    return(torch.mean(mydiff**2))

