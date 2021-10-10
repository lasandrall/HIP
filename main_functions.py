# main_functions.py
# Contains the following functions:
#   generate_data
#   select_K_simple
#   optimize_torch
#   optimize_torch_cont
#   optimize_torch_class
#   select_lambda
#   train_mse
#   test_mse
#   train_class
#   test_class
# Please see helper_functions.py for additional functions
#   not meant to be called by the user.
# Author: Jessica Butts
# Date: June 24, 2021

# Imports and Set-up
#----------------------------------------------------------------------------------
import torch
import math
import numpy as np
import multiprocessing as mp
import time
import random
import pickle
import itertools
import copy

# For adagrad optimization
from torch.autograd import Variable

# Set default tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

#----------------------------------------------------------------------------------
# generate_data: Generate simulated data to use with HIP
# Inputs:
#   seed - int or string - set a seed for replicability
#   n - int vector - number of subjects desired in each subgroup; should be length S
#   p - int vector - number of covariates desired in each data set; should be length D
#   K - int - number of latent components to use in generating the data
#   D - int - number of data sets to generate
#   S - int - number of subgroups
#   nonzero - int - number of important variables for each subgroup; same for all subgroups
#   overlap - int - how many variables before overlap starts between subgroups
#               (i.e. overlap = 0 will be full overlap between subgroups); same for all subgroups
#   sigma_x - double - factor by which the errors added to X are scaled by
#   sigma_y - double - factor by which the errors added to Y are scaled by
#   family - string - determines what type of outcome is generated; either 'gaussian' or 'multiclass'
#       if 'multiclass' is used, then m should also be specified or m will default to 2
#   Optional:
#   m - int - number of classes in multiclass outcome; can be >= 2 but default is 2
#   q - int - number of continuous outcomes; can be >= 1 but default is 1
#   theta_init - float matrix - matrix to use for theta; if None, then a theta matrix will be generated from a U(0,1) distribution
# Outputs:
#   X - matrix list - Contains all X^d,s matrices; access as X[d][s]
#   Y - matrix list - Contains all Y^s matrices; access as Y[s]
#   Z - matrix list - Contains all Z^s matrices used to generate the data; access as Z[s]
#   B - matrix list - Contains all b^d,s matrices used to generate the data; access as B[d][s]
#   theta_init - matrix - Contains the theta matrix used in generating the data
def generate_data(seed, n, p, K, D, S, nonzero, overlap, sigma_x, sigma_y, family, q = 1, m = 2, theta_init = None):
    # Check inputs make sense
    if len(n) != S:
        raise Exception("The length of n does not match the number of subgroups.")
    if len(p) != D:
        raise Exception("The length of p does not match the number of data views.")
    if sigma_x < 0 or sigma_y < 0:
        raise Exception("Sigma_x and sigma_y must be non-negative.")

    # set seed for reproducibility
    torch.manual_seed(seed)

    if family == 'gaussian':
        # Check dimension of theta
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != q):
            raise Exception("The dimensions of theta are incorrect. It must be K x q.")
    
        # set true theta value depending on K and q
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, q).uniform_(0, 1)

        B = [[torch.linalg.qr(torch.cat((torch.zeros(overlap*s, K), torch.Tensor(nonzero, K).uniform_(.5, 1), torch.zeros(p[d]-nonzero-overlap*s, K)))).Q for s in range(S)] for d in range(D)]
        Z = [torch.normal(0,1,(n[s], K)) for s in range(S)]
        E = [[sigma_x*torch.normal(0,1,(n[s], p[d])) for s in range(S)] for d in range(D)]
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        Y = [torch.matmul(Z[s], theta_init) + sigma_y*torch.normal(0,1,(n[s], q)) for s in range(S)]

    elif family == 'multiclass':
        # Check dimension of theta
        if theta_init != None and (theta_init.shape[0] != K or theta_init.shape[1] != m):
            raise Exception("The dimensions of theta are incorrect. It must be K x m.")
    
        # set true theta value depending on K and m
        # user also has the option to set theta_init
        if theta_init == None:
            theta_init = torch.Tensor(K, m).uniform_(0, 1)

        B = [[torch.linalg.qr(torch.cat((torch.zeros(overlap*s, K), torch.Tensor(nonzero, K).uniform_(.5, 1), torch.zeros(p[d]-nonzero-overlap*s, K)))).Q for s in range(S)] for d in range(D)]
        Z = [torch.normal(0,1,(n[s], K)) for s in range(S)]
        E = [[sigma_x*torch.normal(0,1,(n[s], p[d])) for s in range(S)] for d in range(D)]
        X = [[torch.matmul(Z[s], torch.t(B[d][s])) + E[d][s]  for s in range(S)] for d in range(D)]
        Y = [torch.empty((n[s],m)) for s in range(S)]
        for s in range(S):
            P = calc_probs(torch.mm(Z[s], theta_init), 1)
            Y[s] = torch.argmax(P, dim=1).to(torch.double)
    
    else:
        raise Exception("The family you entered is not a valid option.")

    return  X, Y, Z, B, theta_init
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# select_K_simple: Uses the simple approach to select a value for K
# Inputs:
#   X - matrix list - list of X^d,s matrices containing covariates
#   Optional:
#   threshold - float - value to use as the threshold in the simple approach; default = 0.1
#   verbose - boolean - whether to print information about the K selected; default = True
# Outputs:
#   kchoose - int - suggested value for K
def select_K_simple(X, threshold = 0.1, verbose=True):
    if threshold <= 0:
        raise Exception("Threshold value must be positive.")

    D = len(X)
    S = len(X[0])

    for d in range(D):
        for s in range(S):
            X[d][s] = torch.as_tensor(np.array(X[d][s]))
            
    # Form concatenated data matrix
    Xcat = torch.cat([torch.cat(X[d]) for d in range(D)], dim=1)

    # Perform SVD to get eigenvalues
    Z_u, Z_d, Z_vt = torch.svd(Xcat)

    # Select K based on percent change in eigenvalues
    calc2 = list()
    for j in range(len(Z_d)-1):
        calc2.append((Z_d[j] - Z_d[j+1])/Z_d[j])
        if(calc2[j] < threshold):
            kchoose = j+1
            if verbose:
                print("K Based on simple approach using", threshold, "as cut-off:", kchoose)
            return kchoose
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# optimize_torch: Apply HIP with fixed tuning parameters; redirects to appropriate optimization based on family
# Inputs:
#   Y - float or int list - list of s matrices containing outcome data
#   X - float list - list of X^d,s matrices containing covariates
#   lambda_xi - float - value for lambda_xi tuning parameter
#   lambda_g - float - value for lambda_g tuning parameter
#   gamma - int list - Indicators for whether to penalize each data set; should be length D
#   family - string - Type of outcome; either 'gaussian' or 'multiclass' (includes binary outcomes)
#   Optional:
#   K - int - Number of latent components to use in the model; If not provided,
#       K will be selected using the simple approach with a threshold of 0.1
#   threshold - float - threshold to use in simple K selection; default is 0.1
#   standardize - boolean - Whether to standardize the X and Y data; default is True
#   max_iter - int - the maximum number of iterations allowed in the algorithm; default is 200
#   print_time - boolean - determines whether the time for a single iteration is printed; default is True
# Outputs:
#   dict with the following components:
#       - theta - matrix - estimated theta
#       - B - matrix list - estimated B^d,s matrices
#       - G - matrix list - estimated G^d matrices
#       - Xi - matrix list - estimatted Xi^d,s matrices
#       - Z - matrix list - estimated Z^s matrices
#       - lambda - float vector - lambda_xi and lambda_g used in the algorithm
#       - BIC - float - calculated BIC for the model fit
#       - mse_by_var/mse_overall - float list/float - calculated training mse for  each outcome/all outcomes for the model fit.
#            OR accuracy - float - calculated classification accuracy for the model fit
#       - message - string - tells whether the algorithm converged or encountered an error
def optimize_torch(Y, X, lambda_xi, lambda_g, gamma, family, K = None, threshold = 0.1, standardize = True, max_iter=200, print_time = True):
    # make sure data appropriate
    D = len(X)
    S = len(X[0])
    for d in range(D):
        p = [X[d][s].shape[1] for s in range(S)]
        if len(np.unique(p)) != 1:
            raise Exception(' '.join(("Subgroups in view", str(d+1), "have differing numbers of variables.")))
    for s in range(S):
        n = [X[d][s].shape[0] for d in range(D)]
        if len(np.unique(n)) != 1:
            raise Exception(' '.join(("Subgroup", str(s+1), "has differing numbers of observations across the data views.")))
    if type(Y) != list:
        raise Exception("Y must be a list.")
    if type(Y) is list and len(Y) != S:
        raise Exception("Y and X have differing numbers of subgroups.")
    if len(X) != len(gamma):
        raise Exception("Gamma must be length D.")
    if lambda_xi < 0 or lambda_g < 0:
        raise Exception("Lambda parameters must be non-negative.")

    if family == 'gaussian':
        return optimize_torch_cont(Y=Y, X=X, lambda_xi=lambda_xi, lambda_g=lambda_g, gamma=gamma, K = K, threshold=threshold, standardize=standardize, max_iter=max_iter, print_time=print_time)
    elif family == 'multiclass':
        return optimize_torch_class(Y=Y, X=X, lambda_xi=lambda_xi, lambda_g=lambda_g, gamma=gamma, K=K, threshold=threshold, standardize=standardize, max_iter=max_iter, print_time=print_time)
    else:
        raise Exception("The family you entered is not a valid option.")
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# optimize_torch_cont: Apply HIP with fixed tuning parameters for continuous outcome(s)
# Inputs:
#   Y - float or int list - list of s matrices containing outcome data
#   X - float list - list of X^d,s matrices containing covariates
#   lambda_xi - float - value for lambda_xi tuning parameter
#   lambda_g - float - value for lambda_g tuning parameter
#   gamma - int list - Indicators for whether to penalize each data set; should be length D
#   Optional:
#   K - int - Number of latent components to use in the model; If not provided,
#       K will be selected using the simple approach with a threshold of 0.1
#   threshold - float - threshold to use in simple K selection; default is 0.1
#   standardize - boolean - Whether to standardize the X and Y data; default is True
#   max_iter - int - the maximum number of iterations allowed in the algorithm; default is 200
#   epsilon - float - error tolerance for determining convergence in the algorithm; default = 0.0001
#   print_time - boolean - determines whether the time for a single iteration is printed; default is True
# Outputs:
#   dict with the following components:
#       - theta - matrix - estimated theta
#       - B - matrix list - estimated B^d,s matrices
#       - G - matrix list - estimated G^d matrices
#       - Xi - matrix list - estimatted Xi^d,s matrices
#       - Z - matrix list - estimated Z^s matrices
#       - lambda - float vector - lambda_xi and lambda_g used in the algorithm
#       - BIC - float - calculated BIC for the model fit
#       - mse_by_var - float  list - calculated training mse for each outcome for the model fit
#       - mse_overall - float - calculated overall training mse for all outcomes for the model fit
#       - message - string - tells whether the algorithm converged or encountered an error
def optimize_torch_cont(Y, X, lambda_xi, lambda_g, gamma, K = None, threshold = .1, standardize = True, max_iter=200, epsilon = 0.0001, print_time = True):
    tic = time.perf_counter()
    count = 1

    # Get n, p, D, S from the X  provided from user
    D = len(X)
    S = len(X[0])
    n = [X[0][s].shape[0] for s in range(S)]
    p = [X[d][0].shape[1] for d in range(D)]

    if print_time:
        print("D =", D)
        print("S =", S)
        print("n =", n)
        print("p =", p)
        
    if K == None:
        K = select_K_simple(X=X, threshold=threshold, verbose=print_time)
        
    # Perform SVD for initializations
    Xcat = torch.cat([torch.cat(X[d]) for d in range(D)], dim=1)
    Z_u, Z_d, Z_vt = torch.svd(Xcat)

    # Initialize matrices for algorithm
    G_old = [torch.ones((p[d], K)) for d in range(D)]
    Xi_old = [[torch.ones((p[d],K)) for s in range(S)] for d in range(D)]
    n2 = [sum(n[:s]) for s in range(S+1)]
    Z_old = [torch.as_tensor(Z_u[n2[s]:(n2[s+1]),:K]) for s in range(S)]
    Z_oldstack = torch.as_tensor(Z_u[:,:K])
    theta_old = torch.inverse(torch.t(Z_oldstack)@Z_oldstack)@torch.t(Z_oldstack)@torch.cat(Y)
    theta_old = theta_old/torch.norm(theta_old, 2)

    # Initialize matrices to hold udpated results
    Xi_new = [[torch.empty((p[d],K)) for s in range(S)] for d in range(D)]
    G_new = [torch.empty(p[d], K) for d in range(D)]
    B_new = [[torch.empty((p[d],K)) for s in range(S)] for d in range(D)]
    Z_new = [torch.empty(n[s], K) for s in range(S)]

    if standardize:
         # Center and Scale X and Y
        for s in range(S):
            y_mean = torch.mean(Y[s], dim=0)
            y_std = torch.std(Y[s], dim=0)
            Y[s] = torch.div((Y[s] - y_mean),y_std)

        for d in range(D):
            for s in range(S):
                x_mean = torch.mean(X[d][s], dim=0)
                x_std = torch.std(X[d][s], dim=0)
                X[d][s] = torch.div((X[d][s] - x_mean), x_std)
        if print_time:
            print('Data Standardized')

    while(True):
        print("Iteration", count)

        # Optimize G and Xi
        for d in range(D):
            lambda_xi_gamma = lambda_xi*gamma[d]
            lambda_g_gamma = lambda_g*gamma[d]
            for s in range(S):
                if lambda_xi_gamma == 0:
                    Xi_temp = adagrad_xi(X = X[d][s],
                                        Z = Z_old[s],
                                        G = G_old[d],
                                        lambda_xi = lambda_xi_gamma)
                else:
                    Xi_temp = OptimizeXi(Xdata = X[d][s],
                                         myZ = Z_old[s],
                                         myG = G_old[d],
                                         mylambda = lambda_xi_gamma)
                Xi_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in Xi_temp], dtype=torch.double)
                norm_xi = torch.norm(Xi_temp, dim = 0)
                if(Zero(Xi_temp)):
                    print("The algorithm encountered an all zero solution in Xi - consider reducing lambda_xi.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'mse': float('inf'),
                            'message': "All Zero Solution"}
                if(sum(sum(torch.isnan(Xi_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'mse': float('inf'),
                            'message': "Nan in Xi matrix"}
                if sum(norm_xi >= 10**-5) == len(norm_xi):
                    Xi_new[d][s] = torch.div(Xi_temp, norm_xi)
                else:
                    Xi_new[d][s] = Xi_temp
            
            if lambda_g_gamma == 0:
                G_temp = adagrad_G(X = X[d],
                                 Z = Z_old,
                                 Xi = Xi_new[d],
                                 lambda_g = lambda_g_gamma)
            else:
                G_temp = OptimizeG2(Xdata = X[d],
                                    myZ = Z_old,
                                    myXi = Xi_new[d],
                                    mylambda = lambda_g_gamma)
            G_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in G_temp], dtype=torch.double)
            norm_g = torch.norm(G_temp, dim = 0)
            
            if(Zero(G_temp)):
                print("The algorithm encountered an all zero solution in G - consider reducing lambda_G.")
                return {'theta': theta_old.numpy(),
                        'B': 0,
                        'G': 0,
                        'Xi':0,
                        'Z': 0,
                        'Lambda':[lambda_xi, lambda_g],
                        'BIC': float('inf'),
                        'mse': float('inf'),
                        'message': "All Zero Solution"}
            if(sum(sum(torch.isnan(G_temp))) > 0):
                print("The algorithm encountered a Nan solution.")
                return {'theta': theta_old.numpy(),
                        'B': 0,
                        'G': 0,
                        'Xi':0,
                        'Z': 0,
                        'Lambda':[lambda_xi, lambda_g],
                        'BIC': float('inf'),
                        'mse': float('inf'),
                        'message': "Nan in G matrix"}
            if sum(norm_g >=10**-5) == len(norm_g):
                G_new[d] = torch.div(G_temp, norm_g)
            else:
                G_new[d] = G_temp
            
        # Calculate B
        for d in range(D):
            for s in range(S):
                B_temp = G_new[d]*Xi_new[d][s]
                B_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in B_temp])
                if(Zero(B_temp)):
                    print("The algorithm encountered an all zero solution in B.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'mse': float('inf'),
                            'message': "All Zero Solution"}
                if(sum(sum(torch.isnan(B_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'mse': float('inf'),
                            'message': "Nan in B matrix"}
                norm_b = torch.norm(B_temp, dim = 0)
                if sum(norm_b >= 10**-5) == len(norm_b):
                    B_new[d][s] = torch.div(B_temp, norm_b)
                else:
                    B_new[d][s] = B_temp

        # Optimize Z
        for s in range(S):
            # check if concatenated is all zeros
            Z_temp = solve_Z(X = [X[d][s] for d in range(D)],
                             B = [B_new[d][s] for d in range(D)],
                             Y = Y[s],
                             theta = theta_old
                             )
            Z_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in Z_temp])
            if(sum(sum(torch.isnan(Z_temp))) > 0):
                print("The algorithm encountered a Nan solution.")
                return {'theta': theta_old.numpy(),
                        'B': 0,
                        'G': 0,
                        'Xi':0,
                        'Z': 0,
                        'Lambda':[lambda_xi, lambda_g],
                        'BIC': float('inf'),
                        'mse': float('inf'),
                        'message': "Nan in Z matrix"}
            norm_z = torch.norm(Z_temp, p='fro')
            if norm_z >= 10**-5:
                Z_new[s] = torch.div(Z_temp, norm_z)
            else:
                Z_new[s] = Z_temp

        # Optimize theta
        Z_stack = torch.cat(Z_new, dim=0)
        theta_new = torch.inverse(Z_stack.T @ Z_stack + 0.01*torch.eye(Z_stack.shape[1])) @ torch.t(Z_stack) @ torch.cat(Y)
        theta_new = theta_new/torch.norm(theta_new, p = 2)
        
        if sum(sum(torch.isnan(theta_new))) > 0:
            print("The algorithm encountered a Nan solution.")
            return {'theta': theta_new.numpy(),
                    'B': 0,
                    'G': 0,
                    'Xi': 0,
                    'Z': 0,
                    'Lambda':[lambda_xi, lambda_g],
                    'BIC': float('inf'),
                    'mse': float('inf'),
                    'message': "Did not converge - Nan value encountered in theta"}

        # Convergence criteria
        theta_norm = torch.norm(theta_old - theta_new, p=2)**2/torch.norm(theta_old, p=2)**2
        if count == 1:
            rel_cost_old = 1

        iter_cost = 0
        iter_cost += torch.norm((Z_stack @ theta_new) - (torch.cat(Z_old) @ theta_old), 'fro')**2
        rel_cost_new = 0
        rel_cost_new += torch.norm(torch.as_tensor(torch.cat(Y, axis = 0)) - Z_stack @ theta_new, 'fro')**2

        for d in range(D):
            for s in range(S):
                iter_cost += torch.norm((Z_new[s] @ B_new[d][s].T) - (Z_old[s] @ (G_old[d]*Xi_old[d][s]).T),'fro')**2
                rel_cost_new += torch.norm(X[d][s] - Z_new[s] @ B_new[d][s].T, 'fro')**2

        rel_cost = abs(rel_cost_new - rel_cost_old) / rel_cost_old

        if(iter_cost < epsilon or theta_norm < epsilon or rel_cost < epsilon):
            print("Stopped on iteration", count)
            N = sum(n)
            norms = 0
            for d in range(D):
                for s in range(S):
                    norms += torch.norm(X[d][s] - Z_new[s]@torch.t(B_new[d][s]), 'fro')**2
            for s in range(S):
                norms += torch.norm(Y[s] - Z_new[s]@theta_new, 'fro')**2
            
            BIC_new = -2*norms + nonZero(B_new)*math.log(N)

            if(count == max_iter):
                message = "Maximum Iterations Reached - Increase Maximum Iterations"
            else:
                message = "Converged"
                
            # calculate training mse for each column, and overall
            mse_ind, mse_all = train_mse(Y_true = Y, Z = Z_new, theta = theta_new, S = S, standardize = standardize)
            
            toc = time.perf_counter()
            if print_time:
                print("The algorithm completed in", toc-tic, "seconds.")
                print("Convergence status:", message)
                print("Lambda used in solution:", [lambda_xi, lambda_g])
                print("BIC:", BIC_new.numpy())
                print("Overall MSE:", mse_all.numpy())
                print("MSE by Outcome:", mse_ind.numpy())
                common = summary_B(B_new)
                print("Number of Selected Variables:")
                for d in range(D):
                    print("Data View", d)
                    print(common[d])
            return {'theta':theta_new,
                    'B':B_new,
                    'G':G_new,
                    'Xi':Xi_new,
                    'Z':Z_new,
                    'Lambda':[lambda_xi, lambda_g],
                    'BIC': BIC_new,
                    'mse_by_var': mse_ind,
                    'mse_overall': mse_all,
                    'message': message}
        
        G_old = G_new
        Xi_old = Xi_new
        Z_old = Z_new
        theta_old = theta_new
        rel_cost_old = rel_cost_new
        count += 1
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# optimize_torch_class: Apply HIP with fixed tuning parameters for multiclass outcome
# Inputs:
#   Y - float or int list - list of s matrices containing outcome data
#   X - float list - list of X^d,s matrices containing covariates
#   lambda_xi - float - value for lambda_xi tuning parameter
#   lambda_g - float - value for lambda_g tuning parameter
#   gamma - int list - Indicators for whether to penalize each data set; should be length D
#   Optional:
#   K - int - Number of latent components to use in the model; If not provided,
#       K will be selected using the simple approach with a threshold of 0.1
#   threshold - float - threshold to use in simple K selection; default is 0.1
#   standardize - boolean - Whether to standardize the X data; default is True
#   max_iter - int - the maximum number of iterations allowed in the algorithm; default is 200
#   epsilon - float - error tolerance for determining convergence in the algorithm; default = 0.0001
#   print_time - boolean - determines whether the time for a single iteration is printed; default is True
# Outputs:
#   dict with the following components:
#       - theta - matrix - estimated theta
#       - B - matrix list - estimated B^d,s matrices
#       - G - matrix list - estimated G^d matrices
#       - Xi - matrix list - estimatted Xi^d,s matrices
#       - Z - matrix list - estimated Z^s matrices
#       - lambda - float vector - lambda_xi and lambda_g used in the algorithm
#       - BIC - float - calculated BIC for the model fit
#       - accuracy - float - training classification accuracy
#       - message - string - tells whether the algorithm converged or encountered an error
def optimize_torch_class(Y, X, lambda_xi, lambda_g, gamma, K = None, threshold = .1, standardize = True, max_iter=200, epsilon = 0.0001, print_time = True):
    tic = time.perf_counter()
    count = 1

    # Get n, p, D, S from the X  provided from user
    D = len(X)
    S = len(X[0])
    n = [X[0][s].shape[0] for s in range(S)]
    p = [X[d][0].shape[1] for d in range(D)]
    
    if print_time:
        print("D =", D)
        print("S =", S)
        print("n =", n)
        print("p =", p)

    if K == None:
        K = select_K_simple(X=X, threshold=threshold, verbose=print_time)

    Y_ind = class_matrix(Y,S)
        
    # Perform SVD for initializations
    Xcat = torch.cat([torch.cat(X[d]) for d in range(D)], dim=1)
    Z_u, Z_d, Z_vt = torch.svd(Xcat)

    # Initialize matrices for algorithm
    G_old = [torch.ones((p[d], K)) for d in range(D)]
    Xi_old = [[torch.ones((p[d],K)) for s in range(S)] for d in range(D)]
    n2 = [sum(n[:s]) for s in range(S +1)]
    Z_old = [torch.as_tensor(Z_u[n2[s]:(n2[s+1]),:K]) for s in range(S)]
    Z_oldstack = torch.as_tensor(Z_u[:,:K])
    theta_old = torch.inverse(torch.t(Z_oldstack)@Z_oldstack)@torch.t(Z_oldstack)@torch.cat(Y_ind)
    theta_old = theta_old/torch.norm(theta_old, 2)

    # Initialize matrices to hold udpated results
    Xi_new = [[torch.empty((p[d],K)) for s in range(S)] for d in range(D)]
    G_new = [torch.empty(p[d], K) for d in range(D)]
    B_new = [[torch.empty((p[d],K)) for s in range(S)] for d in range(D)]
    Z_new = [torch.empty(n[s], K) for s in range(S)]

    if standardize:
         # Center and Scale X
        for d in range(D):
            for s in range(S):
                x_mean = torch.mean(X[d][s], dim=0)
                x_std = torch.std(X[d][s], dim=0)
                X[d][s] = torch.div((X[d][s] - x_mean), x_std)
        if print_time:
            print('Data Standardized')
            
    while(True):
        print("Iteration", count)

        # Optimize G and Xi
        for d in range(D):
            lambda_xi_gamma = lambda_xi*gamma[d]
            lambda_g_gamma = lambda_g*gamma[d]
            for s in range(S):
                if lambda_xi_gamma == 0:
                    Xi_temp = adagrad_xi(X = X[d][s],
                                         Z = Z_old[s],
                                         G = G_old[d],
                                         lambda_xi = lambda_xi_gamma)
                else:
                    Xi_temp = OptimizeXi(Xdata = X[d][s],
                                            myZ = Z_old[s],
                                            myG = G_old[d],
                                            mylambda = lambda_xi_gamma)
                Xi_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in Xi_temp], dtype=torch.double)
                norm_xi = torch.norm(Xi_temp, dim = 0)
                if(Zero(Xi_temp)):
                    print("The algorithm encountered an all zero solution in Xi - consider reducing lambda_xi.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'AIC': float('inf'),
                            'Test': float('nan'),
                            'Train': float('nan'),
                            'message': "All Zero Solution"}
                if(sum(sum(torch.isnan(Xi_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'accuracy': float('inf'),
                            'message': "Nan in Xi matrix"}
                if sum(norm_xi >= 10**-5) == len(norm_xi):
                    Xi_new[d][s] = torch.div(Xi_temp, norm_xi)
                else:
                    Xi_new[d][s] = Xi_temp
            
            if lambda_g_gamma == 0:
                G_temp = adagrad_G(X = X[d],
                                 Z = Z_old,
                                 Xi = Xi_new[d],
                                 lambda_g = lambda_g_gamma)
            else:
                G_temp = OptimizeG2(Xdata = X[d],
                                    myZ = Z_old,
                                    myXi = Xi_new[d],
                                    mylambda = lambda_g_gamma)
            G_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in G_temp])
            norm_g = torch.norm(G_temp, dim = 0)
            if(Zero(G_temp)):
                print("The algorithm encountered an all zero solution in G - consider reducing lambda_g.")
                return {'theta': theta_old.numpy(),
                        'B': 0,
                        'G': 0,
                        'Xi':0,
                        'Z': 0,
                        'Lambda': [lambda_xi, lambda_g],
                        'BIC': float('inf'),
                        'accuracy': float('inf'),
                        'message': "All Zero Solution"}
            if(sum(sum(torch.isnan(G_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'accuracy': float('inf'),
                            'message': "Nan in G matrix"}
            if sum(norm_g >=10**-5) == len(norm_g):
                G_new[d] = torch.div(G_temp, norm_g)
            else:
                G_new[d] = G_temp
            

        # Calculate B
        for d in range(D):
            for s in range(S):
                B_temp = G_new[d]*Xi_new[d][s]
                B_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in B_temp])
                if(Zero(B_temp)):
                    print("The algorithm encountered an all zero solution in B.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'AIC': float('inf'),
                            'Test': float('nan'),
                            'Train': float('nan'),
                            'message': "All Zero Solution"}
                if(sum(sum(torch.isnan(B_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'accuracy': float('inf'),
                            'message': "Nan in B matrix"}
                norm_b = torch.norm(B_temp, dim = 0)
                if sum(norm_b >= 10**-5) == len(norm_b):
                    B_new[d][s] = torch.div(B_temp, norm_b)
                else:
                    B_new[d][s] = B_temp

        # Optimize Z
        for s in range(S):
            Z_temp = adagrad_Z(X = [X[d][s] for d in range(D)],
                               B = [B_new[d][s] for d in range(D)],
                               Y = Y_ind[s],
                               theta = theta_old
                              )
            Z_temp = torch.as_tensor([[0 if abs(number) < 10**-5 else number for number in row] for row in Z_temp])
            if(sum(sum(torch.isnan(Z_temp))) > 0):
                    print("The algorithm encountered a Nan solution.")
                    return {'theta': theta_old.numpy(),
                            'B': 0,
                            'G': 0,
                            'Xi':0,
                            'Z': 0,
                            'Lambda':[lambda_xi, lambda_g],
                            'BIC': float('inf'),
                            'accuracy': float('inf'),
                            'message': "Nan in Z matrix"}
            norm_z = torch.norm(Z_temp, p='fro')
            if norm_z >= 10**-5:
                Z_new[s] = torch.div(Z_temp, norm_z)
            else:
                Z_new[s] = Z_temp

        # Optimize theta
        Z_stack = torch.cat(Z_new, dim=0)
        theta_new = adagrad_theta(Y_ind, Z_new)
        theta_new = torch.true_divide(theta_new, torch.norm(theta_new, p=2, dim = 0))
        
        if sum(sum(torch.isnan(theta_new))) > 0:
            print("The algorithm encountered a Nan solution.")
            return {'theta': theta_new.numpy(),
                    'B': 0,
                    'G': 0,
                    'Xi': 0,
                    'Z': 0,
                    'Lambda':[lambda_xi, lambda_g],
                    'BIC': float('inf'),
                    'accuracy': float('inf'),
                    'message': "Did not converge - Nan value encountered in theta"}

        # Convergence criteria
        theta_norm = torch.norm(theta_old - theta_new, p=2)**2/torch.norm(theta_old, p=2)**2
        if count == 1:
            rel_cost_old = 1

        iter_cost = 0
        iter_cost += torch.norm((Z_stack @ theta_new) - (torch.cat(Z_old) @ theta_old), 'fro')**2
        rel_cost_new = 0
        rel_cost_new = class_loss(torch.cat(Y_ind), calc_probs(torch.mm(Z_stack, theta_new),1), 1)

        for d in range(D):
            for s in range(S):
                iter_cost += torch.norm((Z_new[s] @ B_new[d][s].T) - (Z_old[s] @ (G_old[d]*Xi_old[d][s]).T),'fro')**2
                rel_cost_new += torch.norm(X[d][s] - Z_new[s] @ B_new[d][s].T, 'fro')**2

        rel_cost = abs(rel_cost_new - rel_cost_old) / rel_cost_old

        if(iter_cost < epsilon or theta_norm < epsilon or rel_cost < epsilon):
            print("Stopped on iteration", count)
            
            norms = rel_cost_new
            
            N = sum(n)
            
            BIC_new = -2*norms + nonZero(B_new)*math.log(N)
        
            if(count == max_iter):
                message = "Maximum Iterations Reached - Increase Maximum Iterations"
            else:
                message = "Converged"
                
            # calculate training mse for each column, and overall
            acc = train_class(Y_true = Y, Z = Z_new, theta = theta_new)
            
            toc = time.perf_counter()
            if print_time:
                print("The algorithm completed in", toc-tic, "seconds.")
                print("Convergence status:", message)
                print("Lambda used in solution:", [lambda_xi, lambda_g])
                print("BIC:", BIC_new.numpy())
                print("Accuracy:", acc.numpy())
                common = summary_B(B_new)
                print("Number of Selected Variables:")
                for d in range(D):
                    print("Data View", d)
                    print(common[d])
            
            return {'theta':theta_new,
                    'B':B_new,
                    'G':G_new,
                    'Xi':Xi_new,
                    'Z':Z_new,
                    'Lambda': [lambda_xi, lambda_g],
                    'BIC': BIC_new,
                    'accuracy': acc,
                    'message': message}
        
        G_old = G_new
        Xi_old = Xi_new
        Z_old = Z_new
        theta_old = theta_new
        rel_cost_old = rel_cost_new
        count += 1

#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# select_lambda: Perform grid or random search to find optimal tuning parameters
# Inputs:
#   Y - float or int list - list of s matrices containing outcome data
#   X - float list - list of X^d,s matrices containing covariates
#   gamma - int list - Indicators for whether to penalize each data set; should be length D
#   family - string - Type of outcome; either 'gaussian' or 'multiclass'
#   Optional:
#   K - int - number of latent components to use in the model
#   threshold - float - cut-off used in simple approach to select K
#   standardize - boolean - whether to standardize the covariates (and outcome(s) if continuous)
#   search - string - type of search, either 'grid' or 'random'; default is 'random'
#   upper - float - upper bound of tuning parameter range to search (inclusive); default is 1
#   lower - float - lower bound of tuning parameter range to search (not inclusive); default is 0
#   num_steps - int - number of steps to consider for each tuning parameter; default is 8
# Outputs:
#   dict with optimal tuning parameters based on BIC; same entries as optimize_torch
def select_lambda(Y, X, gamma, family, K=None, threshold=0.1, standardize = True, search='random', upper=1, lower=0, num_steps=8):
    #Get n, p, d, s from the X provided from user
    D = len(X)
    print("D =", D)
    S = len(X[0])
    print("S =", S)
    n = [X[0][s].shape[0] for s in range(S)]
    print("n =", n)
    p = [X[d][0].shape[1] for d in range(D)]
    print("p =", p)
    
    for d in range(D):
        p = [X[d][s].shape[1] for s in range(S)]
        if len(np.unique(p)) != 1:
            raise Exception(' '.join(("Subgroups in view", str(d+1), "have differing numbers of variables.")))
    for s in range(S):
        n = [X[d][s].shape[0] for d in range(D)]
        if len(np.unique(n)) != 1:
            raise Exception(' '.join(("Subgroup", str(s+1), "has differing numbers of observations across the data views.")))
    if type(Y) != list:
        raise Exception("Y must be a list.")
    if type(Y) is list and len(Y) != S:
        raise Exception("Y and X have differing numbers of subgroups.")
    if len(X) != len(gamma):
        raise Exception("Gamma must be length D.")
    
    if K == None:
        K = select_K_simple(X=X, threshold=threshold, verbose=True)
        
    tic = time.perf_counter()

    test = list()
    for d in range(D):
        lam = [lower,upper]
        steps = list()
        for j in range(0,num_steps+1):
            steps.append(j*(lam[1]-lam[0])/(num_steps) + lam[0])
        test.append(steps[1:])

    # All possible combinations of lambda datasets
    combos = list(itertools.product(*test))

    # Random or Grid search
    if search == 'random':
        random.seed(7)
        selection = random.sample(range(num_steps**D), math.ceil(.15*(num_steps**D)))
    elif search == 'grid':
        selection = range(num_steps**D)

    result = list()
    if family == 'gaussian':
        for val in selection:
            result.append(optimize_torch_cont(Y=Y, X=X, lambda_xi=combos[val][0], lambda_g=combos[val][1], gamma=gamma, K=K, standardize=standardize, print_time=False))
    elif family == 'multiclass':
        for val in selection:
            result.append(optimize_torch_class(Y=Y, X=X, lambda_xi=combos[val][0], lambda_g=combos[val][1], gamma=gamma, K=K, standardize=standardize, print_time=False))
    else:
        raise Exception("The family you entered is not a valid option.")

    BIC = []
    for l in range(len(result)):
        BIC.append(result[l]['BIC'])

    BIC_index = np.argmin(BIC)
    
    toc = time.perf_counter()
    result[BIC_index]['Time'] = toc - tic
    
    print("The lambda selection process finished in", toc-tic, "seconds.")
    print("Convergence status:", result[BIC_index]['message'])
    print("Lambda used in solution:", result[BIC_index]['Lambda'])
    print("BIC:", result[BIC_index]['BIC'].numpy())
    if family == 'gaussian':
        print("Overall MSE:", result[BIC_index]['mse_overall'].numpy())
        print("MSE by Outcome:", result[BIC_index]['mse_by_var'].numpy())
    elif family == 'multiclass':
        print("Accuracy:", result[BIC_index]['accuracy'].numpy())
    common = summary_B(result[BIC_index]['B'])
    print("Number of Selected Variables:")
    for d in range(D):
        print("Data View", d)
        print(common[d])

    return result[BIC_index]
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# train_mse: calculate the training MSE on fitted model
# Inputs:
#   Y_true - matrix list - outcomes for all subgroups
#   Z - matrix list - Estimated Z^s matrices from HIP
#   theta - matrix - Estimated theta from HIP
#   S - int - number of subgroups
#   standardize - boolean - indicates whether to standardize the original data
# Outputs:
#   mse_each - float list - mse for each outcome column in Y
#   mse_all - float - mse averaged across all outcomes
def train_mse(Y_true, Z, theta, S, standardize):
    q = Y_true[0].shape[1]

    Y_pred = [[] for s in range(S)]

    for s in range(S):
        Y_pred[s] = Z[s] @ theta
        if standardize:
            y_mean = torch.mean(Y_true[s],dim=0)
            y_std = torch.std(Y_true[s],dim=0)
            Y_true[s] = torch.div((Y_true[s] - y_mean),y_std)
            
            y_mean = torch.mean(Y_pred[s],dim=0)
            y_std = torch.std(Y_pred[s],dim=0)
            Y_pred[s] = torch.div((Y_pred[s] - y_mean),y_std)
    mydiff=torch.cat(Y_pred) - torch.cat(Y_true)
    mse_all=torch.norm(mydiff,'fro')**2 /(len(torch.cat(Y_pred))*q)
    mse_each=sum((torch.cat(Y_pred) - torch.cat(Y_true))**2)/len(torch.cat(Y_pred))
    return mse_each, mse_all
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_mse: calculate the test MSE using fitted model
# Inputs:
#   Y_true - matrix list - outcomes for all subgroups
#   X - matrix list - X^d,s matrices containing the covariates from the test data
#   B - matrix list - Estimated B^d,s matrices from HIP
#   theta - matrix - Estimated theta from HIP
#   standardize - boolean - indicates whether to standardize the original data
# Outputs:
#   mse_each - float list - mse for each outcome column in Y
#   mse_all - float - mse averaged across all outcomes
def test_mse(Y_true, X, B, theta, standardize):
    D = len(X)
    S = len(X[0])
    q = Y_true[0].shape[1]

    if standardize:
        # Center and Scale X and Y
        for s in range(S):
            y_mean = torch.mean(Y[s], dim=0)
            y_std = torch.std(Y[s], dim=0)
            Y[s] = torch.div((Y[s] - y_mean),y_std)

        for d in range(D):
            for s in range(S):
                x_mean = torch.mean(X[d][s], dim=0)
                x_std = torch.std(X[d][s], dim=0)
                X[d][s] = torch.div((X[d][s] - x_mean), x_std)

    Xcat = [[[] for d in range(D)] for s in range(S)]
    Bcat = [[[] for d in range(D)] for s in range(S)]
    Z_pred = [[] for s in range(S)]
    Y_pred = [[] for s in range(S)]
    for s in range(S):
        Xcat[s] = X[0][s]
        Bcat[s] = B[0][s]
    for s in range(S):
        for d in range(1,D):
            Xcat[s] = torch.cat((Xcat[s], X[d][s]), dim=1)
            Bcat[s] = torch.cat((Bcat[s], B[d][s]))
        Z_temp = Xcat[s] @ Bcat[s] @ torch.inverse(Bcat[s].T @ Bcat[s] + (0.0001*torch.eye(Bcat[s].shape[1])))
        norm_z = torch.norm(Z_temp, p='fro')
        Z_pred[s] = torch.div(Z_temp, norm_z)
        Y_pred[s] = Z_pred[s] @ theta
        
        if standardize:
            y_mean = torch.mean(Y_pred[s], dim=0)
            y_std = torch.std(Y_pred[s], dim=0)
            Y_pred[s] = torch.div((Y_pred[s] - y_mean),y_std)

    mydiff=torch.cat(Y_pred) - torch.cat(Y_true)
    mse_all=torch.norm(mydiff,'fro')**2 /(len(torch.cat(Y_pred))*q)
    mse_each=sum((torch.cat(Y_pred) - torch.cat(Y_true))**2)/len(torch.cat(Y_pred))

    return mse_each, mse_all
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# train_class: Calculates training classification accuracy for multiclass outcome
# Inputs:
#   Y_true - matrix list - outcomes for all subgroups as integer classes
#   Z - matrix list - Estimated Z^s matrices from HIP
#   theta - matrix - Estimated theta matrix from HIP
# Outputs:
#   float - classification accuracy
def train_class(Y_true, Z, theta):
    preds = calc_probs(torch.mm(torch.cat(Z), theta), 1)
    preds_cat = torch.argmax(preds, dim=1)
    return torch.true_divide(torch.sum(torch.cat(Y_true) == preds_cat),len(preds_cat))
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# test_class: Calculates test classification accuracy for multiclass outcome
# Inputs:
#   Y_true - matrix list - outcomes for all subgroups as integer classes from test data
#   X - matrix list - X^d,s matrices containing the covariates from the test data
#   B - matrix list - Estimated B^d,s matrices from HIP
#   theta - matrix - Estimated theta matrix from HIP
#   standardize - boolean - Whether to standardize the X matrices before using them in prediction
# Outputs:
#   float - classification accuracy
def test_class(Y_true, X, B, theta, standardize):
    D = len(X)
    S = len(X[0])

    if standardize:
        for d in range(D):
            for s in range(S):
                x_mean = torch.mean(X[d][s], dim=0)
                x_std = torch.std(X[d][s], dim=0)
                X[d][s] = torch.div((X[d][s] - x_mean), x_std)

    Xcat = [[[] for d in range(D)] for s in range(S)]
    Bcat = [[[] for d in range(D)] for s in range(S)]
    Z_pred = [[] for s in range(S)]
    for s in range(S):
        Xcat[s] = X[0][s]
        Bcat[s] = B[0][s]
    for s in range(S):
        for d in range(1,D):
            Xcat[s] = torch.cat((Xcat[s], X[d][s]), dim=1)
            Bcat[s] = torch.cat((Bcat[s], B[d][s]))
        Z_temp = Xcat[s] @ Bcat[s] @ torch.inverse(Bcat[s].T @ Bcat[s] + (0.0001*torch.eye(Bcat[s].shape[1])))
        norm_z = torch.norm(Z_temp, p='fro')
        Z_pred[s] = torch.div(Z_temp, norm_z)
    return train_class(Y_true, Z_pred, theta)
#----------------------------------------------------------------------------------
