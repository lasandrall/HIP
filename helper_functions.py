#!~/Python/env/bin/python3

# helper_functions.py
# Contains functions not meant to be called by the user.
# Author: Jessica Butts and Sandra Safo
# Last Updated: August 2023

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
        loss = torch.sum(-1*Y[:,0].reshape(-1,1)*(torch.log(Y[:,1]).reshape(-1,1) + (beta + Z.matmul(theta))) + Y[:,1].reshape(-1,1)*torch.exp(beta + Z.matmul(theta)) + torch.log(special.factorial(Y[:,0].reshape(-1,1))))
    
    else:
        loss = 0
        S = len(Y)
        for s in range(S):
            loss += torch.sum(-1*Y[s][:,0].reshape(-1,1)*(torch.log(Y[s][:,1]).reshape(-1,1) + (beta + Z[s].matmul(theta))) + Y[s][:,1].reshape(-1, 1)*torch.exp(beta + Z[s].matmul(theta)) + torch.log(special.factorial(Y[s][:,0].reshape(-1,1))))
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
    return torch.sum(Y[:,0].reshape(-1,1)*(torch.log(Y[:,1]).reshape(-1,1) + torch.log(Ypred)) -
                     Y[:,1].reshape(-1,1)*Ypred - 
                     torch.log(special.factorial(Y[:,0].reshape(-1,1))))
    
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
    y_all = torch.log(special.factorial(Y[:,0].reshape(-1,1))) + torch.log(1 + torch.exp(tau))
    
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
