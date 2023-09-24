# HIP_examples.py: Generate example data and run HIP.
# Author: Jessica Butts
# Date: August 2023

# Read in functions - replace path with location of files
#exec(open("path/main_functions.py").read())
#exec(open("path/helper_functions.py").read())
#exec(open("path/adagrad_functions.py").read())

# set up simulation parameters
family = 'poisson'

seed = 1
seed_const = 500
nonzero = 50
offset = 25
n = [250,260]
K = 2
S = 2
D = 2
p = [300,350]
theta_init = torch.tensor(([0.7], [0.2]))
beta_init = 2.0
sigma_x = 1.0
sigma_y = 1.0
z_mean = 25.0
z_sd = 3.0

theta_dict = {'gaussian': torch.tensor(([0.7],[0.2])),
              'multiclass': torch.tensor(([[1.0,  0.5], [0.2, 0.8]])),
              'poisson': torch.tensor(([0.7],[0.2])),
              'zip': torch.tensor(([0.7],[0.2]))
}

beta_dict = {'gaussian': 2.0,
             'multiclass': torch.tensor(([[0.5,  0.5]])),
             'poisson': 2.0*torch.ones((1,1)),
             'zip': 2.0*torch.ones((1,1))
}

#-----------------------------------------------------------------------
# Generate Data based on `family`
#-----------------------------------------------------------------------
# Generate common Z and B matrices to use in train and test data
dat_all = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family=family, sigma_x=sigma_x, sigma_y=sigma_y, theta_init=theta_dict[family], beta=beta_dict[family], z_mean=z_mean, z_sd=z_sd)

# Generate training data
dat_train = generate_data(seed=seed, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family=family, sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])

# Generate testing data
dat_test = generate_data(seed=seed+seed_const, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family=family, sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])


#-----------------------------------------------------------------------
# Generate multiple continuous outcomes
# `q` controls how many outcomes are generated
#-----------------------------------------------------------------------
family = 'gaussian'
# Generate common Z and B matrices to use in train and test data
dat_all = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family='gaussian', q=3, sigma_x=sigma_x, sigma_y=sigma_y, z_mean=z_mean, z_sd=z_sd)

# Generate training data
dat_train = generate_data(seed=seed, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family='gaussian', q=3, sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])

# Generate testing data
dat_test = generate_data(seed=seed+seed_const, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, q=3, family='gaussian', sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])


#-----------------------------------------------------------------------
# Generate multiclass outcomes with > 2 classes
# `m` controls how many classes are generated
#-----------------------------------------------------------------------
family = 'multiclass'
# Generate common Z and B matrices to use in train and test data
dat_all = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family='multiclass', m=3, sigma_x=sigma_x, sigma_y=sigma_y, z_mean=z_mean, z_sd=z_sd, theta_init = torch.tensor(([[0.6,  0.5, 0.3], [0.1, 0.2, 0.4]])))

# Generate training data
dat_train = generate_data(seed=seed, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, family='multiclass', m=3, sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])

# Generate testing data
dat_test = generate_data(seed=seed+seed_const, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, offset=offset, m=3, family='multiclass', sigma_x=sigma_x, sigma_y=sigma_y, theta_init=dat_all['theta'], beta=dat_all['beta'], B=dat_all['B'], Z=dat_all['Z'])


#-----------------------------------------------------------------------
# Select tuning parameters
#-----------------------------------------------------------------------

# Using BIC
res = select_lambda(Y = dat_train['Y'], X = dat_train['X'], topn = nonzero,
              gamma = [1.0 for d in range(D)], family = family, K = K, ncore = 2,
              search = 'random', rand_prop = 0.20,
              xi_range = [0.0, 2.0], g_range = [0.0, 2.0], num_steps = {'Xi': 4, 'G': 4},
              verbose = False)
# best model based on BIC
best = get_best(res['search_results'], 'BIC')


# Using 5-fold CV
res = select_lambda_CV(Y = dat_train['Y'], X = dat_train['X'], topn = nonzero,
              gamma = [1.0 for d in range(D)], family = family, K = K, ncore = 2,
              search = 'random', rand_prop = 0.20,
              xi_range = [0.0, 2.0], g_range = [0.0, 2.0], num_steps = {'Xi': 4, 'G': 4},
              verbose = False)
# best model based on CV
best = get_best(res['search_results'], 'cv_sub')
    

#-----------------------------------------------------------------------
# Some model results
#-----------------------------------------------------------------------
### Training error
best['res']['subset']['train_err']['comp_val']

# Assign appropriate error functions for the given family
if family == 'gaussian':
    train_error = train_mse
    test_error = test_mse
    std_y = True # may need to update this
    
elif family == 'multiclass':
    train_error = train_class
    test_error = test_class
    std_y = False # should never standardize Y when multiclass
    
elif family == 'poisson':
    train_error = train_pois
    test_error = test_pois
    std_y = False # should never standardize Y when poisson
            
elif family == 'zip':
    train_error = train_zip
    test_error = test_zip
    std_y = False # should never standardize Y when ZIP

### Test error
# Standardize testing data
#  NOTE: std_y should be False for multiclass, poisson, and zip families
dat_test_std = standardize_dat(Y = dat_test['Y'], X = dat_test['X'],
                               Y_train = dat_train['Y'], X_train = dat_train['X'],
                               standardize = 'subgroup', std_type = 'scale_center', std_y = std_y)
                               
# variables included in subset model
Xsub = [[dat_test_std['X'][d][s][:, best['res']['include'][d].eq(1)]  for s in range(S)] for d in range(D)]
test = test_error(Y_test = dat_test_std['Y'], X_test = Xsub, B=best['res']['subset']['B'],
                  theta_dict = {'theta': best['res']['subset']['theta'],
                                'beta':  best['res']['subset']['beta'],
                                'tau': best['res']['subset']['tau']})
test['comp_val']

# plot of loadings
l = top_loadings(best['res']['subset']['B'], top_n = [nonzero for d in range(D)], plot = {'nonzero':nonzero, 'offset':offset})


#-----------------------------------------------------------------------
# Save results
#-----------------------------------------------------------------------
# Save results file to same directory as code
with open(''.join(('results.txt')), 'wb') as filename:
    pickle.dump(res, filename)
    
