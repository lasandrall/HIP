# HIP_examples.py: Generate example data and run HIP.
# Author: Jessica Butts
# Date: June 24, 2021

# Read in functions
#exec(open("path/main_functions.py").read())
#exec(open("path/helper_functions.py").read())

# set up simulation parameters
nonzero = 50
overlap = 25
standardize = True
sigma_x = .2
sigma_y = .5
n = [250,260]
K = 2
S = 2
D = 2
p = [300,350]
theta_init = torch.tensor(([1.], [0.]))

#-----------------------------------------------------------------------
# Generate single continuous outcome
#-----------------------------------------------------------------------
# Generate the training data
X, Y, Z, B, theta = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='gaussian', theta_init=theta_init)
    
# Call lambda selection process
# With the gamma parameter, we do penalize both data sets.
# We use the defaults for the optional parameters.
results=select_lambda(Y, X, gamma=[1,1], family='gaussian')
        
# Generate a set of test data
# We pass in the same theta used to generate the training data using the optional theta_init parameter.
X2, Y2, Z2, B2, theta2 = generate_data(seed=20, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='gaussian', theta_init=theta)

# Calculate the test mse using the test data
each, overall = test_mse(Y_true=Y2, X=X2, B=results['B'], theta=results['theta'], standardize=standardize)
results['test_by_var'] = each
results['test_overall'] = overall

# Save results file to same directory as code
with open(''.join(('results.txt')), 'wb') as filename:
    pickle.dump(results, filename)
    
#-----------------------------------------------------------------------
# Generate multiple continuous outcomes
#-----------------------------------------------------------------------
# Generate the training data
# Add the optional parameter q to generate 3 outcome variables
X, Y, Z, B, theta = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='gaussian', q=3)
    
# Call lambda selection process
# With the gamma parameter, we penalize both data sets.
# We can change the number of steps with num_steps
results2=select_lambda(Y, X, gamma=[1,1], family='gaussian', num_steps=5)
        
# Generate a set of test data
# We pass in the same theta used to generate the training data using the optional theta_init parameter.
X2, Y2, Z2, B2, theta2 = generate_data(seed=100, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='gaussian', theta_init=theta, q=3)

# Calculate the test mse using the test data
each, overall = test_mse(Y_true=Y2, X=X2, B=results2['B'], theta=results2['theta'], standardize=standardize)
results2['test_by_var'] = each
results2['test_overall'] = overall

# Save results file to same directory as code
with open(''.join(('results2.txt')), 'wb') as filename:
    pickle.dump(results2, filename)

#-----------------------------------------------------------------------
# Generate single multiclass outcome
#-----------------------------------------------------------------------
# Generate the training data
# By default, the data will have m=2 classes. This can be changed with the optional parameter m.
X, Y, Z, B, theta = generate_data(seed=0, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='multiclass')
    
# Call lambda selection process
# With the gamma parameter, we penalize both data sets.
# We use the defaults for the optional parameters.
results3=select_lambda(Y, X, gamma=[1,1], family='multiclass')
        
# Generate a set of test data
# We pass in the same theta used to generate the training data using the optional theta_init parameter.
X2, Y2, Z2, B2, theta2 = generate_data(seed=100, n=n, p=p, K=K, D=D, S=S, nonzero=nonzero, overlap=overlap, sigma_x=sigma_x, sigma_y=sigma_y, family='multiclass', theta_init=theta)

# Calculate the test mse using the test data
results3['test_acc'] = test_class(Y_true=Y2, X=X2, B=results3['B'], theta=results3['theta'], standardize=standardize)

# Save results file to same directory as code
with open(''.join(('results3.txt')), 'wb') as filename:
    pickle.dump(results3, filename)
