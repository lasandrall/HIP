# HIP (Heterogeneity in Integration and Prediction)
Epidemiologic and genetic studies for many complex diseases suggest subgroup disparities (e.g., by sex). We consider this problem from the standpoint of integrative analysis where we combine information from different views (e.g., genomics, proteomics, clinical data). Existing integrative analysis methods ignore the heterogeneity in subgroups, and stacking the views and accounting for subgroup heterogeneity does not model the association among the views. We propose a statistical approach for joint association and prediction that leverages the strengths in each view to identify variables that are shared by and specific to different subgroups (e.g., males and females) and that contribute to the variation in the outcome of interest. HIP (Heterogeneity in Integration and Prediction) accounts for subgroup heterogeneity, allows for sparsity in variable selection, is applicable to multi-class and to univariate or multivariate continuous outcomes, and incorporates covariate adjustment. We develop efficient algorithms in PyTorch. 


1.This package depends on the following Python modules:
- torch
- numpy
- random
- pickle
- itertools
- copy
- math
- time

Please ensure these modules are downloaded and available for use.

2. Please run HIP_examples.py for examples. The results files can be read through R using the following commands:

library(reticulate) 	# Needed to call the py_load_object function.
result <- py_load_object("path/filename.txt") 	#replace "path" and "filename" with the actual path and file name.

3. This Python code is for realizing the HIP algorithm proposed in the following paper.
Please cite this paper if you use the code for your research purpose.

Butts J, Wendt C, Bowler R, Hersh C P, Long Q, Eberly L, and Safo S. "Accounting for data heterogeneity in integrative analysis and prediction methods: 
An application to Chronic Obstructive Pulmonary Disease" Submitted.

4.Please send your comments and bugs to ssafo@umn.edu.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

These are the main functions in this package found in main_functions.py:
%--------------------------------------------------------------------------
%optimize_torch: function to perform HIP for fixed tuning parameters
%Outputs dict with estimated B, G, Xi, Z, and theta matrices along with 
% BIC, training mse/accuracy, convergence message, and lambda parameters 
%
%DESCRIPTION:
%It is recommended to use optimize_torch to obtain lower and upper bounds for 
% the tuning parameters since too large tuning parameters will result in 
% trivial solution vector (all zeros) and too small may result in
% non-sparse solutions. 
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%select_lambda: select optimal tuning parameters for HIP by searching over 
% specified range.
%
%DESCRIPTION:
%Function selects optimal tuning parameter based on grid or random search 
% using AIC and BIC as selection criteria.
%If you want to apply optimal tuning parameters to testing data, you may
% also use optimize_torch. 
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%generate_data: Generate example data to run the method on.
%
%DESCRIPTION:
%Function generates data with specified numbers of important variables and 
% specified overlap in variables between subgroups.
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%train_mse: function to calculate the mse using training data
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%test_mse: function to calculate the mse using test data and fitted 
% model
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%train_class: function to calculate the classification accuracy using training data
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%test_class: function to calculate the classification accuracy using test data and fitted 
% model
%
%USAGE:
%See HIP_examples.py for example usage.
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%HIP_examples.py: You can do the following:
% 1. Can use generate_data to generate data under differing scenarios
% 2. Can use optimize_torch for any fixed tuning parameter value to 
%    obtain estimates of B, G, Xi, Z, and theta
% 3. Can use select_lambda to select optimal tuning parameters using a grid
%    or random search
% 4. Can use train_mse to calculate training error after fitting a model 
%    with continuous outcome(s)
% 5. Can use test_mse to calculate testing error after fitting a model 
%    with continuous outcomes(s)
% 6. Can use train_class to calculate training error after fitting a model
%    with multiclass outcome
% 7. Can use test_class to calculate training error after fitting a model
%    with multiclass outcome
%--------------------------------------------------------------------------


