Heterogeneity in Integration and Prediction (HIP) README
August 2023

1.This package depends on the following Python modules:
- torch
- numpy
- random
- pickle
- itertools
- copy
- math
- time
- joblib
- matplotlib
- scipy

Please ensure these modules are downloaded and available for use.

2. Please see HIP_examples.py for examples. The results files can be read through R using the following commands:

library(reticulate) 	# Needed to call the py_load_object function.
result <- py_load_object("path/filename.txt") 	#replace "path" and "filename" with the actual path and file name.

3. This Python code is for realizing the HIP algorithm proposed in the following paper.
Please cite this paper if you use the code for your research purpose.

Butts J, Wendt C, Bowler R, Hersh C P, Long Q, Eberly L, and Safo S. "HIP: a method for high-dimensional multi-view data integration and prediction accounting for subgroup heterogeneity". Under Review.

4.Please send your comments and bugs to ssafo@umn.edu.
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%HIP_examples.py: You can do the following:
% 1. Can use generate_data to generate data under differing scenarios
% 2. Can use optimize_ranking for any fixed tuning parameter value to 
%    obtain estimates of B, G, Xi, Z, theta, and beta
% 3. Can use select_lambda or select_lambda_CV to select optimal tuning parameters 
%    using a grid or random search
% 4. Can use appropriate function based on the type of outcome 
%    to calculate training error after fitting a model 
% 5. Can use appropriate function based on the type of outcome  
%    to calculate testing error after fitting a model 
%--------------------------------------------------------------------------



