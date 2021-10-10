# HIP
Epidemiologic and genetic studies for many complex diseases suggest subgroup disparities (e.g., by sex). We consider this problem from the standpoint of integrative analysis where we combine information from different views (e.g., genomics, proteomics, clinical data). Existing integrative analysis methods ignore the heterogeneity in subgroups, and stacking the views and accounting for subgroup heterogeneity does not model the association among the views. We propose a statistical approach for joint association and prediction that leverages the strengths in each view to identify variables that are shared by and specific to different subgroups (e.g., males and females) and that contribute to the variation in the outcome of interest. HIP (Heterogeneity in Integration and Prediction) accounts for subgroup heterogeneity, allows for sparsity in variable selection, is applicable to multi-class and to univariate or multivariate continuous outcomes, and incorporates covariate adjustment. We develop efficient algorithms in PyTorch. 
