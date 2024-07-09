#' Standardizes input data
#'
#' Helper function used to standardize testing data from training data.
#'
#' @param standardize string - One of "all", "subgroup", or "none"
#' @param std_type string - One of "scale_center", "center", or "norm"
#' @param X matrix list - list of X^d,s matrices containing covariates
#' @param Y matrix list - list of Y^s matrices containing outcomes
#' @param X_train matrix list - list of X^d,s matrices whose mean and sd will be used to standardize X
#' @param Y_train matrix list - list of Y^s matrices whose mean andd sd will be used to standardize Y
#' @param std_x boolean - indicates whether to standardize X; default = True
#' @param std_y boolean - indicates whether to standardize Y; default = True
#'
#' @returns Returns a nested list with the following elements:
#' | `X` | matrix list - standardized X^d,s if standardization was requested |
#' | - | - |
#' | `Y` | matrix list - standardized Y^s if standardization was requested |
#'

standardize_dat=function(standardize, std_type, X=NULL, Y=NULL, X_train=NULL,
                         Y_train=NULL, std_x=TRUE, std_y=TRUE, verbose=TRUE){
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))

  standardized <- standardize_dat(standardize, std_type, X, Y, X_train,
                                  Y_train, std_x, std_y, verbose)
  return(standardized)
}
