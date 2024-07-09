#' Main optimization function for continuous outcomes
#'
#' This function is used to fit a HIP model given X and Y training data, a set of lambda values,
#' and gamma values. This largely exists as a helper function, as the user will more often use
#' `select_lambda` or `fixed_lambda` to fit HIP models (which then access optimize_torch).
#' However, the function is available if the user needs to access the optimization function
#' directly.
#'
#' @usage optimze_torch(X, Y, lambda_xi, lambda_g, gamma, family, K=NULL, k_thresh=0.2,
#' update_thresh=10^-5, epsilon=10^-4, max_iter=50, standardize='subgroup',
#' std_type='scale_center', std_x=TRUE, std_y=TRUE, verbose=FALSE)
#'
#' @param Y matrix list - list of Y^s matrices containing outcomes
#' @param matrix list - list of X^d,s matrices containing covariates
#' @param lambda_xi double - value of lambda_xi in penalty term
#' @param lambda_g double - value of lambda_g in penalty term
#' @param gamma list<double> - indicators of whether to penalize each data view; should be length D
#' @param family string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#' @param K int - number of latent components; will use 'select_K_simple' to choose if not provided
#' @param k_thresh double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#' @param update_thresh double - criteria for convergence in Z, G, Xi, and theta optimization functions; default = 10^-5
#' @param epsilon double - criteria for outer loop convergence; default = 10^-4
#' @param max_iter int - maximum number of outer loop iterations allowed; default = 50
#' @param standardize string - One of "all", "subgroup", or "none"; default = "subgroup"
#' @param std_type string - One of "scale_center", "center", or "norm"; default = "scale_center"
#' @param std_x boolean - indicates whether to standardize X; default = True
#' @param std_y boolean - indicates whether to standardize Y; default = True
#' @param verbose boolean - indicates whether to print additional info during run; default = False
#'
#' @returns Returns a list with the possible following elements based on status:
#' \item{`theta`}{tensor<double> - estimate of theta}
#' \item{`beta`}{tensor<double> - estimate of beta }
#' \item{`B`}{list<tensor<double>> - estimates of each B^d,s}
#' \item{`G`}{list<tensor<double>> - estimates of each G^d}
#' \item{`Xi`}{list<tensor<double>> - estimates of each Xi^d,s}
#' \item{`Z`}{list<tensor<double>> - estimates of each Z^s}
#' \item{`Lambda`}{tuple - values of lambda_xi and lambda_g used}
#' \item{`BIC`}{double - calculated BIC}
#' \item{`AIC`}{double - calculated AIC}
#' \item{`pred`}{double - prediction loss evaluated at final estimates}
#' \item{`train_err`}{nested list - list returned from function to calculate training error}
#' \item{`message`}{string - message with the status of the result; "Converged" if converged successfully, "MAX ITERS" if algorithm reached max_iter without converging.}
#' \item{`paths`}{list - history of losses until convergence}
#' \item{`iter`}{int - number of iterations to converge}
#' \item{`iter_time`}{double - time to find solution in seconds}
#' \item{`conv_criterion`}{double -  value of last convergence criterion}
#' \item{`std_x`}{boolean - whether X was standardized}
#' \item{`std_y`}{boolean - whether Y was standardized}
#'
#' @examples
#' # Generating data
#' dat_gen <- generate_data()
#'
#' # Run the optimization algorithm given the data, lambdas, and gamma
#' optimization_out <- optimize_torch(dat_gen$Y, dat_gen$X, 0.5, 0.5, c(1,1), 'gaussian', K=3)
#'
#'

optimize_torch <- function(Y, X, lambda_xi, lambda_g, gamma, family, K=NULL, k_thresh=0.2,
                           update_thresh=10^-5, epsilon=10^-4, max_iter=50, standardize='subgroup',
                           std_type='scale_center', std_x=TRUE, std_y=TRUE, verbose=FALSE){

  # Import Python functions
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))
  K <- as.integer(K)
  max_iter <- as.integer(max_iter)

  optimize_out <- optimize_torch(Y, X, lambda_xi, lambda_g, gamma, family, K, k_thresh,
                                  update_thresh, epsilon, max_iter, standardize, std_type,
                                  std_x, std_y, verbose)

  return(optimize_out)
}
