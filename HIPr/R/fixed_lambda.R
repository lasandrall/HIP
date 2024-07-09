#' Fits a HIP model given lambda values
#'
#' `fixed_lambda` fits a model from X, Y data and fixed \eqn{\lambda_\xi}, \eqn{\lambda_G}
#' supplied by the user.
#'
#' @usage fixed_lambda(X, Y, gamma, family, topn, lambda_xi=1, lambda_g=1, K=NULL,
#' k_thresh=0.2, update_thresh=10^-5, epsilon=10^-4, max_iter=50,
#' standardize='subgroup', std_type='scale_center', std_x=TRUE,
#' std_y=TRUE, verbose=FALSE)
#'
#' @param X matrix list - list of X^d,s matrices containing covariates
#' @param Y matrix list - list of Y^s matrices containing outcomes
#' @param family string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#' @param topn int or list<int> - number of variables to retain; different values may be specified for each view using a list of length D
#' @param lambda_xi double - value of lambda_xi in penalty term
#' @param lambda_g double - value of lambda_g in penalty term
#' @param K int - number of latent components; will use 'select_K_simple' to choose if not provided
#' @param k_thresh double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#' @param update_thresh double - threshold to use for suboptimization convergence criterion; default = 10**-5
#' @param epsilon double - threshold to use for overall convergence criterion; default = 10**-4
#' @param max_iter int - maximum number of outer loop iterations; default = 50
#' @param standardize string - One of "all", "subgroup", or "none"; default = "subgroup"
#' @param std_type string - One of "scale_center", "center", or "norm"; default = "scale_center"
#' @param std_x boolean - indicates whether to standardize X; default = TRUE
#' @param std_y boolean - indicates whether to standardize Y; default = TRUE
#' @param verbose boolean - whether to print additional information during optimization; default = FALSE
#'
#' @returns
#' \item{`full`}{list - results returned from optimize_torch on full data }
#' \item{`include`}{list<tensor> - list of length D with a 1 indicating the variable
#' was included in subset fit and 0 indicating not included in subset fit }
#' \item{`subset` }{list - results returned from optimize_torch on subset of variables}

#'
#' In addition, `fixed_lambda` returns -99 for `best_index` and NA for `criterion`
#' to indicate they are not applicable to this case. topn is also returned.
#'
#' @examples
#' # Generating data
#' dat_gen <- generate_data()
#'
#' # Getting model from fixed_lambda
#' res <- fixed_lambda(dat_gen$X, dat_gen$Y, c(1,1), 'gaussian', 50)
#'
fixed_lambda <- function(X, Y, gamma, family, topn, lambda_xi=1, lambda_g=1, K=NULL,
                             k_thresh=0.2, update_thresh=10^-5, epsilon=10^-4, max_iter=50,
                             standardize='subgroup', std_type='scale_center', std_x=TRUE,
                             std_y=TRUE, verbose=FALSE){
  res <- list()

  # Import Python functions
  library(reticulate)
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))

  topn <- as.list(topn)
  topn <- as.integer(topn)

  res$out <- optimize_ranking(Y, X, lambda_xi, lambda_g, gamma, family, topn, K,
                              k_thresh, update_thresh, epsilon, max_iter, standardize,
                              std_type, std_x, std_y, verbose)

  # to indicate not applicable to this case
  res$best_index <- -99
  res$criterion <- 'N/A'
  res$topn <- topn

  return(res)
}
