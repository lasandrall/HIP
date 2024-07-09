#' Fit HIP models across multiple lambda values
#'
#' `select_lambda` searches across lambda values (either grid search or random) and fits HIP models
#' given training data. The best model is chosen based on selection criteria such as cross-validation,
#' BIC, or AIC. Results can then be used for model evaluation, prediction, and plotting.
#'
#'
#' @usage select_lambda(X, Y, gamma, family, topn, ncore=NA, K=NULL, k_thresh=0.2, update_thresh=10^-5,
#'               epsilon=10^-4, max_iter=50, search="random", xi_range=c(0,2),
#'               g_range=c(0,2), num_steps=c(8,8), standardize="subgroup",
#'               std_type="scale_center", std_x=TRUE, std_y=TRUE, verbose=FALSE,
#'               selection_criterion="eBIC_0", folds=5)
#'
#' @param X matrix list - list of X^d,s matrices containing covariates
#' @param Y matrix list - list of Y^s matrices containing outcomes
#' @param gamma list<double> - indicators of whether to penalize each data view; should be length D
#' @param family string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'
#' @param topn int or list<int> - number of variables to retain; different values may be specified for each view using a list of length D
#' @param ncore int - number of cores to use in parallel processing; default is half of available cores
#' @param K int - number of latent components; will use 'select_K_simple' to choose if not provided
#' @param k_thresh - double - threshold to use for 'select_K_simple' if K not provided; default = 0.2
#' @param epsilon double - threshold to use for overall convergence criterion; default = 10**-4
#' @param max_iter int - maximum number of outer loop iterations; default = 50
#' @param search string - what type of search to perform for lambda parameter; default = "random"
#' - "random" - tries a random selection of rand_prop*num_steps lambda values in grid
#' - "grid" - tries all lambda values in grid
#' @param xi_range list<double> - minimum  and maximum values to consider for lambda_xi; default = `c(0.0, 2.0)`
#' @param g_range list<double> - minimum  and maximum values to consider for lambda_g; default = `c(0.0, 2.0)`
#' @param num_steps list<int> - list of two integers; the first is the number of steps for lambda_xi (default = 8) and the second is the number of steps for
#' lambda_g. Together these define the number of steps to use in lambda grid
#' @param standardize string - One of "all", "subgroup", or "none"; default = "subgroup"
#' @param std_type string - One of "scale_center", "center", or "norm"; default = "scale_center"
#' @param std_x boolean - indicates whether to standardize X; default = True
#' @param std_y boolean - indicates whether to standardize Y; default = True
#' @param verbose whether to print additional information during optimization; default = False
#' @param selection_criterion string - criterion to use for selecting the best model; one of 'CV', 'BIC', 'AIC', 'eBIC_0', 'eBIC_5', or 'eBIC_1'. eBIC_0 by default.
#' @param folds int - number of folds to use if CV is selected as the selection criterion; default = 5
#'
#' @returns The output is quite large, but most items do not need to be accessed directly by the user and instead are accessed by functions such as `HIP_train_eval`
#' or `HIP_pred`. First, `select_lambda` returns a nested list called `out` which contains the following:
#' \item{`search_results`}{list - list with results returned from the optimize_torch Python function for each lambda value tried (see below for detailed output information)}
#' \item{`total_time`}{double - time to complete entire search in seconds}
#' \item{`xi_range`}{list<double> - minimum and maximum values considered for lambda_xi}
#' \item{`g_range`}{list<double> - minimum and maximum values considered for lambda_g}
#' \item{`num_steps`}{dict - number of steps used in lambda grid}
#' \item{`search`}{string - type of search performed for selecting lambda parameters }
#'
#' select_lambda also returns the following:
#' \item{`best_index`}{int - index of the best model chosen by the selection criterion }
#' \item{`criterion`}{string - criterion for model selection: one of 'CV', 'BIC', 'AIC', 'eBIC_0', 'eBIC_5', or 'eBIC_1'. eBIC_0 by default.}
#' \item{`topn`}{int or list<int> - number of variables retained}
#' \item{`standardize`}{string - stores option used to standardize data}
#' \item{`family`}{string - stores family label for outcome data}
#'
#' `search_results` is a large list containing results from fit models. First, it contains
#' multiple fit models which have the following items:
#' \item{`full`}{list - results returned from optimize_torch on full data}
#' \item{`include`}{list<tensor> - list of length D with a 1 indicating the variable was included in subset fit and 0 indicating not included in subset fit}
#' \item{`subset`}{list - results returned from optimize_torch on subset of variables}
#'
#' `full` and `subset` are large lists which contain the following:
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
#' dat_gen <- generate_data()
#'
#' res <- select_lambda(dat_gen$X, dat_gen$Y, c(1,1), 'gaussian', 50)
#'
select_lambda <- function(X, Y, gamma, family, topn, ncore=NA, K=NULL, k_thresh=0.2, update_thresh=10^-5,
                          epsilon=10^-4, max_iter=50, search="random",
                          xi_range=c(0,2), g_range=c(0,2), num_steps=c(8,8),
                          standardize="subgroup", std_type="scale_center",
                          std_x=TRUE, std_y=TRUE, verbose=FALSE, selection_criterion="eBIC_0",
                          folds=5){

  if(!(selection_criterion %in% c("CV", "BIC", "AIC", "eBIC_0", "eBIC_5", "eBIC_1"))){
    stop("Invalid selection criterion")
    }

  if(is.na(ncore)){
    num_core <- parallel::detectCores()
    ncore <- as.integer(ceiling(num_core/2))
  }

  res <- list()

  if(selection_criterion == "CV"){
    # Import Python functions
    reticulate::source_python(system.file("python/all_functions.py",
                                          package = "HIP"))

    num_steps_Xi <- as.integer(num_steps[[1]])
    num_steps_G <- as.integer(num_steps[[2]])
    topn <- as.integer(topn)
    ncore <- as.integer(ncore)
    max_iter <- as.integer(max_iter)
    folds <- as.integer(folds)

    if(is.null(K)){
      K_choose <- select_K_simple(X=X, threshold=k_thresh, verbose=TRUE)
      K <- K_choose$kchoose
      K <- as.integer(K)
    }else{
      K <- as.integer(K)
    }


    # Running the function from Python script and returning output to R session
    res$out <- select_lambda_CV(Y=Y, X=X, gamma=gamma, family=family, ncore=ncore,
                                topn=topn, K=K, max_iter=max_iter, folds=folds, search=search,
                                num_steps=list("Xi"=num_steps_Xi, "G"=num_steps_G),
                                xi_range = xi_range, g_range=g_range, standardize=standardize,
                                std_type=std_type, std_x=std_x, std_y=std_y,
                                verbose=verbose)

    # Get index of best result
    res$best_index <- get_best_R(res$out$search_results, selection_criterion, family)

    # Also store criterion and topn used in the results
    res$criterion <- selection_criterion
    res$topn <- topn
    res$standardize <- standardize
    res$family <- family

    return(res)
  } else{
    reticulate::source_python(system.file("python/all_functions.py",
                                          package = "HIP"))

    np <- reticulate::import("numpy")

    num_steps_Xi <- as.integer(num_steps[[1]])
    num_steps_G <- as.integer(num_steps[[2]])
    topn <- as.integer(topn)
    ncore <- as.integer(ncore)
    max_iter <- as.integer(max_iter)

    if(is.null(K)){
      K_choose <- select_K_simple(X=X, threshold=k_thresh, verbose=TRUE)
      K <- K_choose$kchoose
      K <- as.integer(K)
    }else{
      K <- as.integer(K)
    }

    # Running the function from Python script and returning output to R session
    res$out <- select_lambda(Y=Y, X=X, gamma=gamma, family=family, ncore=ncore,
                             topn=topn, K=K, max_iter=max_iter, search=search,
                             num_steps=list("Xi"=num_steps_Xi, "G"=num_steps_G),
                             xi_range = xi_range, g_range=g_range, standardize=standardize,
                             std_type=std_type, std_x=std_x, std_y=std_y,
                             verbose=verbose)

    # Get index of best result
    res$best_index <- get_best_r(res$out$search_results, selection_criterion, family)

    # Also store criterion and topn used in the results
    res$criterion <- selection_criterion
    res$topn <- topn
    res$standardize <- standardize
    res$family <- family


    return(res)

  }
}
