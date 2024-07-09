#' Make predictions from HIP model
#'
#' Outputs predicted Y values given X data and results from `select_lambda` or `fixed_lambda`.
#'
#' @usage HIP_pred(X_dat, res, fix_or_search, standardize="no")
#'
#' @param X_dat matirx list - X^d,s matrices from `generate_data` or `format_data`
#' @param res output from `select_lambda` or `fixed_lambda`
#' @param fix_or_search string - 'fixed' if `res` comes from `fixed_lambda`, else 'search'
#' @param standardize string - 'yes' to standardize Y data if it was standardized in `select_lambda`, else
#' return unstandardized predicted Y values
#'
#' @returns The function returns a list of \eqn{S} tensors containing predictions for
#' observations in each subgroup.
#'
#' @examples
#' # Generate data
#' dat_gen_test <- generate_data(test_data=TRUE)
#'
#' # Get results from select_lambda
#' res <- select_lambda(dat_gen_test$X, dat_gen_test$Y, c(1,1), 'gaussian', 50,
#'                      K = 2, num_steps=c(4,4))
#'
#' # Make predictions on X test data (and standardize predictions)
#' Y_pred <- HIP_pred(dat_gen_test$X_test, res, 'search', 'yes')
#'
HIP_pred <- function(X_dat, res, fix_or_search, standardize="no"){

  # Import Python functions
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))
  torch <- reticulate::import("torch")

  D <- length(X_dat)
  S <- length(X_dat[[1]])

  if(fix_or_search == "fixed"){
    res_obj <- res$out
  } else {
    res_obj <- res$out$search_results[[res$best_index]]
  }

  # X variables used in subset fit
  Xsub <- lapply(1:D, function(d){
    lapply(1:S, function(s){
      X_dat[[d]][[s]][,torch$eq(res_obj$include[[d]], 1)]
    })
  })

  # Get predictions based on the family
  Y_pred <- switch(res$family,
                "gaussian" = pred_mse(X_test = Xsub,
                                      B = res_obj$subset$B,
                                      theta_dict = list("theta" = res_obj$subset$theta,
                                                        "beta" = res_obj$subset$beta)),
                "multiclass" = pred_multiclass(X_test = Xsub,
                                          B = res_obj$subset$B,
                                          theta_dict = list("theta" = res_obj$subset$theta,
                                                            "beta" = res_obj$subset$beta)),
                "poisson" = pred_poisson(X_test = Xsub,
                                      B = res_obj$subset$B,
                                      theta_dict = list("theta" = res_obj$subset$theta,
                                                        "beta" = res_obj$subset$beta)),
                "zip" = pred_zero_poisson(X_test = Xsub,
                                 B = res_obj$subset$B,
                                 theta_dict = list("theta" = res_obj$subset$theta,
                                                   "beta" = res_obj$subset$beta,
                                                   "tau" = res_obj$subset$tau))
  )

  if(standardize=="yes" & res$standardize == "none"){
    print("Unable to standardize, as data was not standardized in select_lambda results")
    print("Returning unstandardized Y values")
  }

  if(res$family == "gaussian" & res$standardize != "none" & standardize == "yes"){
    std_dat <- standardize_dat(Y = Y_pred, standardize = res$standardize, std_type = "scale_center", std_y = T)
    Y_pred <- std_dat$Y
  }else{
    Y_pred <- Y_pred
  }

  return(Y_pred)
}
