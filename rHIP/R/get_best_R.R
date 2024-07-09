#' Helper function to get best result from select_lambda
#'
#' 'get_best_R' is used within the select_lambda function to get the best result based on
#' model performance. Not generally called by the user.
#'
#' get_best_R takes in the list of results from the select_lambda tuning function, and
#' returns the index of the best result. It also takes in the criterion used in training
#' such as AIC or BIC, in addition to the family the outcome data belongs to.
#'
#' @usage get_best_r(res_list, criterion="", family)
#' @param res_list list - search_results list from the select_lambda output
#' @param criterion string - criterion for model selection: one of 'CV', 'BIC', 'AIC', 'eBIC_0', 'eBIC_5', or 'eBIC_1'
#' @param family string - family of outcome data ('gaussian', 'multiclass', etc.)
#'
#' @returns `get_best_R` returns the index of the best results from the select_lambda output
#'
#' @examples
#' # Generate data
#' dat_gen <- generate_data()
#'
#' # Get results from select_lambda
#' res <- select_lambda(dat_gen$X, dat_gen$Y, c(1,1), "gaussian", 50, K=2, num_steps=c(4,4))
#'
#' best_index <- get_best_r(res$out$search_results, "eBIC_0", "gaussian")
#'
get_best_r <- function(res_list, criterion="", family){
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))
  np <- reticulate::import("numpy")

    if(!(criterion %in% c("CV", "BIC", "AIC", "eBIC_0", "eBIC_5", "eBIC_1"))){
    stop("Invalid selection criterion")
  }

  if(criterion == 'CV'){
    crit_list <- sapply(res_list, function(x){np$double(x$cv_error$subset)})
    if(family == "gaussian"){
      # minimize MSE
      index <- which.min(crit_list)
    } else {
      # maximize classification accuracy/D^2
      index <- which.max(crit_list)
    }
  } else {
    crit_list <- sapply(res_list, function(x){np$double(x$full[[criterion]])})
    index <- which.min(crit_list)
  }
  return(index)
}
