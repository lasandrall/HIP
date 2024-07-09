#' Format data for use with HIP
#'
#' User-supplied data must be formatted properly to work with
#' HIP functions such as `select_lambda` or `HIP_test_eval`. Use this function to format the data, which will
#' return a list containing training/testing data converted to PyTorch tensors
#' and useful information such as the number of data views and subgroups. Note, either X or Y data
#' must contain the subgroup variable, and each dataframe must have an ID variable.
#' Y data should contain an outcome variable, which may belong to a Gaussian, multiclass, Poisson, or ZIP family.
#' You may exclude the Y data if you simply want to format X data for use in prediction functions such as `HIP_pred`.
#' See the example below for a demonstration of the function using a COVID-19 dataset included in the `HIP` package.
#'
#' As HIP uses functions integrated in Python, it is necessary to convert dataframes
#' to PyTorch tensors prior to use. This function also gathers training and testing
#' data together for use in functions such as `HIP_test_eval`. See individual function
#' documentation for examples. Also note that data from the `generate_data` function is
#' already formatted, so they do not need to be fed into this function.
#'
#' @usage format_data(X_train, Y_train=NULL, id_var=NULL,
#' subgroup_var=NULL, outcome_var=NULL, X_test=NULL, Y_test=NULL,
#' family="", data_source="")
#'
#' @param X_train list of dataframes - list of D dataframes containing X training data (where D is the number of data views).
#' NOTE: X_train should also contain the subgroup_var if it is not contained in the Y_train data.
#' @param Y_train list of dataframes - list of D dataframes containing Y training data
#' @param id_var string - name of ID variable in data
#' @param subgroup_var string - name of subgroup variable in data
#' @param outcome_var string - name of outcome variable in data
#' @param X_test list of dataframes - list of D dataframes containing X test data, if available
#' @param Y_test list of datamframes - list of D dataframes containing Y test data, if available
#' @param family string - family of outcome data; one of either "gaussian", "multiclass", "poisson", or "zip"
#'
#' @returns Returns a list with the following:
#' | `X` | matrix list - Contains training data X^d,s matrices; access as `data$X[[d]][[s]]`|
#' | - | - |
#' | `Y` | matrix list - Contains training data Y^s matrices; access as `data$Y[[d]][[s]]`|
#' | `X_test` | matrix list - Contains test data X^d,s matrices; access as `data$X_test[[d]][[s]]`|
#' | `Y_test` | matrix list - Contains test data Y^s matrices; access as `data$Y_test[[d]][[s]]`|
#' | `D` | int - Number of data views |
#' | `S` | int - Number of unique subgroups |
#' | `sub_vec` | int vec - Vector with labels for subgroups |
#' | `var_list` | string list - D lists containing names of variables in X data |
#'
#' @examples
#' # Read in COVID Data
#' data("covid_data")
#'
#' # Join data and declare variables for IDs, subgroups, and outcomes
#' X_train <- list(covid_data$X_train_genomic, covid_data$X_train_proteomic)
#' X_test <- list(covid_data$X_test_genomic, covid_data$X_test_proteomic)
#'
#' id_var <- 'ID'
#' subgroup_var <- 'Sex'
#' S <- length(unique(covid_data$Y_train[[subgroup_var]]))
#' outcome_var <- 'HFD45'
#' D <- 2
#'
#' # Structure data and convert it to Python format
#' formatted_data <- format_data(X_train, covid_data$Y_train, id_var, subgroup_var, outcome_var,
#'                               X_test, covid_data$Y_test, family='gaussian')
#'
format_data <- function(X_train, Y_train=NULL, id_var=NULL,
                      subgroup_var=NULL, outcome_var=NULL, X_test=NULL, Y_test=NULL,
                      family=""){

  ######## Structure Data ###########

  ## Training Data
  # Combine all data by id to ensure in same order + remove missing values
  if(!is.null(Y_train)){
    Xflat <- plyr::join_all(X_train, by = id_var, type = "inner")
    full_dat <- dplyr::full_join(Y_train, Xflat, by = id_var) %>%
      dplyr::mutate(across(!c(id_var, subgroup_var, outcome_var), as.numeric)) %>%
      na.omit()
  } else {
    Xflat <- plyr::join_all(X_train, by = id_var, type = "inner")
    full_dat <- Xflat %>%
      dplyr::mutate(across(!c(id_var, subgroup_var), as.numeric)) %>%
      na.omit()
  }

  # define subgroup values
  sub_vec <- unique(full_dat[[subgroup_var]])
  # subgroup indices
  sub_indices <- lapply(sub_vec, function(s){which(full_dat[[subgroup_var]] == s)})

  # split Y into subgroups
  if(!is.null(Y_train)){
    Y <- lapply(sub_indices,  function(s){as.numeric(full_dat[s, outcome_var])})
  } else{
    Y <- NULL
  }

  # variable names
  pd <- lapply(X_train, function(d){dplyr::setdiff(colnames(d), id_var)})

  # Set up X as list of lists
  X <- lapply(pd, function(d){lapply(sub_indices, function(s){full_dat[s,d]})})

  # If the X dataframe includes the subgroup variable, remove it (not numeric)
  if(subgroup_var %in% colnames(X)){
    X <- subset(X, select=-c(subgroup_var))
  }

  ## Testing Data
  if(!is.null(X_test) & !is.null(Y_test)){

    # Combine all data by id to ensure in same order + remove missing values
    Xflat_test <- plyr::join_all(X_test, by = id_var, type = "inner")

    if(subgroup_var %in% colnames(Xflat_test)){
      Xflat_test <- subset(Xflat_test, select=-c(eval(as.symbol(subgroup_var))))
    }

    full_dat_test <- dplyr::full_join(Y_test, Xflat_test, by = id_var) %>%
      dplyr::mutate(across(!c(id_var, subgroup_var, outcome_var), as.numeric)) %>%
      na.omit()

    # define subgroup values
    # TODO: may have issues if less than S subgroups in test data
    sub_vec_test <- unique(full_dat_test[[subgroup_var]])
    #validate(
    #  need(sum(!(sub_vec_test %in% sub_vec)) == 0, 'Test Data cannot contain any subgroups not in the training data.')
    #)

    # subgroup indices - NOTE: subgroups need to be in same order as in B^d,s and X^d,s
    sub_indices_test <- lapply(sub_vec_test, function(s){which(full_dat_test[[subgroup_var]] == s)})

    # split Y into subgroups
    Y_test <- lapply(sub_indices,  function(s){as.numeric(full_dat_test[s, outcome_var])})

    # variable names
    pd_test <- lapply(X_train, function(d){dplyr::setdiff(colnames(d), id_var)})
    #validate(
    #  need(
    #    # must have same set of variables for all d; sapply function returns 0 if sets equivalent and 1 ow; sum == 0 indicates variables same for all d
    #    sum( sapply(1:D, FUN = function(d){ !setequal(pd_test[[d]], pd[[d]])}) ) == 0,
    #    "Variables in test data must match training data")
    #)

    # Set up X as list of lists
    X_test <- lapply(pd, function(d){lapply(sub_indices_test, function(s){full_dat_test[s,d]})})
  }else{
    X_test <- NULL
    Y_test <- NULL
  }

  # If the X dataframe includes the subgroup variable, remove it (not numeric)
  for(i in 1:length(X)){
    for(j in 1:length(X[[i]])){
      if(subgroup_var %in% colnames(X[[i]][[j]])){
        X[[i]][[j]] <- subset(X[[i]][[j]], select=-c(eval(as.symbol(subgroup_var))))
      }
    }
  }

  structured_data <- list(X = X,
                          Y = Y,
                          X_test = lapply(X_test, na.omit),
                          Y_test = lapply(Y_test, na.omit),
                          S = length(sub_vec),  # should be integer
                          sub_vec = sub_vec,
                          D = length(pd),  # should be integer
                          var_list = pd)

  ###### Python Data #####

  X_test <- NULL
  Y_test <- NULL

  if(!is.null(structured_data$X_test)){
    X_test <- lapply(1:structured_data$D, function(d){lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(structured_data$X_test[[d]][[s]]))})})
  }
  if(!is.null(structured_data$Y_test)){
    Y_test <- lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(structured_data$Y_test[[s]]))})
  }

  if(!is.null(structured_data$Y)){
    # TODO: no way to specify offset at this point, so just add column of 1s
    if(family == "zip" | family=="poisson"){
      Y <- lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(cbind(structured_data$Y[[s]], 1)))})
      Y_test <- lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(cbind(structured_data$Y_test[[s]], 1)))})
    } else {
      Y <- lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(structured_data$Y[[s]]))})
      Y_test <- lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(structured_data$Y_test[[s]]))})
    }
  } else{
    Y <- NULL
  }


  rlist <- list(X = lapply(1:structured_data$D, function(d){lapply(1:structured_data$S, function(s){rTorch::torch$tensor(as.matrix(structured_data$X[[d]][[s]]))})}),
                Y = Y,
                X_test = X_test,
                Y_test = Y_test,
                D = structured_data$D,
                S = structured_data$S,
                sub_vec = structured_data$sub_vec,
                var_list=structured_data$var_list)

  return(rlist)
}
