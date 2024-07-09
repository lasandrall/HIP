#' Generate simulated data to use with HIP
#'
#' `generate_data` returns a nested list containing data matrices and values used to generate the data.
#'
#' `generate_data` can be used to generate simulated multi-view data across different subgroups. The input
#' parameters specify relevant information such as the number of data views,
#' the number of subgroups, subjects desired for each subgroup, and more. The output of `generate_data`
#' gives the \eqn{X^{d,s}} data matrices and \eqn{Y^s} outcome matrices, as well as values used to
#' generate the data such as the \eqn{Z^s} and \eqn{B^{d,s}} matrices. See the HIP paper for details.
#'
#' @usage generate_data(seed=1, n=c(250,260), p=c(300,350), K=2, D=2, S=2, nonzero=50, offset=25,
#' sigma_x=1, sigma_y=1, family='gaussian', test_data=FALSE,
#' q=1, m = 2, theta_init=NULL, B=NULL, Z=NULL, y_mean=0,
#' y_sd=1, z_mean=25, z_sd=3, beta=0, tau=NULL, t_dist=FALSE, t_df=20)
#' @param seed int or string - set a seed for replicability
#' @param n int vector - number of subjects desired in each subgroup; should be length S
#' @param p int vector - number of covariates desired in each data set; should be length D
#' @param K int - number of latent components to use in generating the data
#' @param D int - number of data sets to generate
#' @param S int - number of subgroups
#' @param nonzero int - number of important variables for each subgroup; same for all subgroups
#' @param offset int - how many variables to offset the important variables between subgroups; same for all subgroups
#' @param sigma_x double - factor by which the errors added to X are scaled by
#' @param sigma_y double - factor by which the errors added to Y are scaled by; not applicable when family = 'poisson' or 'zip'
#' @param family string - determines what type of outcome is generated:
#' - 'gaussian' - q should also be specified or q will default to 1
#' - 'multiclass' - m should also be specified or m will default to 2
#' - 'poisson'
#' - 'zip'
#' @param test_data bool - TRUE to generate test data, FALSE otherwise
#' @param q int - number of continuous outcomes; can be >= 1; default = 1
#' @param m int - number of classes in multiclass outcome; can be >= 2; default = 2
#' @param theta_init  double matrix - matrix to use for theta; if None, then a theta matrix will be generated from a U(0,1) distribution
#' @param B matrix list - B^d,s matrices used to generate the data; will randomly generate if not provided
#' @param Z matrix list - Z^s matrices used to generate the data; if not provided, will generate as N(mu = z_mean, sigma = z_sd)
#' @param y_mean double - Value for mean of Y; default = 0
#' @param y_sd double - Value for sd of Y; default = 1
#' @param z_mean double - Value for mean of Y; default = 0
#' @param z_sd double - Value for sd of Y; default = 1
#' @param beta double - Value of intercept term; default = 0
#' @param t_dist boolean - if family = "gaussian", setting t_dist = TRUE will generate data with errors drawn from a t-distribution with some specified degrees of freedom
#' @param t_df int - degrees of freedom for t-distribution if t_dist = TRUE; default = 20
#'
#' @returns `generate_data` returns a nested list object with the following elements:
#'
#' | `X` | matrix list - Contains X^d,s matrices; access as `data$X[[d]][[s]]`|
#' | - | - |
#' | `Y` | matrix list - Contains Y^s matrices; access as `data$Y[[d]][[s]]`|
#' | `X_test` | matrix list - Contains test data X^d,s matrices; access as `data$X_test[[d]][[s]]`|
#' | `Y_test` | matrix list - Contains test data Y^s matrices; access as `data$Y_test[[d]][[s]]`|
#' | `D` | int - Number of data sets generated |
#' | `S` | int - Number of subgroups |
#' | `sub_vec` | int vec - Vector with labels for subgroups (from 1 to S) |
#' | `Z` | matrix list - Contains all Z^s matrices used to generate the data; access as `data[[Z]][[d]][[s]]`|
#' | `B` | matrix list - Contains all B^d,s matrices used to generate the data; access as `data[[B]][[d]][[s]]`|
#' | `theta` | matrix - Contains the theta matrix used in generating the data; access as `data$theta` |
#' | `beta` | double - Value for intercept |
#' | `tau` | double - If family = 'zip', proportion of observations in zero state with default = 0.25; None ow |
#' | `y_mean` | double - Value for mean of Y |
#' | `y_sd` | double - Value for sd of Y |
#' | `z_mean` | double - Value for mean of Z |
#' | `z_sd` | double - Value for sd of Z |
#' | `seed` | int or string - Value of seed used to generate the data |
#' | `var_list` | vec list - Contains D vectors with labels for variables in each dataset (1 to p^d) |
#'
#' @examples
#' # The values below are the function defaults, but can be changed for other simulated datasets
#' family <- 'gaussian'
#' seed <- 1
#' nonzero <- 50
#' offset <- 25
#' n <- c(250,260)
#' K <- 2
#' S <- 2
#' D <- 2
#' p <- c(300,350)
#' sigma_x <- 1.0
#' sigma_y <- 1.0
#'
#' # Generating data (no test data)
#' data_gen <- generate_data(seed, n, p, K, D, S, nonzero, offset, sigma_x,
#' sigma_y, family)
#'
#' # Generating data (with test data)
#' data_gen_test <- generate_data(seed, n, p, K, D, S, nonzero, offset,
#'                               sigma_x, sigma_y, family, test_data=TRUE)
#'
generate_data=function(seed=1, n=c(250,260), p=c(300,350), K=2, D=2, S=2, nonzero=50, offset=25,
                       sigma_x=1, sigma_y=1, family='gaussian', test_data=FALSE,
                       q=1, m = 2, theta_init=NULL, B=NULL, Z=NULL, y_mean=0,
                       y_sd=1, z_mean=25, z_sd=3, beta=0, tau=NULL, t_dist = FALSE, t_df = 20){

  if(!test_data){
    # Import Python functions
    reticulate::source_python(system.file("python/all_functions.py",
                                          package = "HIP"))

    # Conforming values to work with Python script
    seed <- as.integer(seed)

    n <- as.list(n)
    n <- as.integer(n)
    n_py <- reticulate::r_to_py(n)

    # var_list <- lapply(p, function(h){1:h})

    # generate variable name vectors for each Di
    var_list <- lapply(1:D, function(i) {
      sapply(1:p[i], function(j) paste0("D", i, "_v", j))
    })

    p <- as.list(p)
    p <- as.integer(p)
    p_py <- reticulate::r_to_py(p)

    m <- as.integer(m)

    K <- as.integer(K)
    D <- as.integer(D)
    S <- as.integer(S)
    nonzero <- as.integer(nonzero)
    offset <- as.integer(offset)

    q <- as.integer(q)
    t_df <- as.integer(t_df)

    # Running the function from Python script and returning output to R session
    gendat_all <- generate_data(seed=0L, n=n, p=p, K=K, D=D, S=S,
                                   nonzero=as.integer(nonzero), offset=offset,
                                   sigma_x=sigma_x, sigma_y=sigma_y, family=family,
                                   q = q, m = m, z_mean=25, z_sd=3, theta_init = theta_init, beta = rTorch::torch$tensor(beta),
                                   t_dist = t_dist, t_df = t_df)

    gendat_train <- generate_data(seed=seed, n=n, p=p, K=K, D=D, S=S,
                                     nonzero=nonzero, offset=offset,
                                     sigma_x=sigma_x, sigma_y=sigma_y, family=family,
                                     theta_init = gendat_all$theta, beta = gendat_all$beta, tau = gendat_all$tau,
                                     B = gendat_all$B, Z = gendat_all$Z, y_mean = gendat_all$y_mean, y_sd = gendat_all$y_sd,
                                     t_dist = t_dist, t_df = t_df)


    return(list(X=gendat_train$X,
                Y=gendat_train$Y,
                X_test=NULL,
                Y_test=NULL,
                D=D,
                S=S,
                sub_vec=1:S,
                Z=gendat_all$Z,
                B=gendat_all$B,
                theta=gendat_all$theta,
                beta=gendat_all$beta,
                tau=gendat_all$tau,
                y_mean=gendat_all$y_mean,
                y_sd=gendat_all$y_sd,
                z_mean=gendat_all$z_mean,
                z_sd=gendat_all$z_sd,
                seed=gendat_all$seed,
                var_list=var_list))

  } else {
    # Import Python functions
    library(reticulate)
    reticulate::source_python(system.file("python/all_functions.py",
                                          package = "HIP"))

    # Conforming values to work with Python script
    seed <- as.integer(seed)

    n <- as.integer(n)
    n <- as.list(n)
    n_py <- reticulate::r_to_py(n)

    m <- as.integer(m)

    # var_list <- lapply(p, function(h){1:h})

    # generate variable name vectors for each Di
    var_list <- lapply(1:D, function(i) {
      sapply(1:p[i], function(j) paste0("D", i, "_v", j))
    })

    p <- as.list(p)
    p <- as.integer(p)
    p_py <- reticulate::r_to_py(p)

    K <- as.integer(K)
    D <- as.integer(D)
    S <- as.integer(S)
    nonzero <- as.integer(nonzero)
    offset <- as.integer(offset)

    q <- as.integer(q)

    gendat_all <- generate_data(seed=0L, n=n, p=p, K=K, D=D, S=S,
                                   nonzero=nonzero, offset=offset,
                                   sigma_x=sigma_x, sigma_y=sigma_y, family=family,
                                   q = q, m = m, z_mean=25, z_sd=3, theta_init = theta_init, beta = rTorch::torch$tensor(beta))

    gendat_train <- generate_data(seed=seed, n=n, p=p, K=K, D=D, S=S,
                                     nonzero=nonzero, offset=offset,
                                     sigma_x=sigma_x, sigma_y=sigma_y, family=family,
                                     theta_init = gendat_all$theta, beta = gendat_all$beta, tau = gendat_all$tau,
                                     B = gendat_all$B, Z = gendat_all$Z, y_mean = gendat_all$y_mean, y_sd = gendat_all$y_sd,
                                     t_dist = t_dist, t_df = t_df)

    gendat_test <- generate_data(seed=seed+500L, n=n, p=p, K=K, D=D, S=S,
                                    nonzero=nonzero, offset=offset,
                                    sigma_x=sigma_x, sigma_y=sigma_y, family=family,
                                    theta_init = gendat_all$theta, beta = gendat_all$beta, tau = gendat_all$tau,
                                    B = gendat_all$B, Z = gendat_all$Z, y_mean = gendat_all$y_mean, y_sd = gendat_all$y_sd,
                                    t_dist = t_dist, t_df = t_df)
     return(list(X=gendat_train$X,
                 Y=gendat_train$Y,
                 X_test=gendat_test$X,
                 Y_test=gendat_test$Y,
                 D=D,
                 S=S,
                 sub_vec=1:S,
                 Z=gendat_all$Z,
                 B=gendat_all$B,
                 theta=gendat_all$theta,
                 beta=gendat_all$beta,
                 tau=gendat_all$tau,
                 y_mean=gendat_all$y_mean,
                 y_sd=gendat_all$y_sd,
                 z_mean=gendat_all$z_mean,
                 z_sd=gendat_all$z_sd,
                 seed=gendat_train$seed,
                 var_list=var_list))
  }
}
