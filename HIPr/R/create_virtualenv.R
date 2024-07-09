#' Create virtual environment
#'
#' Creates a Python virtual environment and installs necessary packages
#'
#' This function creates a Python virtual environment called "HIP_env" and installs
#' necessary packages such as torch, numpy, and matplotlib into the environment. Once
#' created, the environment will be used when the package is loaded.
#'
#' @usage create_virtualenv()
#'
#' @examples
#' create_virtualenv()
#'
create_virtualenv=function(){

  env_name="HIP_env"
  new_env = identical(env_name, "HIP_env")

  if(new_env && reticulate::virtualenv_exists(envname=env_name) == TRUE){
    reticulate::virtualenv_remove(env_name)
  }

  package_req <- c("torch", "numpy", "joblib", "matplotlib", "scipy", "psutil", "torchvision")

  reticulate::virtualenv_create(env_name, packages = package_req)

  { cat(paste0("\033[0;", 32, "m","Virtual environment installed, please restart your R session and reload the package.","\033[0m","\n"))}
}
