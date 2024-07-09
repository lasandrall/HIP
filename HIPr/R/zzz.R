.onAttach <- function(libname, pkgname) {
  # Load the virtual environment if it exists (created with createVirtualenv() function)
  if(reticulate::virtualenv_exists(envname="HIP_env")){
    reticulate::use_virtualenv(virtualenv = "HIP_env", required=FALSE)
  } else{
    warning("It's recommended you create the package virtual environment with the createVirtualenv() function.")
  }
}
