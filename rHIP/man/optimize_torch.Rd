% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/optimize_torch.R
\name{optimize_torch}
\alias{optimize_torch}
\title{Main optimization function for continuous outcomes}
\usage{
optimze_torch(X, Y, lambda_xi, lambda_g, gamma, family, K=NULL, k_thresh=0.2,
update_thresh=10^-5, epsilon=10^-4, max_iter=50, standardize='subgroup',
std_type='scale_center', std_x=TRUE, std_y=TRUE, verbose=FALSE)
}
\arguments{
\item{Y}{matrix list - list of Y^s matrices containing outcomes}

\item{lambda_xi}{double - value of lambda_xi in penalty term}

\item{lambda_g}{double - value of lambda_g in penalty term}

\item{gamma}{list\if{html}{\out{<double>}} - indicators of whether to penalize each data view; should be length D}

\item{family}{string - family of outcome; options are 'gaussian', 'multiclass', 'poisson' or 'zip'}

\item{K}{int - number of latent components; will use 'select_K_simple' to choose if not provided}

\item{k_thresh}{double - threshold to use for 'select_K_simple' if K not provided; default = 0.2}

\item{update_thresh}{double - criteria for convergence in Z, G, Xi, and theta optimization functions; default = 10^-5}

\item{epsilon}{double - criteria for outer loop convergence; default = 10^-4}

\item{max_iter}{int - maximum number of outer loop iterations allowed; default = 50}

\item{standardize}{string - One of "all", "subgroup", or "none"; default = "subgroup"}

\item{std_type}{string - One of "scale_center", "center", or "norm"; default = "scale_center"}

\item{std_x}{boolean - indicates whether to standardize X; default = True}

\item{std_y}{boolean - indicates whether to standardize Y; default = True}

\item{verbose}{boolean - indicates whether to print additional info during run; default = False}

\item{matrix}{list - list of X^d,s matrices containing covariates}
}
\value{
Returns a list with the possible following elements based on status:
\item{\code{theta}}{tensor\if{html}{\out{<double>}} - estimate of theta}
\item{\code{beta}}{tensor\if{html}{\out{<double>}} - estimate of beta }
\item{\code{B}}{list<tensor\if{html}{\out{<double>}}> - estimates of each B^d,s}
\item{\code{G}}{list<tensor\if{html}{\out{<double>}}> - estimates of each G^d}
\item{\code{Xi}}{list<tensor\if{html}{\out{<double>}}> - estimates of each Xi^d,s}
\item{\code{Z}}{list<tensor\if{html}{\out{<double>}}> - estimates of each Z^s}
\item{\code{Lambda}}{tuple - values of lambda_xi and lambda_g used}
\item{\code{BIC}}{double - calculated BIC}
\item{\code{AIC}}{double - calculated AIC}
\item{\code{pred}}{double - prediction loss evaluated at final estimates}
\item{\code{train_err}}{nested list - list returned from function to calculate training error}
\item{\code{message}}{string - message with the status of the result; "Converged" if converged successfully, "MAX ITERS" if algorithm reached max_iter without converging.}
\item{\code{paths}}{list - history of losses until convergence}
\item{\code{iter}}{int - number of iterations to converge}
\item{\code{iter_time}}{double - time to find solution in seconds}
\item{\code{conv_criterion}}{double -  value of last convergence criterion}
\item{\code{std_x}}{boolean - whether X was standardized}
\item{\code{std_y}}{boolean - whether Y was standardized}
}
\description{
This function is used to fit a HIP model given X and Y training data, a set of lambda values,
and gamma values. This largely exists as a helper function, as the user will more often use
\code{select_lambda} or \code{fixed_lambda} to fit HIP models (which then access optimize_torch).
However, the function is available if the user needs to access the optimization function
directly.
}
\examples{
# Generating data
dat_gen <- generate_data()

# Run the optimization algorithm given the data, lambdas, and gamma
optimization_out <- optimize_torch(dat_gen$Y, dat_gen$X, 0.5, 0.5, c(1,1), 'gaussian', K=3)


}
