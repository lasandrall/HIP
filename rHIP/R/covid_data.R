#' Example COVID-19 data
#'
#' Data taken from a multi-omic study of COVID-19 severity.
#' This particular dataset has Sex (male/female) as the subgroup variable
#' and HFD45 (the number of hospital free days out of the 45 days enrolled in the study) as the outcome variable;
#' a zero indicates the patient was either still in the hospital or died.
#' There are two views in the data: genomic and proteomic data. The genomic data consists of RNAseq expression data for
#' 5800 genes in each patient, while the proteomic data consists of mass spectrometry data for 264 proteins in each patient.
#' Each also has an ID variable such as 'COVID_01' which provides a label for each patient across the data views.
#'
#' @docType data
#'
#' @usage data("covid_data")
#'
#' @format The data is divided up into 6 dataframes: 3 for training data and 3 for testing data.
#' The training datasets consist of 100 observations, while the testing datasets consist of
#' 20 observations.
#' `X_train_genomic` and `X_test_genomic` are dataframes of 5801 variables: the first column is the
#' ID variable, while the remaining columns are genes.
#' `X_train_proteomic` and `X_test_proteomic` are dataframes of 265 variables, the first column being an
#' ID variable and the remaining are proteins.
#' `Y_train` and `Y_test` consist of 3 columns: the first is the ID variable, the second column is the subgroup variable
#' variable Sex, and the third column is the outcome variable HFD45.
#'
#' @keywords datasets
#'
#' @source Source of data: Overmeyer et al. (2021), <https://doi.org/10.1016/j.cels.2020.10.003>
#'
#' Preprocessing of data: Lipman et al. (2022), <https://doi.org/10.1371/journal.pone.0267047>
#'
"covid_data"
