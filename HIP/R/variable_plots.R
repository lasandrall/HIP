#' Variable importance plots
#'
#' Outputs plots showing the highest ranking variables contributing to model performance.
#' Plots are shown for each data view and subgroup.
#'
#' @usage variable_plots(data, res, fix_or_search, top_plotted=15, compact=TRUE,
#' output_table=FALSE, file_path=NA)
#'
#' @param data data object from `generate_data` or `format_data`
#' @param res output from `select_lambda` or `fixed_lambda`
#' @param fix_or_search string - 'fixed' if results obtained from `fixed_lambda`, 'search' otherwise
#' @param top_plotted int - number of variables to include in each variable importance plot; 15 by default
#' @param compact boolean - if TRUE (default), only plots top_plotted variables; if set to FALSE, plots as many as possible
#' @param output_table boolean - set to TRUE to write variable importance results to an Excel file; FALSE by default
#' @param file_path string - if output_table is TRUE, you may also specify a file path and file name; make sure to add the .xlsx extension to your file name.
#' By default, it will be written to your current working directory with the name "variable_importance_table.xlsx"
#'
#' @returns Returns the variable importance plot as a ggplot object
#'
#' @examples
#'
#' # Generate data
#' dat_gen <- generate_data()
#'
#' # Get results from select_lambda
#' res <- select_lambda(dat_gen$X, dat_gen$Y, c(1,1), 'gaussian', 50)
#'
#' # Variable importance plots
#' variable_plots(dat_gen, res, 'search', output_table=TRUE)
#'
variable_plots <- function(data, res, fix_or_search, top_plotted=15, compact=TRUE, output_table=FALSE, file_path=NA){

  pd <- data$var_list

  # Import Python functions
  reticulate::source_python(system.file("python/all_functions.py",
                                        package = "HIP"))
  np <- reticulate::import("numpy")

  # Generating plots
  plot_list <- lapply(1:data$D, function(d){
    subs <- lapply(1:data$S, function(s){
      if(fix_or_search == "fixed"){
        plotdat <- data.frame(weight = sqrt(rowSums(np$array(res$out$subset$B[[d]][[s]])^2)),
                              varname = pd[[d]][np$array(res$out$include[[d]]) == 1]) %>%
          dplyr::arrange(dplyr::desc(weight)) %>%
          tibble::rowid_to_column(var = "rank") %>%
          dplyr::mutate(fill = as.numeric(rank <= res$topn[d]))
      } else{
        plotdat <- data.frame(weight = sqrt(rowSums(np$array(res$out$search_results[[res$best_index]]$subset$B[[d]][[s]])^2)),
                              varname = pd[[d]][np$array(res$out$search_results[[res$best_index]]$include[[d]]) == 1]) %>%
          dplyr::arrange(desc(weight)) %>%
          tibble::rowid_to_column(var = "rank") %>%
          dplyr::mutate(fill = as.numeric(rank <= res$topn)) # if needed: or topn[d] if different between views
      }

      if(compact){
        plotdat <- subset(plotdat, rank <= top_plotted)
      }

      ggplot(plotdat, aes(x = rank, y = weight, fill = fill)) +
        theme_classic() +
        geom_col(position = position_stack(reverse = TRUE)) +
        geom_text(aes(label=substr(varname,1,10),  hjust="left"), size=3) +
        coord_flip() +
        scale_x_reverse() +
        ylim(0, max(plotdat$weight)+0.5) +
        labs(x = "Variable", y = "Weight", title = paste0("View ", d, ", Subgroup ", s)) +
        theme(panel.border = element_rect(color = "black", fill = NA),
              axis.ticks.y = element_blank(),
              axis.text.y = element_blank(),
              legend.position = "none")
    })
    ggpubr::ggarrange(plotlist = subs, ncol = 1)
  })
  plot_total <- ggpubr::ggarrange(plotlist = plot_list, nrow = 1) %>%
    ggpubr::annotate_figure(bottom = ggpubr::text_grob(label = latex2exp::TeX(r"(Note: Light blue fill indicates the variable is in $N_{top}$ for that view and subgroup.)"),
                                       #"Note: Light blue fill indicates the variable is in $N_{top}$ for that view and subgroup.",
                                       hjust = 0, x = 0))
  print(plot_total)

  if(output_table){
    view_list <- lapply(1:data$D, function(d){
      subs <- lapply(1:data$S, function(s){
        if(fix_or_search == "fixed"){
          data.frame(Weight = round(sqrt(rowSums(np$array(res$out$subset$B[[d]][[s]])^2)), 4),
                     Variable = pd[[d]][np$array(res$out$include[[d]]) == 1],
                     View = d,
                     Subgroup = s) %>%
            arrange(desc(Weight)) %>%
            tibble::rowid_to_column(var = "Rank")
        } else{
          data.frame(Weight = round(sqrt(rowSums(np$array(res$out$search_results[[res$best_index]]$subset$B[[d]][[s]])^2)), 4),
                     Variable = pd[[d]][np$array(res$out$search_results[[res$best_index]]$include[[d]]) == 1],
                     View = d,
                     Subgroup = s) %>%
            dplyr::arrange(desc(Weight)) %>%
            tibble::rowid_to_column(var = "Rank")
        }
      })
      subs %>% dplyr::bind_rows()
    }) #end view_list
    view_list %>% dplyr::bind_rows()

    if(!is.na(file_path)){
      writexl::write_xlsx(view_list, file_path)
    } else{
      writexl::write_xlsx(view_list, "variable_importance_table.xlsx")
    }

    return(list(plot=plot_total, table=view_list))

  }
  else{
    return(plot_total)
  }
}
