setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/baselines/ts-FCI/RCode_TETRADjar_tsFCI/RCode_TETRADjar')
start_up <- function() {
  
  source('dconnected.R')
  source('genData.R')
  source('main_tetrad_fci.R')
  source('plot_timeseries.R')
  source('Plotting_Commands_Barplots.R')
  source('plot_ts_pag.R')
  source('realData_tsfci.R')
  source('scores.R')
  source('Simulation_Commands.R')
  source('Simulations_data_cont.R')
  source('Simulations_data_disc.R')
  source('Simulations_graph.R')
  source('Tetrad_R_interact.R')
  source('ts_functions.R')
  
}
start_up()
temporal_to_summary <- function(temporal_graph, nrep=5){
  dlag <- ncol(temporal_graph)
  d <- dlag/nrep
  idx_count = rep(1:d, nrep)
  summary_graph = matrix(data=0, nrow=d, ncol = d)
  for (i in 1:dlag){
    temp = temporal_graph[i,]
    for (j in 1:dlag){
      if (temp[j]==2){
        i_summary <- idx_count[i]
        j_summary <- idx_count[j]
        summary_graph[i_summary,j_summary] = 2
      }
      else if (temp[j]==3){
        i_summary <- idx_count[i]
        j_summary <- idx_count[j]
        if (summary_graph[i_summary,j_summary]!=2){
          summary_graph[i_summary,j_summary] = 3
        }
      }
      else if (temp[j]==1){
        i_summary <- idx_count[i]
        j_summary <- idx_count[j]
        if (summary_graph[i_summary,j_summary]==0){
          summary_graph[i_summary,j_summary] = 1
        }
      }
    }
  }
  return(summary_graph)
}

scale_name <- "unscaled"
struct_name <- "7ts_hidden"
path_data <- '/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/data/simulated_ts_data'
path_algo <- '/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/baselines/ts-FCI/RCode_TETRADjar_tsFCI/RCode_TETRADjar'
path_save_res <- '/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/experiments/causal_discovery/R_results/tsfci'
setwd(paste(path_data,scale_name, sep="/"))
filenames <- list.files(struct_name, pattern="*.csv")
sig=0.01 
lag=5
for (i in 1:length(filenames)){
  setwd(paste(path_data,scale_name, sep="/"))
  data = read.csv(paste(struct_name,filenames[i],sep="/"))
  data$X <- NULL
  setwd(path_algo)
  tsfci_graph <- realData_tsfci(data=data, sig=sig, nrep=lag, inclIE=FALSE, alg="tscfci", datatype="continuous", makeplot=FALSE)
  
  unit_graph <- temporal_to_summary(tsfci_graph, nrep=lag)
  print(unit_graph)

  setwd(path_save_res)
  path<-paste(struct_name,filenames[i],sep="/res_")
  write.csv(unit_graph,path)
}




