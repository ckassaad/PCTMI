library(gam)
library(kernlab)
library(gptk)
setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/baselines/onlineCodeTimino/codeTimino')
source("granger_causality.R")
source("timino_causality.R")
source("./util/hammingDistance.R")
source("./util/indtestAll.R")
source("./util/indtestHsic.R")
source("./util/indtestPcor.R")
source("./util/TSindtest.R")
source("./util/fitting_ts.R")

#setwd('/home/kassaad/Documents/Codes/R - codes/simulated_ts_data/')
scale_name <- "unscaled"
struct_name <- "7ts"
setwd(paste('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/data/simulated_ts_data',scale_name, sep="/"))
# "fork", "v_structure", "cycle", "diamond", "hidden", "complex"
filenames <- list.files(struct_name, pattern="*.csv")
for (i in 1:length(filenames)){
  data = read.csv(paste(struct_name,filenames[i],sep="/"))
  data$X <- NULL
  # traints_linear, traints_gam or traints_gp
  # indtestts_hsic or indtestts_crosscov
  timino_graph <- timino_dag(data, alpha = 0.05, max_lag = 5, model = traints_linear, indtest = indtestts_crosscov, output = TRUE)

  unit_graph <- diag(nrow(timino_graph))

  unit_graph[is.na(timino_graph)] <- 3
  timino_graph[is.na(timino_graph)] <- 3
  
  for (j1 in 1:nrow(timino_graph)){
    for (j2 in 1:nrow(timino_graph)){
      if (timino_graph[j1,j2] == 1){
        unit_graph[j1,j2] <- 2  
        unit_graph[j2,j1] <- 1  
      }
    }
  }
  for (j1 in 1:nrow(timino_graph)){
    for (j2 in 1:nrow(timino_graph)){
      if (timino_graph[j1,j2] == 3){
        unit_graph[j2,j1] = 1
      }
    }
  }
  print(unit_graph)
  path<-paste(struct_name,filenames[i],sep="/results/res_")
  write.csv(unit_graph,path)
}













setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/data/fMRI_processed_by_Nauta/returns/small_datasets')
filenames <- list.files("./", pattern="*.csv")
for (i in 1:length(filenames)){
  data = read.csv(filenames[i], header = FALSE)
  # traints_linear, traints_gam or traints_gp
  # indtestts_hsic or indtestts_crosscov
  timino_graph <- timino_dag(data, alpha = 0.05, max_lag = 5, model = traints_linear, indtest = indtestts_crosscov, output = TRUE)
  unit_graph <- diag(nrow(timino_graph))
  
  unit_graph[is.na(timino_graph)] <- 3
  timino_graph[is.na(timino_graph)] <- 3
  
  for (j1 in 1:nrow(timino_graph)){
    for (j2 in 1:nrow(timino_graph)){
      if (timino_graph[j1,j2] == 1){
        unit_graph[j1,j2] <- 2  
        unit_graph[j2,j1] <- 1  
      }
    }
  }
  for (j1 in 1:nrow(timino_graph)){
    for (j2 in 1:nrow(timino_graph)){
      if (timino_graph[j1,j2] == 3){
        unit_graph[j2,j1] = 1
      }
    }
  }
  print(unit_graph)
  path<-paste("./results_timino/res_",filenames[i],sep="")
  write.csv(unit_graph,path)
}





# setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/data/FinanceCPT/returns/')
# filenames <- list.files("./", pattern="*.csv")
# for (i in 1:length(filenames)){
#   data = read.csv(filenames[i], header = FALSE)
#   # traints_linear, traints_gam or traints_gp
#   # indtestts_hsic or indtestts_crosscov
#   timino_graph <- timino_dag(data, alpha = 0.05, max_lag = 5, model = traints_linear, indtest = indtestts_crosscov, output = TRUE)
#   unit_graph <- diag(nrow(timino_graph))
#
#   unit_graph[is.na(timino_graph)] <- 3
#   timino_graph[is.na(timino_graph)] <- 3
#
#   for (j1 in 1:nrow(timino_graph)){
#     for (j2 in 1:nrow(timino_graph)){
#       if (timino_graph[j1,j2] == 1){
#         unit_graph[j1,j2] <- 2
#         unit_graph[j2,j1] <- 1
#       }
#     }
#   }
#   for (j1 in 1:nrow(timino_graph)){
#     for (j2 in 1:nrow(timino_graph)){
#       if (timino_graph[j1,j2] == 3){
#         unit_graph[j2,j1] = 1
#       }
#     }
#   }
#   print(unit_graph)
#   path<-paste("../results/res_",filenames[i],sep="")
#   write.csv(unit_graph,path)
# }