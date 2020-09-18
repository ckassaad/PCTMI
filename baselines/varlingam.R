#setwd('/home/kassaad/Documents/Codes/R - codes/VARLiNGAM')
setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/baselines/VARLiNGAM')
source("VARLiNGAM.R")
source("VAR_estim.R")
source("ols_est.R")
source("Gauss_Stats.R")
source("Gauss_Tests.R")
source("tsdata2canonicalform.R")
setwd('./lingam/code')
source("lingam.R")
source("estimate.R")
source("nzdiagbruteforce.R")
source("all.perm.R")
source("nzdiagscore.R")
source("iperm.R")
source("sltbruteforce.R")
source("sltscore.R")
source("prune.R")
source("tridecomp.R")
source("sqrtm.R")



#getwd()
#source("sourcedir.R")
#sourceDir("./")
#source("main1.R")

#setwd('/home/kassaad/Documents/Codes/R - codes/simulated_ts_data/')
setwd('/home/kassaad/Documents/Codes/Causality-time-series-different-sampling-rate/data/simulated_ts_data/')
# "fork", "v_structure", "cycle", "diamond", "hidden", "complex"
struct_name <- "fork"
filenames <- list.files(struct_name, pattern="*.csv")
for (i in 1:length(filenames)){
  i=1
  data = read.csv(paste(struct_name,filenames[i],sep="/"))
  data$X <- NULL
  d = ncol(data)
  
  data_processed <- tsdata2canonicalform(data, nlags=5)
  res <- VARLiNGAM(data_processed, ntests=FALSE)

  unit_graph = matrix(0,d,d)
  for i in 1:d{
    unit_graph[i,:] = 1
  }
  path<-paste(struct_name,filenames[i],sep="/results/res_")
  write.csv(unit_graph,path)
}

res$Mhat
res$Bhat