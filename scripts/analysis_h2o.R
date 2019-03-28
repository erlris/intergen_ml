rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(haven)
library(forcats)
library(caret)
library(recipes)
library(doMC)
library(foreach)

#Load data
train <- readRDS("data/train_caret.rds")

#Estimate h2o####
library(h2o)
h2o.init(nthreads = 10)
h2o.removeAll()

train_h2o <- train %>%
    select(earncdf_child,earncdf_father,earncdf_mother,wealthcdf_father,
           wealthcdf_mother,eduy_father,eduy_mother,nus_father,nus_mother,
           occ70_father,occ70_mother,mstat70_father,mstat70_mother,urban70_father,
           urban70_mother,studa70_father,studa70_mother,srcinc70_father,srcinc70_mother,
           nindh70_father,nindh70_mother,region_child) %>%
    as.h2o()

# test_h2o <- test %>%
#     select(earncdf_child,earncdf_father,earncdf_mother,wealthcdf_father,
#            wealthcdf_mother,eduy_father,eduy_mother,nus_father,nus_mother,
#            occ70_father,occ70_mother,mstat70_father,mstat70_mother,urban70_father,
#            urban70_mother,studa70_father,studa70_mother,srcinc70_father,srcinc70_mother,
#            nindh70_father,nindh70_mother,region_child) %>%
#     as.h2o()

layerfun <- function(layers) {
    layerone <- sample(seq(5,200,1),size=1)
    layertwo <- sample(seq(1,layerone,1),size=1)
    layerthree <- sample(seq(1,layertwo,1),size=1)
    ifelse(layers==3,
                  return(c(layerone,layertwo,layerthree)),
                  ifelse(layers==2,
                         return(c(layerone,layertwo)),
                         return(layerone)))
}

params <- list(
    activation=c("Tanh","TanhWithDropout", "Rectifier",
                 "RectifierWithDropout","Maxout","MaxoutWithDropout"),
    hidden=lapply(sample.int(n=2,size=1000,replace = TRUE),
                  function(x) layerfun(x)),
    input_dropout_ratio=seq(0,0.5,0.10),#could do this for the layers as well
    epochs=seq(5,30,5),
    l1=c(0,0.001,0.01),
    l2=c(0,0.001,0.01))

search_criteria <- list(strategy="RandomDiscrete",
                        #max_models=10,
                        max_runtime_secs=60*60*72#,
                        #stopping_metric="RMSE",
                        #stopping_tolerance=0.0001,
                        #stopping_rounds=10
                        )

rm(h2o_grid)

h2o_grid <- h2o.grid(algorithm="deeplearning",
                     grid_id="dl_grid",
                     y="earncdf_child",
                     training_frame=train_h2o,
                     nfolds=10,
                     seed=10101,
                     #validation_frame=test_h2o,
                     #nfolds=10,
                     hyper_params=params,
                     search_criteria=search_criteria,
                     keep_cross_validation_predictions=TRUE,
                     do_hyper_params_check = FALSE)

grid <- h2o.getGrid("dl_grid",sort_by = "r2",decreasing=TRUE)
grid_metrics <- as.data.frame(grid@summary_table)
saveRDS(grid_metrics,file = "models/grid_metrics.rds")

best_model <- h2o.getModel(grid@model_ids[[1]])
h2o.saveModel(best_model,path = "models/",force=TRUE)

best_model_metrics <- as.data.frame(best_model@model$cross_validation_metrics_summary)
saveRDS(best_model_metrics,file = "models/best_model_metrics.rds")

# grid_metrics <- readRDS("models/grid_metrics.rds")
# best_model_metrics <- readRDS("models/best_model_metrics.rds")
# best_model <- h2o.loadModel(path = "models/dl_grid_model_14")

h2o.shutdown(prompt = FALSE)

