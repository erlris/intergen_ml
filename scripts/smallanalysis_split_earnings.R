rm(list=ls())

setwd("~/mlpaper/")

library(dplyr)
library(ggplot2)
library(caret)
library(doMC)
library(recipes)

load("data/traindta.RData")

#Put gender back in later
traindta <- select(traindta,-gender_child,-eduy_child)

set.seed(10101)

traindta <- sample_n(traindta,80000)

splitindex <- createDataPartition(traindta$earncdf_child,
                                  p=0.5,
                                  list=FALSE)

training <- traindta[splitindex,]
testing <- traindta[-splitindex,]

rm(splitindex,traindta)

#Recipes
#Without Interactions
rec_obj <- recipe(earncdf_child ~ ., data=training)

rec_obj <- rec_obj %>% 
    step_dummy(mob_child,starts_with("occ")) %>%
    step_nzv(all_predictors(),
             options=list(freq_cut=99/1,
                          unique_cut=0.01)) %>%
    step_corr(all_predictors(),
              threshold=0.9) %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors())

#With Interactions
rec_obj_interact <- recipe(earncdf_child ~ ., data=training)

rec_obj_interact <- rec_obj_interact %>% 
    step_interact(~ (all_predictors() -contains("yob") -all_nominal())^2) %>%
    step_poly(all_predictors(),-all_nominal()) %>%
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors(),
             options=list(freq_cut=99/1,
                          unique_cut=0.01)) %>%
    step_corr(all_predictors(),
              threshold=0.9) %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors())

#Parallel processing
registerDoMC(cores=5)

#Random Forest
source("scripts/randomforest.R",
       print.eval = TRUE)

#glmnet
source("scripts/glmnet.R",
       print.eval = TRUE)

#rpart
source("scripts/rpart.R",
       print.eval = TRUE)

#nnet
source("scripts/nnet.R",
       print.eval = TRUE)

#OLS

tc <- trainControl(method="none")

simplelm_fit <- train(earncdf_child ~ eduy_father,
                      data=training,
                      trControl=tc,
                      method="lm")

linearlm_fit <- train(rec_obj,
                      data=training,
                      trControl=tc,
                      method="lm")

fulllm_fit <- train(rec_obj_interact,
                    data=training,
                    trControl=tc,
                    method="lm")

registerDoSEQ()

rm(tc)

save.image(file="data/smallanalysis_earnings.RData")

# #Evaluating


#Loading models
rf_fit <- readRDS("models/rf_fit.rds")
glmnet_fit <- readRDS("models/glmnet_fit.rds")
rpart_fit <- readRDS("models/rpart_fit.rds")
nnet_fit <- readRDS("models/nnet_fit.rds")

#
models <- list(simplelm_fit,linearlm_fit,fulllm_fit,glmnet_fit,rpart_fit,
               rf_fit,nnet_fit)

simplelm_pred <- predict(simplelm_fit,newdata=testing)
linearlm_pred <- predict(linearlm_fit,newdata=testing)
fulllm_pred <- predict(fulllm_fit,newdata=testing)
glmnet_pred <- predict(glmnet_fit,newdata=testing)
rpart_pred <- predict(rpart_fit,newdata=testing)
rf_pred <- predict(rf_fit,newdata=testing)
nnet_pred <- predict(nnet_fit,newdata=testing)

predictions <- data.frame(simplelm_pred,linearlm_pred,fulllm_pred,glmnet_pred,
                          rpart_pred,rf_pred,nnet_pred)

evaluations <- data.frame(sapply(predictions,function(x)
    cor(x,testing$earncdf_child)^2))

evaluations$model  <- factor(c("Simple LM","LM with covariates",
                               "LM with interactions and polynomials","Glmnet","Rpart",
                               "Random Forest","Neural Net (nnet)"))

evaluations$model <- factor(evaluations$model,
                            levels=c("Simple LM","LM with covariates",
                                     "LM with interactions and polynomials","Glmnet","Rpart",
                                     "Random Forest","Neural Net (nnet)"))

colnames(evaluations) <- c("rsquared","Model")

ggplot(data=evaluations,aes(x=Model,y=rsquared)) +
    geom_col() +
    theme_bw(base_size = 16) +
    labs(x="Model",y="R-Squared") +
    coord_flip() +
    scale_y_continuous(breaks=seq(0,0.20,0.02))

ggsave("graphs/rsquareds_earnings.pdf",
       width=11.5,height=7)
