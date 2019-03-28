rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(haven)
library(forcats)
library(caret)
library(recipes)
library(doMC)
library(foreach)

load("data/sampledata_completeness.RData")

#Keep only wanted variables
sampledata <- sampledata %>%
    select(-nmpid,-nfpid,-eduy_child,
           -nus_child,-gender_child,-matches("ability"),
           -matches("isic"),-matches("hhtype")) #Could also drop wact70 here

#Handle Factors
sampledata <- sampledata %>%
    mutate(occ70_father=factor(str_sub(occ70_father,1,3)),
           occ70_mother=factor(str_sub(occ70_mother,1,3)),
           nus_father=factor(str_sub(nus_father,1,3)),
           nus_mother=factor(str_sub(nus_mother,1,3)))

sampledata <- sampledata %>%
    mutate_at(vars(matches("occ70|emplstat70")),
              funs(fct_explicit_na))

# sampledata <- sampledata %>%
#     mutate_at(vars(matches(("nus|mstat|urban|studa|srcinc|
#                             wact|hours|occ|emplstat"))),
#               fct_lump,
#               prop=0.01)

sampledata <- sampledata %>%
    mutate_if(is.factor,
              fct_drop)

#Keep only complete observations
sampledata <- sampledata[complete.cases(sampledata),]

#Draw smaller random sample####
set.seed(10101)
holdout <- sample_frac(sampledata,size=0.5,replace=FALSE)

trainsample <- sampledata %>%
    anti_join(y=holdout,by="npid")

#Split into training and test sets####
set.seed(10101)
splitindex <- createDataPartition(trainsample$earncdf_child,
                                  p=0.8,
                                  list=FALSE)

train <- trainsample[splitindex,]
test <- trainsample[-splitindex,]

rm(splitindex)

#Save datasets
saveRDS(holdout,file = "data/holdout_caret.rds")
saveRDS(sampledata,file = "data/sampledata_caret.rds")
saveRDS(test,file = "data/test_caret.rds")
saveRDS(train,file = "data/train_caret.rds")
saveRDS(trainsample,file = "data/trainsample_caret.rds")

rm(holdout,sampledata,trainsample)

#Generate cross-validation folds
# set.seed(10101)
# trainfolds <- createFolds(train$earncdf_child,
#                                k=10)

#Set parameters
tc <- trainControl(method="cv",
                   number = 10,
                   #repeats=reps,
                   #index=trainfolds,
                   verboseIter=FALSE,
                   returnData=FALSE)

xgblength <- 4
rflength <- 1
glmnetlength <- 5

registerDoMC(cores=8) #Remember that cores*threads will be used
threads  <- 1

#Rank-Rank####

print(paste("Start","Rank-Rank",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_joint, data=train)

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m1_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

m1modellist <- list("OLS"=m1_lm)

resampledata <- data.frame("OLS"=(m1_lm$resample[,2]),
                           data="Training Resamples",
                           variables="Rank-Rank",
                           modelnumber=1)

testdata <- as.data.frame(lapply(m1modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Rank-Rank"
testdata$modelnumber <- 1

m1results <- bind_rows(resampledata,testdata)

rm(m1modellist)

#Income with multiple functional forms####
print(paste("Start","Income with multiple functional forms",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother,
                  data=train)

rec_obj <- rec_obj %>%
    step_interact(terms = ~ earncdf_father:earncdf_mother) %>%
    step_poly(all_predictors(),options=list(degree=3))

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m2_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m2_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

m2modellist <- list("OLS"=m2_lm,
                    "ElasticNet"=m2_glmnet)

resampledata <- as.data.frame(resamples(m2modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income with multiple functional forms"
resampledata$modelnumber <- 2
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m2modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income with multiple functional forms"
testdata$modelnumber <- 2

m2results <- bind_rows(resampledata,testdata)

rm(m2modellist)

#Income with polynomials####
print(paste("Start","Income with polynomials",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_joint, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_joint,options=list(degree=3))

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m3_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m3_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

m3modellist <- list("OLS"=m3_lm,
                    "ElasticNet"=m3_glmnet)

resampledata <- as.data.frame(resamples(m3modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income with polynomials"
resampledata$modelnumber <- 3
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m3modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income with polynomials"
testdata$modelnumber <- 3

m3results <- bind_rows(resampledata,testdata)

rm(m3modellist)

#Income (both parents)####

print(paste("Start","Income (both parents)",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother)

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m4_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

m4modellist <- list("OLS"=m4_lm)

resampledata <- data.frame("OLS"=(m4_lm$resample[,2]),
                           data="Training Resamples",
                           variables="Income (both parents)",
                           modelnumber=4)

testdata <- as.data.frame(lapply(m4modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income (both parents)"
testdata$modelnumber <- 4

m4results <- bind_rows(resampledata,testdata)

rm(m4modellist)

#Income & wealth####

print(paste("Start","Income & wealth",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother)

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m5_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m5_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m5_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = xgblength,
                nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m5_rf <- train(rec_obj,
               data=train,
               method="ranger",
               metric="Rsquared",
               trControl = tc,
               tuneLength = rflength,
               num.threads=threads,
               verbose=FALSE)

m5modellist <- list("OLS"=m5_lm,
                    "ElasticNet"=m5_glmnet,
                    "XGBoost"=m5_xgb,
                    "Ranger"=m5_rf)

resampledata <- as.data.frame(resamples(m5modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income & wealth"
resampledata$modelnumber <- 5
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m5modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income & wealth"
testdata$modelnumber <- 5

m5results <- bind_rows(resampledata,testdata)

rm(m5modellist)

#Income & education length####

print(paste("Start","Income & education length",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      eduy_father + eduy_father, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother)

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m6_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m6_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m6_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = xgblength,
                nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m6_rf <- train(rec_obj,
               data=train,
               method="ranger",
               metric="Rsquared",
               trControl = tc,
               tuneLength = rflength,
               num.threads=threads,
               verbose=FALSE)

m6modellist <- list("OLS"=m6_lm,
                    "ElasticNet"=m6_glmnet,
                    "XGBoost"=m6_xgb,
                    "Ranger"=m6_rf)

resampledata <- as.data.frame(resamples(m6modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income & education length"
resampledata$modelnumber <- 6
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m6modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income & education length"
testdata$modelnumber <- 6

m6results <- bind_rows(resampledata,testdata)

rm(m6modellist)

#Income, wealth & education length####
print(paste("Start","Income, wealth & education length",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother)

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m7_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m7_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m7_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = xgblength,
                nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m7_rf <- train(rec_obj,
               data=train,
               method="ranger",
               metric="Rsquared",
               trControl = tc,
               tuneLength = rflength,
               num.threads=threads,
               verbose=FALSE)

m7modellist <- list("OLS"=m7_lm,
                    "ElasticNet"=m7_glmnet,
                    "XGBoost"=m7_xgb,
                    "Ranger"=m7_rf)

resampledata <- as.data.frame(resamples(m7modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth & education length"
resampledata$modelnumber <- 7
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m7modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income, wealth & education length"
testdata$modelnumber <- 7


m7results <- bind_rows(resampledata,testdata)

rm(m7modellist)

#Income, wealth, education length & occupation####

print(paste("Start","Income, wealth, education length & occupation",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      occ70_father + occ70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_dummy(all_nominal())


print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m8_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m8_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m8_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = xgblength,
                nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m8_rf <- train(rec_obj,
               data=train,
               method="ranger",
               metric="Rsquared",
               trControl = tc,
               tuneLength = rflength,
               num.threads=threads,
               verbose=FALSE)

set.seed(10101)
m8modellist <- list("OLS"=m8_lm,
                    "ElasticNet"=m8_glmnet,
                    "XGBoost"=m8_xgb,
                    "Ranger"=m8_rf)

resampledata <- as.data.frame(resamples(m8modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth, education length & occupation"
resampledata$modelnumber <- 8
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m8modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income, wealth, education length & occupation"
testdata$modelnumber <- 8
m8results <- bind_rows(resampledata,testdata)

rm(m8modellist)

#Income, wealth, education length and type & occupation####

print(paste("Start","Income, wealth, education length and type & occupation",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      nus_father + nus_mother +
                      occ70_father + occ70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_dummy(all_nominal())

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m9_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m9_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m9_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = xgblength,
                nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m9_rf <- train(rec_obj,
               data=train,
               method="ranger",
               metric="Rsquared",
               trControl = tc,
               tuneLength = rflength,
               num.threads=threads,
               verbose=FALSE)

set.seed(10101)
m9modellist <- list("OLS"=m9_lm,
                    "ElasticNet"=m9_glmnet,
                    "XGBoost"=m9_xgb,
                    "Ranger"=m9_rf)

resampledata <- as.data.frame(resamples(m9modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth, education length and type & occupation"
resampledata$modelnumber <- 9
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m9modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income, wealth, education length and type & occupation"
testdata$modelnumber <- 9

m9results <- bind_rows(resampledata,testdata)

rm(m9modellist)

#Everything####

print(paste("Start","Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      nus_father + nus_mother +
                      occ70_father + occ70_mother +
                      mstat70_father + mstat70_mother +
                      urban70_father + urban70_mother +
                      studa70_father + studa70_mother +
                      srcinc70_father + srcinc70_mother +
                      nindh70_father + nindh70_mother + region_child,
                  data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_dummy(all_nominal())

print(paste("Start","OLS",Sys.time()))
set.seed(10101)
m10_lm <- train(rec_obj,
                data=train,
                method="lm",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 0)

print(paste("Start","Elastic Net",Sys.time()))
set.seed(10101)
m10_glmnet <- train(rec_obj,
                    data=train,
                    method="glmnet",
                    metric="Rsquared",
                    trControl = tc,
                    tuneLength = glmnetlength)

print(paste("Start","XGBoost",Sys.time()))
set.seed(10101)
m10_xgb <- train(rec_obj,
                 data=train,
                 method="xgbTree",
                 metric="Rsquared",
                 trControl = tc,
                 tuneLength = xgblength,
                 nthread=threads)

print(paste("Start","Ranger",Sys.time()))
set.seed(10101)
m10_rf <- train(rec_obj,
                data=train,
                method="ranger",
                metric="Rsquared",
                trControl = tc,
                tuneLength = rflength,
                num.threads=threads,
                verbose=FALSE)

m10modellist <- list("OLS"=m10_lm,
                     "ElasticNet"=m10_glmnet,
                     "XGBoost"=m10_xgb,
                     "Ranger"=m10_rf)

resampledata <- as.data.frame(resamples(m10modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region"
resampledata$modelnumber <- 10
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m10modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region"
testdata$modelnumber <- 10

m10results <- bind_rows(resampledata,testdata)

rm(m10modellist)

registerDoSEQ()

rm(holdout,test,train,trainsample,testdata,resampledata,sampledata,
   glmnetlength,rflength,tc,threads,trainfolds,xgblength,rec_obj)

rsquareds <- bind_rows(m1results,m2results,m3results,m4results,m5results,m6results,
                      m7results,m8results,m9results,m10results)

saveRDS(rsquareds,file = "models/caretrsquareds.rds")

rm(m1results,m2results,m3results,m4results,m5results,m6results,
          m7results,m8results,m9results,m10results,rsquareds)

save.image("models/caret.RData")
