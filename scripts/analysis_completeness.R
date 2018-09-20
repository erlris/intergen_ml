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
    select(-npid,-nmpid,-nfpid,-eduy_child,
           -nus_child,-gender_child,-matches("ability"),
           -matches("isic"),-matches("hhtype"))

#Handle Factors

sampledata <- sampledata %>%
    mutate(occ70_father=factor(str_sub(occ70_father,1,2)),
           occ70_mother=factor(str_sub(occ70_mother,1,2)),
           nus_father=factor(str_sub(nus_father,2,3)),
           nus_mother=factor(str_sub(nus_mother,2,3)))

sampledata <- sampledata %>%
    mutate_at(vars(matches("occ70|emplstat70")),
              funs(fct_explicit_na))

sampledata <- sampledata %>%
    mutate_at(vars(matches(("nus|mstat|urban|studa|srcinc|
                            wact|hours|occ|emplstat"))),
              fct_lump,
              prop=0.01)

sampledata <- sampledata %>%
    mutate_if(is.factor,
              fct_drop)

#Calculate log earnings
sampledata <- sampledata %>%
    mutate(logpearn_father = ifelse(meanpearn_father!=0,log(meanpearn_father),NA),
           logpearn_mother = ifelse(meanpearn_mother!=0,log(meanpearn_mother),NA),
           logpearn_joint = ifelse(sumpearn_joint!=0,log(sumpearn_joint),NA))

#Multiply earncdf by 100 for polynomials
sampledata <- sampledata %>%
    mutate(earncdf_father = earncdf_father*100,
           earncdf_mother = earncdf_mother*100,
           earncdf_joint = earncdf_joint*100,
           earncdf_child = earncdf_child*100)

#To make coefficients comparable
sampledata <- sampledata %>%
    mutate(wealthcdf_father=wealthcdf_father*100,
           wealthcdf_mother=wealthcdf_mother*100)

#Keep only complete observations
sampledata <- sampledata[complete.cases(sampledata),]

#Make summary statistics####

sampledata %>%
    select(meanpearn_child,
           meanpearn_father,
           meanpearn_mother,
           meanwealth_father,
           meanwealth_mother,
           eduy_father,
           eduy_mother) %>%
    data.frame() %>%
    stargazer::stargazer(summary=TRUE,
                         type="latex",
                         omit.summary.stat=(c("min","max","p25","p75")),
                         covariate.labels=c("Child's Average Earnings",
                                            "Father's Average Earnings",
                                            "Mother's Average Earnings",
                                            "Father's Average Wealth",
                                            "Mother's Average Wealth",
                                            "Father's Years of Education",
                                            "Mother's Years of Education"),
                         digit.separator="",
                         digits=2,
                         float=FALSE,
                         out="tables/summarystats.tex")

stargazer::stargazer(sampledata,
                     type="text",
                     summary=TRUE)

#Plots

sampledata %>%
    select("Child's Average Income"=meanpearn_child,
           "Father's Average Income"=meanpearn_father,
           "Mother's Average Income"=meanpearn_mother) %>%
    gather() %>%
    filter(value < 1000000) %>%
    ggplot(aes(x=value,fill=key,linetype=key)) +
    geom_density(alpha=0.5) +
    scale_linetype(name="") +
    scale_fill_brewer(name="",
                      palette="Set1") +
    theme_grey(base_size = 18) +
    theme(legend.position = "bottom",legend.direction = "vertical") +
    labs(x="Average Income",
         y="Density")

ggsave("~/git/intergen_ml/graphs/earndensities.pdf",
       height=7,width=7)

sampledata %>%
    select("Father's Average Wealth"=meanwealth_father,
           "Mother's Average Wealth"=meanwealth_mother) %>%
    gather() %>%
    filter(value < 1000000) %>%
    ggplot(aes(x=value,fill=key,linetype=key)) +
    geom_density(alpha=0.5) +
    scale_linetype(name="") +
    scale_fill_brewer(name="",
                      palette="Set1") +
    theme_grey(base_size = 18) +
    theme(legend.position = "bottom") +
    labs(x="Average Wealth",
         y="Density") +
    coord_cartesian(xlim=c(0,150000))

ggsave("~/git/intergen_ml/graphs/wealthdensities.pdf",
       height=7,width=7)

sampledata %>%
    select("Father's Years of Education"=eduy_father,
           "Mother's Years of Education"=eduy_mother) %>%
    gather() %>%
    filter(value >= 8) %>%
    ggplot(aes(x=value,fill=key,linetype=key)) +
    geom_histogram(binwidth=1,
                   position = "dodge") +
    scale_linetype(name="") +
    scale_fill_brewer(name="",
                      palette="Set1") +
    theme_grey(base_size = 18) +
    theme(legend.position = "bottom") +
    labs(x="Years of Education",
         y="Observations") +
    scale_x_continuous(breaks=seq(8,22,2))

ggsave("~/git/intergen_ml/graphs/eduhistograms.pdf",
       height=7,width=7)

#Draw smaller random sample####
set.seed(10101)
smalldata <- sample_n(sampledata,125000)

#Split into training and test sets####
set.seed(10101)
splitindex <- createDataPartition(smalldata$earncdf_child,
                                  p=0.8,
                                  list=FALSE)

train <- smalldata[splitindex,]
test <- smalldata[-splitindex,]

rm(splitindex,smalldata)

#Generate cross-validation folds

reps <- 5

set.seed(10101)
trainfolds <- createMultiFolds(train$earncdf_child,
                               k=10,
                               times=reps)

#Set parameters
tc <- trainControl(method="repeatedcv",
                   number = 10,
                   repeats=reps,
                   index = trainfolds,
                   verboseIter = FALSE,
                   returnData=FALSE)

xgblength <- 4
rflength <- 3
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

#Income & wealth####

print(paste("Start","Income & wealth",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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

#Income & education length####

print(paste("Start","Income & education length",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      eduy_father + eduy_father, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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

#Income, wealth & education length####
print(paste("Start","Income, wealth & education length",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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

#Income, wealth, education length & occupation####

print(paste("Start","Income, wealth, education length & occupation",Sys.time()))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      occ70_father + occ70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())


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
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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

#Everything####

print(paste("Start","Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household",Sys.time()))

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
                      nindh70_father + nindh70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_poly(earncdf_father,earncdf_mother) %>%
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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
resampledata$variables <- "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household"
resampledata$modelnumber <- 10
resampledata$Resample <- NULL

testdata <- as.data.frame(lapply(m10modellist,function(x) cor(predict(x,newdata=test),test$earncdf_child)^2))
testdata$data <- "Testing"
testdata$variables <- "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household"
testdata$modelnumber <- 10

m10results <- bind_rows(resampledata,testdata)

registerDoSEQ()

save.image("models/completeness.RData")

#Plotting####

#load("models/completeness.RData")

plotdata <- bind_rows(m1results,m2results,m3results,
                      m4results,m5results,m6results,
                      m7results,m8results,m9results,
                      m10results)

plotdata %>%
    gather(key="Model",value="rsquared",-data,
           -variables,-modelnumber) %>%
    ggplot(aes(x=reorder(str_wrap(variables,15),modelnumber),
               y=rsquared,
               fill=factor(Model,levels=c("OLS","ElasticNet","Ranger","XGBoost"),
                           labels=c("OLS","Elastic Net","Ranger","XGBoost")))) +
    stat_summary(fun.y = "mean",geom="col",position=position_dodge(),
                 color="black") +
    stat_summary(fun.data="mean_se",
                 fun.args = list(mult=2),
                 position=position_dodge(width=0.9),
                 geom="errorbar",
                 aes(width=0.3)) +
    facet_wrap("data",nrow=2) +
    scale_fill_brewer(palette = "Blues",name="Estimator") +
    labs(x="Included Variables",y="R-Squared") +
    theme_grey(base_size=15) +
    theme(legend.position = "bottom") +
    guides(fill = guide_legend(title.position = "top",title.hjust = 0.5))

ggsave(file="~/git/intergen_ml/graphs/completeness_modelcomp.pdf",
       height=9,width=16)

#Regional comparisons####

#Calculate log child earnings
sampledata <- sampledata %>%
    mutate(logpearn_child = ifelse(meanpearn_child!=0,log(meanpearn_child),NA))

estfun <- function(regionnumber) {
    
    #Rank-rank
    tempfit <- lm(earncdf_child ~ earncdf_father,
                  data=filter(sampledata,region_child==regionnumber))
    
    rankcoef <- coef(tempfit)["earncdf_father"]
    
    #Rank-education
    tempfit <- lm(earncdf_child ~ eduy_father,
                  data=filter(sampledata,region_child==regionnumber))
    
    educoef <- coef(tempfit)["eduy_father"]
    
    #Rank-wealth
    tempfit <- lm(earncdf_child ~ wealthcdf_father,
                  data=filter(sampledata,region_child==regionnumber))
    
    wealthcoef <- coef(tempfit)["wealthcdf_father"]
    
    #Observations
    observations <- nrow(filter(sampledata,region_child==regionnumber))
    
    results <- data.frame(region=regionnumber,
                          rank_coef=rankcoef,
                          edu_coef=educoef,
                          wealth_coef=wealthcoef,
                          observations=observations)
    return(results)
}


regions <- unique(sampledata$region_child)

results <- foreach (region = regions, .combine = rbind) %do% {
    estfun(region)
}

#Calculate variances and correlations 
stats <- sampledata %>%
    group_by(region_child) %>%
    summarize(earn_sd = sd(earncdf_father),
              edu_sd = sd(eduy_father),
              wealth_sd = sd(wealthcdf_father),
              earn_edu_cor = cor(earncdf_father,eduy_father),
              earn_wealth_cor = cor(earncdf_father,meanwealth_father))

results <- left_join(x=results,y=stats,by=c("region"="region_child"))


#Calculate r-squareds by region (kunne gjort dette i funksjonen over)

sampledata$xgbpred <- predict(m7_xgb,newdata=sampledata)
sampledata$rankpred <- predict(m1_lm,newdata=sampledata)

rsquareds <- foreach (region=regions, .combine = rbind) %do% {
    tempdta <- sampledata[sampledata$region_child==region,]
    rsquared_rank = cor(tempdta$earncdf_child,tempdta$rankpred)^2
    rsquared_full = cor(tempdta$earncdf_child,tempdta$xgbpred)^2
    returndf = data.frame(region=region,
                          rsquared_rank=rsquared_rank,
                          rsquared_full=rsquared_full)
    return(returndf)
}

results <- left_join(x=results,y=rsquareds,by=c("region"))

results <- results %>%
    mutate(rsquared_ratio=rsquared_full/rsquared_rank)

results %>%
    filter(observations > 200) %>%
    select(rsquared_ratio,earn_sd,edu_sd,wealth_sd,
           earn_edu_cor,earn_wealth_cor) %>%
    rename("Stanard Deviation of Earnings Percentile"=earn_sd,
           "Standard Deviation of Education"=edu_sd,
           "Standard Deviation of Wealth Percentile"=wealth_sd,
           "Correlation between Earnings Percentile and Education"=earn_edu_cor,
           "Correlation between Earnings Percentile and Wealth Percentile"=earn_wealth_cor) %>%
    gather(key=variable,value=value,-rsquared_ratio) %>%
    ggplot(aes(x=value,y=rsquared_ratio)) +
    geom_point() + 
    geom_smooth(method="lm") +
    labs(x="Value",y="Full R-Squared / Rank-Rank R-Squared")+
    theme_grey(base_size=18) +
    facet_wrap("variable",scales = "free_x")

ggsave("~/git/intergen_ml/graphs/rsquaredratio_stats.pdf",
       width=16,height=9)

results %>%
    filter(observations > 200) %>%
    ggplot(aes(x=rank_coef,y=edu_coef)) +
    geom_point() +
    geom_smooth(method="lm") +
    labs(x="Rank-Rank Coefficient",
         y="Rank-Education Coefficient") +
    theme_grey(base_size=18)

ggsave("~/git/intergen_ml/graphs/rankcoef_educoef.pdf",
       height=7,width=7)

results %>%
    filter(observations > 200) %>%
    ggplot(aes(x=rank_coef,y=wealth_coef)) +
    geom_point() +
    geom_smooth(method="lm") +
    labs(x="Rank-Rank Coefficient",
         y="Rank-Wealth Coefficient") +
    theme_grey(base_size=18)

ggsave("~/git/intergen_ml/graphs/rankcoef_wealthcoef.pdf",
       height=7,width=7)

sampledata %>%
    ggplot(aes(x=earncdf_joint,y=earncdf_child)) +
    stat_summary_bin(fun.data="mean_cl_boot",
                     geom="pointrange",
                     breaks=seq(0,99,2)) +
    labs(x="Parents' Joint Income Percentile",
         y="Child's Income Percentile") +
    scale_x_continuous(breaks=seq(0,100,20)) +
    theme_grey(base_size=18)
    
ggsave("~/git/intergen_ml/graphs/conditionalmeans_childearn_jointearn.pdf",
       height=7,width=7)

sampledata %>%
    filter(eduy_father >= 8) %>%
    ggplot(aes(x=eduy_father,y=earncdf_child)) +
    stat_summary(fun.data="mean_cl_boot",
                     geom="pointrange") +
    labs(x="Father's Years of Education",
         y="Child's Income Percentile") +
    scale_x_continuous(breaks=seq(8,22,2)) +
    theme_grey(base_size=18) 

ggsave("~/git/intergen_ml/graphs/conditionalmeans_childearn_fatheredu.pdf",
       height=7,width=7)
    
sampledata %>%
    filter(eduy_mother >= 8) %>%
    ggplot(aes(x=eduy_mother,y=earncdf_child)) +
    stat_summary(fun.data="mean_cl_boot",
                 geom="pointrange") +
    labs(x="Mother's Years of Education",
         y="Child's Income Percentile") +
    scale_x_continuous(breaks=seq(8,22,2)) +
    theme_grey(base_size=18) 

ggsave("~/git/intergen_ml/graphs/conditionalmeans_childearn_motheredu.pdf",
       height=7,width=7)

sampledata %>%
    ggplot(aes(x=wealthcdf_father,y=earncdf_child)) +
    stat_summary_bin(fun.data="mean_cl_boot",
                 geom="pointrange",
                 binwidth=2) +
    labs(x="Father's Wealth Percentile",
         y="Child's Earnings Percentile") +
    theme_grey(base_size=18)

ggsave("~/git/intergen_ml/graphs/conditionalmeans_childearn_fatherwealth.pdf",
       height=7,width=7)

#Generate joint wealth rank

sampledata <- sampledata %>%
    mutate(meanwealth_joint = meanwealth_father + meanwealth_mother) %>%
    group_by(yob_child) %>%
    mutate(wealthcdf_joint = cume_dist(meanwealth_joint)*100) %>%
    ungroup()
    
sampledata %>%
    ggplot(aes(x=wealthcdf_joint,y=earncdf_child)) +
    stat_summary_bin(fun.data="mean_cl_boot",
                     geom="pointrange",
                     breaks=(seq(25,100,2))) +
    labs(x="Joint Wealth Percentile",
         y="Child's Income Percentile") +
    theme_grey(base_size=18)

ggsave("~/git/intergen_ml/graphs/conditionalmeans_childearn_jointwealth.pdf",
       height=7,width=7)

sampledata <- sampledata %>%
    select(-meanwealth_joint,-wealthcdf_joint)

#Making maps

coords <- read_dta("/data/gis/norge/arbeidsmarked/koordinat_arbeidsmarked.dta")

colnames(coords) <- c("region","id","x","y")

mapdata <- results %>%
    mutate(region=as.numeric(as.character(region))) %>%
    select(region,rank_coef,edu_coef,wealth_coef)

mapdata <- inner_join(x=mapdata,y=coords,by=c("region"))

mapdata %>%
    ggplot(aes(x=x,y=y,group=region,fill=rank_coef)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank\nCoefficient",
                         palette = "Blues",direction = 1) +
    theme_void(base_size=18) +
    theme(legend.position = c(0.7,0.5))

ggsave("~/git/intergen_ml/graphs/map_rankrank.pdf",
       height=7,width=7)

mapdata %>%
    ggplot(aes(x=x,y=y,group=region,fill=edu_coef)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Education\nCoefficient",
                         palette = "Blues",direction = 1) +
    theme_void(base_size=18) +
    theme(legend.position = c(0.7,0.5))

ggsave("~/git/intergen_ml/graphs/map_rankedu.pdf",
       height=7,width=7)

mapdata %>%
    ggplot(aes(x=x,y=y,group=region,fill=wealth_coef)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Wealth\nCoefficient",
                         palette = "Blues",direction = 1) +
    theme_void(base_size=18) +
    theme(legend.position = c(0.7,0.5))

ggsave("~/git/intergen_ml/graphs/map_rankwealth.pdf",
       height=7,width=7)

#Merge names for region results####

regiondata <- readxl::read_xlsx("rawdata//labormarketregions_names.xlsx")

regiondata$code <- as.factor(regiondata$code)

results <- results %>%
    inner_join(y=regiondata,by=c("region"="code"))
    
#Save datasets####

saveRDS(object=plotdata,
        file="~/git/intergen_ml/data/modelresults.rds")

saveRDS(object = results,
        file = "~/git/intergen_ml/data/regionresults.rds")

#Comparing predictions from ML and OLS####

df1 <- data.frame(lapply(m1modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=1)


df2 <- data.frame(lapply(m2modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=2)


df3 <- data.frame(lapply(m3modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=3)


df4 <- data.frame(lapply(m4modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=4)


df5 <- data.frame(lapply(m5modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=5)


df6 <- data.frame(lapply(m6modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=6)


df7 <- data.frame(lapply(m7modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=7)

df8 <- data.frame(lapply(m8modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=8)


df9 <- data.frame(lapply(m9modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=9)

df10 <- data.frame(lapply(m10modellist,function(x) predict(x,newdata=test)),
                  observed=test$earncdf_child,
                  modelnumber=10)

predictionresults <- bind_rows(df1,df2,df3,df4,df5,df6,df7,df8,df9,df10)

variablelist <- plotdata %>%
    select(variables,modelnumber) %>%
    group_by(modelnumber) %>%
    filter(row_number(modelnumber)==1)

variablelist <- variablelist %>%
    mutate(variables=str_replace_all(variables,"Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household","Extended"))

predictionresults <- inner_join(x=predictionresults,y=variablelist,by=c("modelnumber"))

predictionresults %>%
    gather(key="key",value="value",-observed,-variables,-modelnumber) %>%
    filter(key=="OLS" | key=="XGBoost") %>%
    ggplot(aes(x=observed,y=value,color=key,shape=key)) +
    stat_summary_bin(binwidth=2,
                     fun.y="mean",
                     geom="point") +
    stat_summary_bin(binwidth=2,
                     fun.y="mean",
                     geom="line") +
    stat_summary_bin(binwidth=2,
                     fun.data="mean_cl_boot",
                     geom="ribbon",
                     alpha=0.2,
                     aes(color=NULL,
                         fill=key)) +
    scale_fill_brewer(name="",palette = "Set1") +
    scale_linetype(name="") + 
    scale_color_brewer(name="",palette = "Set1") +
    scale_shape(name="") +
    facet_wrap(~ reorder(str_wrap(variables,40),modelnumber),
               nrow=4) +
    theme_grey(base_size=16) +
    theme(legend.position = c(0.85,0.1)) +
    labs(x="Observed Income Percentile",
         y="Predicted Income Percentile")
    
ggsave("~/git/intergen_ml/graphs/predictioncomparisons.pdf",
       height=13,width=11)

predictionresults %>%
    gather(key="key",value="value",-observed,-variables,-modelnumber) %>%
    filter(key=="OLS" | key=="XGBoost") %>%
    ggplot(aes(x=observed,y=value,color=key,shape=key)) +
    stat_summary_bin(binwidth=2,
                     fun.y="mean",
                     geom="point") +
    stat_summary_bin(binwidth=2,
                     fun.y="mean",
                     geom="line") +
    stat_summary_bin(binwidth=2,
                     fun.data="mean_cl_boot",
                     geom="ribbon",
                     alpha=0.2,
                     aes(color=NULL,
                         fill=key)) +
    geom_abline(aes(intercept=0,slope=1)) +
    scale_fill_brewer(name="",palette = "Set1") +
    scale_linetype(name="") + 
    scale_color_brewer(name="",palette = "Set1") +
    scale_shape(name="") +
    facet_wrap(~ reorder(str_wrap(variables,40),modelnumber),
               nrow=4) +
    theme_grey(base_size=16) +
    theme(legend.position = c(0.85,0.1)) +
    labs(x="Observed Income Percentile",
         y="Predicted Income Percentile")

ggsave("~/git/intergen_ml/graphs/predictioncomparisons_45degreelines.pdf",
       height=13,width=11)
