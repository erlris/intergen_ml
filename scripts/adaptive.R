rm(list=ls())

setwd("~/mlpaper/")

library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(doMC)
library(recipes)
library(stringr)

load("data/traindta.RData")

#Generate County Factor (could do this in data wrangling script)
counties <- as.character(traindta$mob_child)

counties[str_length(counties) < 4]  <- paste0(0,counties[str_length(counties) < 4])

counties <- substring(counties,1,2)

traindta$county_child <- counties

traindta <- traindta %>%
    filter(county_child!=99) %>%
    mutate(county_child=factor(county_child))

rm(counties)

#Redefine year of birth to factor
traindta <- traindta %>%
    mutate(yob_child=factor(yob_child))

#Draw smaller random sample and split into training and test set
traindta <- select(traindta,-gender_child,-eduy_child)

set.seed(10101)
traindta <- sample_n(traindta,200000)

splitindex <- createDataPartition(traindta$earncdf_child,
                                  p=0.8,
                                  list=FALSE)

training <- traindta[splitindex,]
testing <- traindta[-splitindex,]

rm(splitindex,traindta)

#Recipes####
#Without Interactions

rec_obj <- recipe(earncdf_child ~ ., data=training)

rec_obj <- rec_obj %>% 
    step_dummy(yob_child,contains("occ"),mob_child,county_child) %>%
    step_nzv(all_predictors(),
             options=list(freq_cut=200/1,
                          unique_cut=0.0025)) %>%
    step_corr(all_predictors(),use="everything") %>%
    step_center(all_predictors(),-contains("yob_child|occ|mob_child|county_child")) %>%
    step_scale(all_predictors(),-contains("yob_child|occ|mob_child|county_child"))

#Carry out preprocessing on training set####
baked <- bake(prep(rec_obj),
              newdata=training)

outcome <- baked$earncdf_child
preds <- data.frame(select(baked,-earncdf_child))

#Summary Function####
summaryfun <- function(data, lev=NULL, model = NULL) {
    var <- var(data$obs,na.rm=TRUE)
    mse <- mean((data$obs-data$pred)^2,na.rm=TRUE)
    fve <- 1 - mse/var
    
    summary <- c(mse,var,fve)
    names(summary) <- c("MSE","Var","FVE")
    return(summary)
}

#Estimating models####

# seeds <- vector(mode = "list", length = 5*reps + 1)
# 
# set.seed(10101)
# for(j in seq(1,5*reps,1)) seeds[[j]] <- sample.int(1000, 3) #3 is tuneLength here
# rm(j)
# seeds[[5*reps+1]] <- sample.int(1000, 1)

#Parameters
reps <- 10

tc <- trainControl(method="adaptive_cv",
                   number = 5,
                   repeats=reps,
                   adaptive=list(min=5,
                                 alpha=0.05,
                                 method="gls",
                                 complete=TRUE),
                   verboseIter = TRUE,
                   #seeds=seeds,
                   summaryFunction=summaryfun,
                   returnData=FALSE)

registerDoMC(cores=3)

grid <- expand.grid(nrounds=seq(50,400,50),
                    max_depth=seq(1,5,1),
                    eta=c(0.05,0.1,0.2,0.3),
                    gamma=0,
                    colsample_bytree=seq(0.5,0.9,0.2),
                    min_child_weight=1,
                    subsample=seq(0.5,1,0.25))

print(paste("Xgboost begins",Sys.time()))
set.seed(10101)
xgb_fit <- train(y=outcome,x=preds,
                 #data=training,
                 trControl=tc,
                 tuneGrid=grid,
                 method="xgbTree",
                 metric="FVE",
                 maximize=TRUE,
                 nthread=3,
                 verbose=0) 

save.image("models/modelcomp_adaptive.RData")
print(paste("Xgboost ends",Sys.time()))
print(xgb_fit)

registerDoMC(cores=8)

print(paste("Glmnet begins",Sys.time()))
set.seed(10101)
glmnet_fit <- train(y=outcome,x=preds, 
                    #data=training,
                    trControl=tc,
                    tuneLength=10,
                    method="glmnet",
                    metric="FVE",
                    maximize=TRUE) 

save.image("models/modelcomp_adaptive.RData")
print(paste("Glmnet ends",Sys.time()))
print(glmnet_fit)

print(paste("Boosting begins",Sys.time()))
set.seed(10101)
gbm_fit <- train(y=outcome,x=preds, 
                 #data=training,
                 trControl=tc,
                 tuneLength=5,
                 method="gbm",
                 metric="FVE",
                 maximize=TRUE,
                 verbose=FALSE)

save.image("models/modelcomp_adaptive.RData")
print(paste("Boosting ends",Sys.time()))
print(gbm_fit)

tc <- trainControl(method="repeatedcv",
                   number = 5,
                   repeats=reps,
                   verboseIter = TRUE,
                   #seeds=seeds,
                   summaryFunction=summaryfun,
                   returnData=FALSE)

grid <- expand.grid(mtry=floor(sqrt(ncol(preds))),
                    splitrule="variance",
                    min.node.size=10)

print(paste("Ranger begins",Sys.time()))
set.seed(10101)
ranger_fit <- train(y=outcome,x=preds, 
                    #data=training,
                    trControl=tc,
                    tuneGrid=grid,
                    method="ranger",
                    metric="FVE",
                    maximize=TRUE,
                    num.threads=1,
                    verbose=FALSE) 

save.image("models/modelcomp_adaptive.RData")
print(paste("Ranger ends",Sys.time()))
print(ranger_fit)

# print(paste("Random forest begins",Sys.time()))
# set.seed(10101)
# rf_fit <- train(y=outcome,x=preds, 
#                 #data=training,
#                 trControl=tc,
#                 tuneLength=1,
#                 method="rf",
#                 metric="FVE",
#                 maximize=TRUE)
# 
# save.image("models/modelcomp_adaptive.RData")
# print(paste("Random forest ends",Sys.time()))
# print(rf_fit)

set.seed(10101)
rank_fit <- train(earncdf_child ~ earncdf_father_age_25to29, 
                  data=training,
                  trControl=tc,
                  tuneLength=1,
                  method="lm",
                  metric="FVE",
                  maximize=TRUE)

set.seed(10101)
ols_fit <- train(y=outcome,x=preds, 
                 #data=training,
                 trControl=tc,
                 tuneLength=1,
                 method="lm",
                 metric="FVE",
                 maximize=TRUE)

save.image("models/modelcomp_adaptive.RData")

registerDoSEQ()

#Aggregating data####
#load("models/modelcomp_adaptive.RData")

glmnetdata <- glmnet_fit$resample
glmnetdata$model <- "Elastic Net"

gbmdata <- gbm_fit$resample
gbmdata$model <- "GBM"

# rfdata <- rf_fit$resample
# rfdata$model <- "Random Forest"

rangerdata <- ranger_fit$resample
rangerdata$model <- "Ranger"

xgbdata <- xgb_fit$resample
xgbdata$model <- "XGBoost"

rankdata <- rank_fit$resample
rankdata$model <- "Rank Rank"

olsdata <- ols_fit$resample
olsdata$model <- "OLS (with covariates)"

plotdata <- bind_rows(glmnetdata,gbmdata,rangerdata,xgbdata,rankdata,olsdata)

library(tidyr)

#FVE
plotdata %>% 
    select(-Resample,-Var,-MSE) %>%
    gather(key="Measure",value="Value",-model) %>%
    ggplot(aes(x=reorder(model,Value,FUN=mean),y=Value)) +
    geom_violin(draw_quantiles=c(0.5)) +
    geom_jitter(width=0.1,alpha=0.2) +
    theme_gray(base_size = 16) +
    scale_color_brewer(name="Model",palette = "Set1") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    labs(x="Model",y="FVE")

ggsave("graphs/modelcomp_adaptive_fve_training.pdf",
       height=7,width=7)

plotdata %>% 
    select(-Resample,-Var,-MSE) %>%
    gather(key="Measure",value="Value",-model) %>%
    ggplot(aes(x=reorder(model,Value),y=Value)) +
    stat_summary(fun.data="mean_cl_boot",geom="pointrange") +
    stat_summary(fun.y="mean",geom="point") +
    theme_gray(base_size = 16) +
    scale_color_brewer(name="Model",palette = "Set1") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    labs(x="Model",y="FVE")

ggsave("graphs/modelcomp_adaptive_fve_training_pointrange.pdf",
       height=7,width=7)

plotdata %>% 
    select(-Resample,-Var,-MSE) %>%
    filter(model!="Rank Rank") %>%
    gather(key="Measure",value="Value",-model) %>%
    ggplot(aes(x=reorder(model,Value),y=Value)) +
    stat_summary(fun.data="mean_cl_boot",geom="pointrange") +
    stat_summary(fun.y="mean",geom="point") +
    theme_gray(base_size = 16) +
    scale_color_brewer(name="Model",palette = "Set1") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    labs(x="Model",y="FVE")

ggsave("graphs/modelcomp_adaptive_fve_training_pointrange_norank.pdf",
       height=7,width=7)

#MSE
plotdata %>% 
    select(-Resample,-Var,-FVE) %>%
    gather(key="Measure",value="Value",-model) %>%
    ggplot(aes(x=reorder(model,-Value,FUN=mean),y=Value)) +
    geom_violin(draw_quantiles=c(0.5)) +
    geom_jitter(width=0.1,alpha=0.2) +
    theme_gray(base_size = 16) +
    scale_color_brewer(name="Model",palette = "Set1") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    labs(x="Model",y="MSE")

ggsave("graphs/modelcomp_adaptive_mse_training.pdf",
       height=7,width=7)

#Out of Sample####

modellist <- list("Rank Rank"=rank_fit,
                  "Ranger"=ranger_fit,
                  "OLS (with covariates)"=ols_fit,
                  #"Random Forest"=rf_fit,
                  "Elastic Net"=glmnet_fit,
                  "XGBoost"=xgb_fit,
                  "GBM"=gbm_fit)

testbaked <- bake(prep(rec_obj,
                       training=training),
                  newdata=testing)

preds <- predict(modellist,newdata=testbaked)

#Have to do this one separately because of preprocessing in "baked" data 
preds[["Rank Rank"]]  <- predict(rank_fit,newdata=testing) 

fvefun <- function(outcome,prediction) {
    fve <- 1 - (mean((outcome-prediction)^2,na.rm=TRUE) / var(outcome))
    return(fve)
}

fves <- data.frame(sapply(preds,function(model) 
    fvefun(outcome=testing$earncdf_child,prediction=model)))

colnames(fves) <- "FVE"

fves$Model <- rownames(fves) 

ggplot(data=fves,aes(x=reorder(Model,FVE),y=FVE)) +
    geom_col() +
    theme_grey(base_size = 16) +
    labs(x="Model") +
    theme(axis.text.x = element_text(angle=45,hjust=1))

ggsave("graphs/modelcomp_adaptive_fve_testing.pdf",
       height=7,width=7)

#Estimation time####

times <- sapply(modellist, function(x)
    x$times$everything)

times <- data.frame(times[3,])

colnames(times) <- "Seconds"

times$Model <- rownames(times)

ggplot(data=times,aes(x=reorder(Model,Seconds),y=Seconds/60)) +
    geom_col() +
    geom_label(aes(y=Seconds/60/2,label=round(Seconds/60,0))) +
    theme_grey(base_size = 16) +
    labs(x="Model",y="Estimation Time (minutes)") +
    theme(axis.text.x = element_text(angle=45,hjust=1))

ggsave("graphs/modelcomp_adaptive_times.pdf",
       height=7,width=7)

scatterdata <- left_join(x=times,y=fves,by=c("Model"))

ggplot(data=scatterdata,aes(x=Seconds/60,y=FVE,label=Model)) +
    geom_text(check_overlap = FALSE) +
    theme_grey(base_size = 16) +
    labs(x="Estimation Time (minutes)",y="FVE")
ggsave("graphs/modelcomp_adaptive_scatter.pdf",
       width=16,height=9)

#Possibly useful commands from the caret package####
# 
# summary(resamples(modellist))
# 
# dotplot(resamples(modellist))
# 
# splom(resamples(modellist))
# 
# diffs <- diff(resamples(modellist))
# 
# dotplot(diffs)