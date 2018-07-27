rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(caret)
library(recipes)
library(doMC)
library(foreach)

load("data/sampledata_classification.RData")

#Check this in makedata script
sampledata$mob_child <- as.factor(sampledata$mob_child)

#Sample restrictions####

#Keep only second and third digit of nus
sampledata <- sampledata %>%
    mutate(edutype_father = factor(str_sub(nus_father,2,3)),
           edutype_mother = factor(str_sub(nus_mother,2,3)))

#Generate parents' age at birth
sampledata <- sampledata %>%
    mutate(age_father = factor(yob_child - yob_father),
           age_mother = factor(yob_child - yob_mother))

#Lump factors
# sampledata <- sampledata %>%
#     mutate(mob_child = fct_lump(mob_child,prop=1/200),
#            edutype_father = fct_lump(edutype_father,prop=1/200),
#            edutype_mother = fct_lump(edutype_mother,prop=1/200),
#            age_father = fct_lump(age_father,prop=1/200),
#            age_mother = fct_lump(age_mother,prop=1/200))


#Restrict birth years
sampledata <- sampledata %>%
    filter(yob_child >= 1970 & yob_child <= 1975)

sampledata <- sampledata %>%
    select(earncdf_child,earncdf_father,earncdf_mother,
           wealthcdf_father,wealthcdf_mother,
           eduy_father,eduy_mother,
           edutype_father,edutype_mother,
           age_father,age_mother,
           mob_child)

sampledata <- sampledata[complete.cases(sampledata),]

####Draw Random Sample####
set.seed(10101)

sampledata <- sample_n(sampledata,100000)

####Split into training and control####
set.seed(10101)
splitindex <- createDataPartition(y = sampledata$earncdf_child,
                                  p=0.8,
                                  list=FALSE)

training <- sampledata[splitindex,]
testing <- sampledata[-splitindex,]

rm(splitindex,sampledata)

#Estimating models
tc_lm <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 5,
                     returnData=FALSE,
                     verboseIter=TRUE,
                     allowParallel = TRUE)

tc_rf <- trainControl(method = "repeatedcv",
                   number = 5,
                   repeats = 5,
                   returnData=FALSE,
                   verboseIter=TRUE,
                   allowParallel = FALSE)

threads <- 16

#Grid for one variable ranger

grid <- expand.grid(mtry=1,
                    min.node.size=5,
                    splitrule="variance")


registerDoMC(cores=8)

#Rank-Rank

rec_obj = recipe(earncdf_child ~ earncdf_father,
                 data=training)

rec_obj <- rec_obj %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors())

set.seed(10101)
m1_lm <- train(rec_obj,
               data=training,
               trControl=tc_lm,
               method="lm")

set.seed(10101)
m1_rf <- train(rec_obj,
               data=training,
               trControl=tc_rf,
               method="ranger",
               num.threads=threads,
               tuneGrid = grid) 

#Income and wealth

rec_obj = recipe(earncdf_child ~ earncdf_father + earncdf_mother +
                     wealthcdf_father + wealthcdf_mother,
                 data=training)

rec_obj <- rec_obj %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors()) %>%
    step_nzv(all_predictors())

set.seed(10101)
m2_lm <- train(rec_obj,
               data=training,
               trControl=tc_lm,
               method="lm")

set.seed(10101)
m2_rf <- train(rec_obj,
               data=training,
               trControl=tc_rf,
               method="ranger",
               num.threads=threads) 

#Income, wealth and education length
rec_obj = recipe(earncdf_child ~ earncdf_father + earncdf_mother +
                     wealthcdf_father + wealthcdf_mother + 
                     eduy_father + eduy_mother,
                 data=training)

rec_obj <- rec_obj %>%
    step_center(all_predictors()) %>%
    step_scale(all_predictors()) %>%
    step_nzv(all_predictors())


set.seed(10101)
m3_lm <- train(rec_obj,
               data=training,
               trControl=tc_lm,
               method="lm")

set.seed(10101)
m3_rf <- train(rec_obj,
               data=training,
               trControl=tc_rf,
               method="ranger",
               num.threads=threads) 

#Income, wealth, education and age
rec_obj = recipe(earncdf_child ~ earncdf_father + earncdf_mother +
                     wealthcdf_father + wealthcdf_mother + 
                     eduy_father + eduy_mother +
                     age_father + age_mother,
                 data=training)

rec_obj <- rec_obj %>%
    step_center(all_predictors(),-matches("age")) %>%
    step_scale(all_predictors(),-matches("age")) %>%
    step_dummy(matches("age")) %>%
    step_nzv(all_predictors())


set.seed(10101)
m4_lm <- train(rec_obj,
               data=training,
               trControl=tc_lm,
               method="lm")


set.seed(10101)
m4_rf <- train(rec_obj,
               data=training,
               trControl=tc_rf,
               method="ranger",
               num.threads=threads) 

#Income, wealth, education, age, education type and municipality
rec_obj = recipe(earncdf_child ~ earncdf_father + earncdf_mother +
                     wealthcdf_father + wealthcdf_mother + 
                     eduy_father + eduy_mother +
                     age_father + age_mother + 
                     edutype_father + edutype_mother +
                     mob_child,
                 data=training)

rec_obj <- rec_obj %>%
    step_center(all_predictors(),-matches("age|edutype|mob")) %>%
    step_scale(all_predictors(),-matches("age|edutype|mob")) %>%
    step_dummy(matches("age|edutype|mob")) %>%
    step_nzv(all_predictors())


set.seed(10101)
m5_lm <- train(rec_obj,
               data=training,
               trControl=tc_lm,
               method="lm")

set.seed(10101)
m5_rf <- train(rec_obj,
               data=training,
               trControl=tc_rf,
               method="ranger",
               num.threads=threads)

registerDoSEQ()

modellist <- list(m1_lm,
                  m1_rf,
                  m2_lm,
                  m2_rf,
                  m3_lm,
                  m3_rf,
                  m4_lm,
                  m4_rf,
                  m5_lm,
                  m5_rf)

insamplefun <- function(x) {
    temp <- postResample(pred = predict(x,newdata=training),
                 obs=training$earncdf_child)
    return(temp["Rsquared"])
}

holdoutfun <- function(x) {
    temp <- postResample(pred = predict(x,newdata=testing),
                         obs=testing$earncdf_child)
    return(temp["Rsquared"])
}

q1fun <- function(x) {
    temptesting <- filter(testing,
                          earncdf_child < 0.2)
    temp <- postResample(pred = predict(x,newdata=temptesting),
                         obs=temptesting$earncdf_child)
    return(temp["Rsquared"])
}

q2fun <- function(x) {
    temptesting <- filter(testing,
                          earncdf_child >= 0.2 & earncdf_child < 0.4)
    temp <- postResample(pred = predict(x,newdata=temptesting),
                         obs=temptesting$earncdf_child)
    return(temp["Rsquared"])
}

q3fun <- function(x) {
    temptesting <- filter(testing,
                          earncdf_child >= 0.4 & earncdf_child < 0.6)
    temp <- postResample(pred = predict(x,newdata=temptesting),
                         obs=temptesting$earncdf_child)
    return(temp["Rsquared"])
}

q4fun <- function(x) {
    temptesting <- filter(testing,
                          earncdf_child >= 0.6 & earncdf_child < 0.8)
    temp <- postResample(pred = predict(x,newdata=temptesting),
                         obs=temptesting$earncdf_child)
    return(temp["Rsquared"])
}

q5fun <- function(x) {
    temptesting <- filter(testing,
                          earncdf_child >= 0.8)
    temp <- postResample(pred = predict(x,newdata=temptesting),
                         obs=temptesting$earncdf_child)
    return(temp["Rsquared"])
}

save.image("models/regressionresults.RData")

####Loading####

#load("models/regressionresults.RData")
#library(ranger)

specification <- c("Linear",
                   "Flexible",
                   "Linear",
                   "Flexible",
                   "Linear",
                   "Flexible",
                   "Linear",
                   "Flexible",
                   "Linear",
                   "Flexible")

variables <- c("Father's income",
               "Father's income",
               "Income & wealth",
               "Income & wealth",
               "Income, wealth & education length",
               "Income, wealth & education length",
               "Income, wealth, education length & age",
               "Income, wealth, education length & age",
               "Income, wealth, education length and type, age & municipality",
               "Income, wealth, education length and type, age & municipality")

insample <- sapply(modellist,insamplefun)
holdout <- sapply(modellist,holdoutfun)
q1 <- sapply(modellist,q1fun)
q2 <- sapply(modellist,q2fun)
q3 <- sapply(modellist,q3fun)
q4 <- sapply(modellist,q4fun)
q5 <- sapply(modellist,q5fun)

# q1 <- round(q1,3)
# q2 <- round(q2,3)
# q3 <- round(q3,3)
# q4 <- round(q4,3)
# q5 <- round(q5,3)

results <- data.frame(variables,specification,insample,holdout,
                      q1,q2,q3,q4,q5)

results

####Plot####

#Fixed y-axis
results %>%
    rename("Quintile 1"=q1,"Quintile 2"=q2,"Quintile 3"=q3,
           "Quintile 4"=q4,"Quintile 5"=q5,"Testing"=holdout,
           "Training"=insample) %>%
    mutate(specification=ifelse(specification=="Linear","Linear","Ranger")) %>%
    gather(-c(variables,specification),key="sample",value="rsquared") %>%
    ggplot(aes(x=str_wrap(variables,30),y=rsquared,fill=specification)) +
    geom_col(position="dodge") +
    facet_wrap("sample") +
    theme(legend.position = c(0.5,0)) +
    scale_fill_brewer(name="Model",palette = "Set1") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    labs(x="Included Variables",y="R-Squared")

ggsave(filename = "graphs/regression_quantiles_fixedy.pdf",
       width=16,height=9)

####Plotting Predictions vs observed####

m1_lmdf <- data.frame(predicted = predict(m1_lm,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Linear",
                      variables="Father's income")

m1_rfdf <- data.frame(predicted = predict(m1_rf,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Ranger",
                      variables="Father's income")

m2_lmdf <- data.frame(predicted = predict(m2_lm,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Linear",
                      variables="Income & wealth")

m2_rfdf <- data.frame(predicted = predict(m2_rf,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Ranger",
                      variables="Income & wealth")

m3_lmdf <- data.frame(predicted = predict(m3_lm,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Linear",
                      variables="Income, wealth & education length")

m3_rfdf <- data.frame(predicted = predict(m3_rf,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Ranger",
                      variables="Income, wealth & education length")

m4_lmdf <- data.frame(predicted = predict(m4_lm,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Linear",
                      variables="Income, wealth, education length & age")

m4_rfdf <- data.frame(predicted = predict(m4_rf,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Ranger",
                      variables="Income, wealth, education length & age")

m5_lmdf <- data.frame(predicted = predict(m5_lm,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Linear",
                      variables="Income, wealth, education length and type, age & municipality")

m5_rfdf <- data.frame(predicted = predict(m5_rf,newdata=testing),
                      observed = testing$earncdf_child,
                      model="Ranger",
                      variables="Income, wealth, education length and type, age & municipality")

predictdf <- rbind(m1_lmdf,
                   m1_rfdf,
                   m2_lmdf,
                   m2_rfdf,
                   m3_lmdf,
                   m3_rfdf,
                   m4_lmdf,
                   m4_rfdf,
                   m5_lmdf,
                   m5_rfdf)

predictdf %>%
    ggplot(aes(x=observed,y=predicted,fill=model,color=model)) +
    stat_summary_bin(fun.data="mean_se",
                     fun.args = list(mult=2),
                     binwidth=0.02,
                     geom="ribbon",
                     alpha=0.2,
                     aes(color=NULL)) +
    stat_summary_bin(fun.y="mean",
                     binwidth=0.02,
                     geom="line") +
    stat_summary_bin(fun.y="mean",
                     binwidth=0.02,
                     geom="point") +
    facet_wrap("variables") +
    scale_fill_brewer(name="Model",
                      palette = "Set1") +
    scale_color_brewer(name="Model",
                      palette = "Set1") +
    theme(legend.position = c(0.85,0.25)) +
    labs(x="Observed Values",y="Predicted Values")

ggsave(filename = "graphs/regression_obsvspred.pdf",
       width=16,height=9)





