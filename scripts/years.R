rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(caret)
library(doMC)
library(recipes)
library(haven)
library(foreach)

load("data/sampledata.RData")

sampledata <- sampledata %>%
    group_by(yob_child) %>%
    mutate(earncdf_child = cume_dist(meanpearn_child)) %>%
    ungroup()

matchnames <- c("earncdf_child",
                "eduy_child",
                "gender_child",
                "yob_child",
                "earncdf_father",
                "earncdf_mother",
                "eduy_father",
                "eduy_mother",
                "occ",
                "_mob",
                "mothermob_child")

expr <- paste(matchnames,collapse="|")

sampledata <- sampledata %>%
    select(matches(expr))

rm(matchnames,expr)

#Occupation factors
sampledata <- sampledata %>%
    mutate(occ70_father = ifelse(is.na(occ70_father),"", occ70_father),
           occ80_father = ifelse(is.na(occ80_father),"", occ80_father),
           occ70_mother = ifelse(is.na(occ70_mother),"", occ70_mother),
           occ80_mother = ifelse(is.na(occ80_mother),"", occ80_mother))

sampledata$occ70_father <- factor(sampledata$occ70_father,exclude=NULL)
sampledata$occ80_father <- factor(sampledata$occ80_father,exclude=NULL)
sampledata$occ70_mother <- factor(sampledata$occ70_mother,exclude=NULL)
sampledata$occ80_mother <- factor(sampledata$occ80_mother,exclude=NULL)

#Gender factor
sampledata$gender_child  <- ifelse(sampledata$gender_child == 1, "Male","Female")
sampledata$gender_child  <- factor(sampledata$gender_child)

#Municipality of birth factor
sampledata$mob_child <- factor(sampledata$mothermob_child)
sampledata$mothermob_child <- NULL

sampledata <- sampledata[complete.cases(sampledata),]

#Generate County Factor (could do this in data wrangling script)
counties <- as.character(sampledata$mob_child)

counties[str_length(counties) < 4]  <- paste0(0,counties[str_length(counties) < 4])

counties <- substring(counties,1,2)

sampledata$county_child <- counties

sampledata <- sampledata %>%
    filter(county_child!=99) %>%
    mutate(county_child=factor(county_child))

rm(counties)

#Redefine year of birth to factor
sampledata <- sampledata %>%
    mutate(yob_child=factor(yob_child))

#Drop gender and child and check for missing obsevations
sampledata <- select(sampledata,-gender_child,-eduy_child)

sampledata <- sampledata[complete.cases(sampledata),]

#Define Summary Function, TrainControl and Search Grid####
summaryfun <- function(data, lev=NULL, model = NULL) {
    var <- var(data$obs,na.rm=TRUE)
    mse <- mean((data$obs-data$pred)^2,na.rm=TRUE)
    fve <- 1 - mse/var
    
    summary <- c(mse,var,fve)
    names(summary) <- c("MSE","Var","FVE")
    return(summary)
}

tc <- trainControl(method="adaptive_cv",
                   number = 5,
                   repeats=10,
                   adaptive=list(min=5,
                                 alpha=0.05,
                                 method="gls",
                                 complete=TRUE),
                   verboseIter = TRUE,
                   #seeds=seeds,
                   summaryFunction=summaryfun,
                   returnData=FALSE)

#Write estimation function####
estfun <- function(year) {
    tempdata <- sampledata %>% 
        filter(yob_child==year) %>%
        select(-yob_child)
    
    rec_obj <- recipe(earncdf_child ~ .,
                      data=tempdata)
    
    rec_obj <- rec_obj %>% 
        step_dummy(contains("occ"),mob_child,county_child) %>%
        step_nzv(all_predictors(),
                 options=list(freq_cut=200/1,
                              unique_cut=0.0025)) %>%
        step_corr(all_predictors(),use="everything") %>%
        step_center(all_predictors(),-contains("county_child|occ|mob_child")) %>%
        step_scale(all_predictors(),-contains("county_child|occ|mob_child"))
    
    tempdata <- juice(prep(rec_obj,retain=TRUE))
    
    outcome <- tempdata$earncdf_child
    preds <- data.frame(select(tempdata,-earncdf_child))
    
    rm(tempdata)
    
    set.seed(10101)
    xgb_fit <- train(y=outcome,x=preds,
                     trControl=tc,
                     #tuneGrid=grid,
                     tuneLength=3,
                     method="xgbTree",
                     metric="FVE",
                     maximize=TRUE,
                     nthread=threads,
                     verbose=0)
    
    results <- xgb_fit$resample
    results$year <- year
    return(results)
}

registerDoMC(cores=6) #Remember that the function will use threads*cores
threads <- 1

years <- unique(sampledata$yob_child)

#sampledata <- sample_n(sampledata,5000)

results <- foreach (year = years,.combine = rbind) %do% {
    print(paste("Start",year,Sys.time()))
    estfun(year)
}

saveRDS(results,"models/yearresults.rds")

#results <- readRDS("models/yearresults.rds")

#Comparison to Full Model####
load("models/modelcomp_adaptive.RData")

yeardata <- training

yeardata$xgbpred <- predict(xgb_fit,baked) #This needs to be baked to have all predictors

plotdata <- yeardata %>%
    mutate(sqerror=(earncdf_child-xgbpred)^2) %>%
    group_by(yob_child) %>%
    mutate(var=var(earncdf_child),
           fve=1-mean(sqerror)/var(earncdf_child)) %>%
    summarize(fve_fullmodel=mean(fve))

plotdata <- left_join(x=plotdata,y=results,by=c("yob_child"="year"))

plotdata %>%
    ggplot(aes(x=factor(yob_child))) +
    stat_summary(aes(y=FVE,color="By Year",shape="By Year"),
                 fun.data = "mean_cl_boot",
                 geom="pointrange") +
    stat_summary(aes(y=fve_fullmodel,color="Full Sample",shape="Full Sample"),
               size=5,
               fun.y="mean",
               geom="point") +
    theme_grey(base_size = 16) +
    labs(x="Child's Year of Birth") +
    scale_color_brewer(name="Model",palette = "Set1") +
    scale_shape_discrete(name="Model")

ggsave("graphs/modelcomp_year_pointrange.pdf",
       width=16,height=9)

plotdata %>%
    ggplot(aes(x=factor(yob_child))) +
    geom_violin(aes(y=FVE),draw_quantiles=c(0.5)) +
    geom_jitter(aes(y=FVE,color="By Year",shape="By Year"),width=0.1,alpha=0.2) +
    geom_point(aes(y=fve_fullmodel,color="Full Sample",shape="Full Sample"),size=5) +
    theme_grey(base_size = 16) +
    labs(x="Child's Year of Birth") +
    scale_color_brewer(name="Model",palette = "Set1") +
    scale_shape_discrete(name="Model")

ggsave("graphs/modelcomp_year_violin.pdf",
       width=16,height=9)

plotdata %>%
    group_by(yob_child) %>%
    summarize(full=mean(fve_fullmodel),
              separate=mean(FVE)) %>%
    ggplot(aes(x=separate,y=full)) +
    geom_text(aes(label=yob_child)) +
    labs(x="FVE from Estimation by Year",y="FVE from Estimation on Full Sample") +
    theme_grey(base_size = 16)

ggsave("graphs/modelcomp_year_point.pdf",
       width=16,height=9)
