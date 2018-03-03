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

sampledata$county_child <- as.numeric(counties)

rm(counties)

#Merge with county names####

countydata <- read_dta("/data/gis/norge/fylke/datafylke.dta")

countydata <- countydata %>%
    select(NAVN,id) %>%
    rename(countyname=NAVN) %>%
    mutate(id=as.integer(id),
           countyname=as.character(countyname))

sampledata <- left_join(x=sampledata,y=countydata,by=c("county_child"="id"))

rm(countydata)

#Redefine year of birth to factor
sampledata <- sampledata %>%
    mutate(yob_child=factor(yob_child))

#Draw smaller random sample and split into training and test set
sampledata <- select(sampledata,-gender_child,-eduy_child,-county_child)

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

# grid <- expand.grid(nrounds=seq(50,400,50),
#                     max_depth=seq(1,5,1),
#                     eta=c(0.05,0.1,0.2,0.3),
#                     gamma=0,
#                     colsample_bytree=seq(0.5,0.9,0.2),
#                     min_child_weight=1,
#                     subsample=seq(0.5,1,0.25))

#Write estimation function####
estfun <- function(county) {
    tempdata <- sampledata %>% 
        filter(countyname==county) %>%
        select(-countyname)
    
    rec_obj <- recipe(earncdf_child ~ .,
                      data=tempdata)

    rec_obj <- rec_obj %>% 
        step_dummy(yob_child,contains("occ"),mob_child) %>%
        step_nzv(all_predictors(),
                 options=list(freq_cut=200/1,
                              unique_cut=0.0025)) %>%
        step_corr(all_predictors(),use="everything") %>%
        step_center(all_predictors(),-contains("yob_child|occ|mob_child")) %>%
        step_scale(all_predictors(),-contains("yob_child|occ|mob_child"))
    
    tempdata <- bake(prep(rec_obj),
                     newdata=tempdata)
    
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
    results$county <- county
    return(results)
}

registerDoMC(cores=8) #Remember that the function will use threads*cores
threads <- 1

counties <- unique(sampledata$countyname)

#sampledata <- sample_n(sampledata,20000)

results <- foreach (county = counties,.combine = rbind) %do% {
             print(paste("Start",county,Sys.time()))
             estfun(county)
}

saveRDS(results,"models/regionresults.rds")

#results <- readRDS("models/regionresults.rds")

#Comparison to Full Model####
load("models/modelcomp_adaptive.RData")

countydata <- read_dta("/data/gis/norge/fylke/datafylke.dta")

countydata <- countydata %>%
    select(NAVN,id) %>%
    rename(countyname=NAVN) %>%
    mutate(id=as.integer(id),
           countyname=as.character(countyname))

geographydata <- training

geographydata$xgbpred <- predict(xgb_fit,baked) #This needs to be baked to have all predictors

geographydata <- geographydata %>%
    mutate(county_child=as.integer(as.character(county_child)))

geographydata <- left_join(x=geographydata,y=countydata,by=c("county_child"="id"))

plotdata <- geographydata %>%
    mutate(sqerror=(earncdf_child-xgbpred)^2) %>%
    group_by(countyname) %>%
    mutate(var=var(earncdf_child),
           fve=1-mean(sqerror)/var(earncdf_child)) %>%
    summarize(fve_fullmodel=mean(fve))

plotdata <- left_join(x=plotdata,y=results,by=c("countyname"="county"))

plotdata %>%
    filter(!is.na(countyname)) %>%
ggplot(aes(x=factor(countyname))) +
    stat_summary(aes(y=FVE,color="By County",shape="By County"),
                 fun.data = "mean_cl_boot",
                 geom="pointrange") +
    geom_point(aes(y=fve_fullmodel,color="Full Sample",shape="Full Sample"),
               size=5) +
    theme_grey(base_size = 16) +
    labs(x="County") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    scale_color_brewer(name="Model",palette = "Set1") +
    scale_shape_discrete(name="Model")

ggsave("graphs/modelcomp_region_pointrange.pdf",
       width=16,height=9)

plotdata %>%
    filter(!is.na(countyname)) %>%
    ggplot(aes(x=factor(countyname))) +
    geom_violin(aes(y=FVE),draw_quantiles=c(0.5)) +
    geom_jitter(aes(y=FVE,color="By County",shape="By County"),width=0.1,alpha=0.2) +
    geom_point(aes(y=fve_fullmodel,color="Full Sample",shape="Full Sample"),size=5) +
    theme_grey(base_size = 16) +
    labs(x="County") +
    theme(axis.text.x = element_text(angle=45,hjust=1)) +
    scale_color_brewer(name="Model",palette = "Set1") +
    scale_shape_discrete(name="Model")

ggsave("graphs/modelcomp_region_violin.pdf",
       width=16,height=9)

plotdata %>%
    filter(!is.na(countyname)) %>%
    group_by(countyname) %>%
    summarize(full=mean(fve_fullmodel),
              separate=mean(FVE)) %>%
    ggplot(aes(x=separate,y=full)) +
    geom_text(aes(label=countyname)) +
    labs(x="FVE from Estimation by County",y="FVE from Estimation on Full Sample") +
    theme_grey(base_size = 16)

ggsave("graphs/modelcomp_region_point.pdf",
       width=16,height=9)