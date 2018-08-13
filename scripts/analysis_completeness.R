rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(haven)
library(forcats)
library(caret)
library(recipes)
library(doMC)

load("data/sampledata_completeness.RData")

#Keep only wanted variables
sampledata <- sampledata %>%
    select(-npid,-nmpid,-nfpid,-eduy_child,
           -nus_child,-gender_child,-matches("meanpearn"),
           -matches("meanwealth"),-matches("ability"),
           -matches("isic"))

#Handle Factors

sampledata <- sampledata %>%
    mutate(occ70_father=factor(str_sub(occ70_father,1,2)),
           occ70_mother=factor(str_sub(occ70_mother,1,2)),
           nus_father=factor(str_sub(nus_father,2,3)),
           nus_mother=factor(str_sub(nus_mother,2,3)))

sampledata <- sampledata %>%
    mutate_at(vars(matches("occ70|emplstat70")),
              funs(fct_explicit_na))

# sampledata <- sampledata %>%
#     mutate_if(is.factor,
#               fct_explicit_na)

sampledata <- sampledata %>%
    mutate_at(vars(matches(("nus|mstat|urban|studa|srcinc|
                            wact|hours|occ|emplstat|hhtype"))),
              fct_lump,
              prop=0.01)
# 
# sampledata <- sampledata %>%
#     mutate_if(is.factor,
#               fct_lump,
#               prop=0.01)

sampledata <- sampledata %>%
    mutate_if(is.factor,
              fct_drop)

#Keep only complete observations
sampledata <- sampledata[complete.cases(sampledata),]

#Draw smaller random sample####
set.seed(10101)
smalldata <- sample_n(sampledata,60000)

#Split into training and test sets####
set.seed(10101)
splitindex <- createDataPartition(smalldata$earncdf_child,
                                  p=0.8,
                                  list=FALSE)

train <- smalldata[splitindex,]
temptest <- smalldata[-splitindex,]

set.seed(10101)
splitindex <- createDataPartition(temptest$earncdf_child,
                                  p=0.5,
                                  list=FALSE)

test1 <- temptest[splitindex,]
test2 <- temptest[-splitindex,]

rm(splitindex,temptest,smalldata)

#Generate cross-validation folds
set.seed(10101)
trainfolds <- createMultiFolds(train$earncdf_child,
                          k=10,
                          times=5)

#Set parameters
tc <- trainControl(method="repeatedcv",
                   number = 10,
                   repeats=5,
                   index = trainfolds,
                   verboseIter = TRUE,
                   returnData=FALSE)

registerDoMC(cores=8) #Remember that cores*threads will be used
threads  <- 1

#Rank-Rank####

print(paste("Rank-Rank"))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father, data=train)

rec_obj <- rec_obj %>% 
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

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

test2data <- as.data.frame(lapply(m1modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Rank-Rank"
test2data$modelnumber <- 1

m1results <- bind_rows(resampledata,test2data)

#Income & wealth####


print(paste("Income & wealth"))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother, data=train)

rec_obj <- rec_obj %>% 
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())

set.seed(10101)
m2_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

set.seed(10101)
m2_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 5)

set.seed(10101)
m2_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 4,
                nthread=threads)

set.seed(10101)
m2_rf <- train(rec_obj,
                   data=train,
                   method="ranger",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 3,
                   num.threads=threads)

m2modellist <- list("OLS"=m2_lm,
                    "ElasticNet"=m2_glmnet,
                    "XGBoost"=m2_xgb,
                    "Ranger"=m2_rf)

resampledata <- as.data.frame(resamples(m2modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income & wealth"
resampledata$modelnumber <- 2
resampledata$Resample <- NULL

test2data <- as.data.frame(lapply(m2modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Income & wealth"
test2data$modelnumber <- 2

m2results <- bind_rows(resampledata,test2data)

#Income, wealth & education length####

print(paste("Income, wealth & education length"))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother, data=train)

rec_obj <- rec_obj %>% 
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())


set.seed(10101)
m3_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

set.seed(10101)
m3_glmnet <- train(rec_obj,
               data=train,
               method="glmnet",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 5)

set.seed(10101)
m3_xgb <- train(rec_obj,
               data=train,
               method="xgbTree",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 4,
               nthread=threads)

set.seed(10101)
m3_rf <- train(rec_obj,
                data=train,
                method="ranger",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 3,
                num.threads=threads)

m3modellist <- list("OLS"=m3_lm,
                  "ElasticNet"=m3_glmnet,
                  "XGBoost"=m3_xgb,
                  "Ranger"=m3_rf)

resampledata <- as.data.frame(resamples(m3modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth & education length"
resampledata$modelnumber <- 3
resampledata$Resample <- NULL

test2data <- as.data.frame(lapply(m3modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Income, wealth & education length"
test2data$modelnumber <- 3


m3results <- bind_rows(resampledata,test2data)

#Income, wealth, education length & occupation####

print(paste("Income, wealth, education length & occupation"))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      occ70_father + occ70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())


set.seed(10101)
m4_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

set.seed(10101)
m4_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 5)

set.seed(10101)
m4_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 4,
                nthread=threads)

set.seed(10101)
m4_rf <- train(rec_obj,
                   data=train,
                   method="ranger",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 3,
                   num.threads=threads)

set.seed(10101)
m4modellist <- list("OLS"=m4_lm,
                    "ElasticNet"=m4_glmnet,
                    "XGBoost"=m4_xgb,
                    "Ranger"=m4_rf)

resampledata <- as.data.frame(resamples(m4modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth, education length & occupation"
resampledata$modelnumber <- 4
resampledata$Resample <- NULL

test2data <- as.data.frame(lapply(m4modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Income, wealth, education length & occupation"
test2data$modelnumber <- 4
m4results <- bind_rows(resampledata,test2data)

#Income, wealth, education length and type & occupation####

print(paste("Income, wealth, education length and type & occupation"))

#Recipe
rec_obj <- recipe(earncdf_child ~ earncdf_father + earncdf_mother + 
                      wealthcdf_father + wealthcdf_mother +
                      eduy_father + eduy_mother +
                      nus_father + nus_mother +
                      occ70_father + occ70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())


set.seed(10101)
m5_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

set.seed(10101)
m5_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 5)

set.seed(10101)
m5_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 4,
                nthread=threads)

set.seed(10101)
m5_rf <- train(rec_obj,
                   data=train,
                   method="ranger",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 3,
                   num.threads=threads)

set.seed(10101)
m5modellist <- list("OLS"=m5_lm,
                    "ElasticNet"=m5_glmnet,
                    "XGBoost"=m5_xgb,
                    "Ranger"=m5_rf)

resampledata <- as.data.frame(resamples(m5modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Income, wealth, education length and type & occupation"
resampledata$modelnumber <- 5
resampledata$Resample <- NULL

test2data <- as.data.frame(lapply(m5modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Income, wealth, education length and type & occupation"
test2data$modelnumber <- 5

m5results <- bind_rows(resampledata,test2data)

#Everything####

print(paste("Everything"))

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
                      nindh70_father + nindh70_mother +
                      hhtype70_father + hhtype70_mother, data=train)

rec_obj <- rec_obj %>% 
    step_dummy(all_nominal()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_predictors())


set.seed(10101)
m6_lm <- train(rec_obj,
               data=train,
               method="lm",
               metric="Rsquared",
               trControl = tc,
               tuneLength = 0)

set.seed(10101)
m6_glmnet <- train(rec_obj,
                   data=train,
                   method="glmnet",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 5)

set.seed(10101)
m6_xgb <- train(rec_obj,
                data=train,
                method="xgbTree",
                metric="Rsquared",
                trControl = tc,
                tuneLength = 4,
                nthread=threads)

set.seed(10101)
m6_rf <- train(rec_obj,
                   data=train,
                   method="ranger",
                   metric="Rsquared",
                   trControl = tc,
                   tuneLength = 3,
                   num.threads=threads)

m6modellist <- list("OLS"=m6_lm,
                    "ElasticNet"=m6_glmnet,
                    "XGBoost"=m6_xgb,
                    "Ranger"=m6_rf)

resampledata <- as.data.frame(resamples(m6modellist),metric="Rsquared")
resampledata$data <- "Training Resamples"
resampledata$variables <- "Everything"
resampledata$modelnumber <- 6
resampledata$Resample <- NULL

test2data <- as.data.frame(lapply(m6modellist,function(x) cor(predict(x,newdata=test2),test2$earncdf_child)^2))
test2data$data <- "Testing"
test2data$variables <- "Everything"
test2data$modelnumber <- 6

m6results <- bind_rows(resampledata,test2data)

registerDoSEQ()

#Plotting####

save.image("models/completeness.RData")

#load("models/completeness.RData")

plotdata <- bind_rows(m1results,m2results,m3results,
                      m4results,m5results,m6results)

plotdata %>%
    gather(key="Model",value="rsquared",-data,
           -variables,-modelnumber) %>%
    ggplot(aes(x=reorder(str_wrap(variables,15),modelnumber),
               y=rsquared,fill=Model)) +
    stat_summary(fun.y = "mean",geom="col",position=position_dodge()) +
    stat_summary(fun.data="mean_se",
                 fun.args = list(mult=2),
                 position=position_dodge(width=0.9),
                 geom="errorbar",
                 aes(width=0.3)) +
    facet_wrap("data") +
    scale_fill_brewer(palette = "Set1") +
    labs(x="Data",y="R-Squared") +
    theme(legend.position = "bottom") +
    guides(fill = guide_legend(title.position = "top",title.hjust = 0.5)) 

ggsave(file="graphs/completeness_modelcomp.pdf",
       height=9,width=16)



#xyplot(resamples(modellist))
#bwplot(resamples(modellist))
#dotplot(resamples(modellist))
#modelCor(resamples(m3modellist))

#Geography####

coords <- read_dta("/data/gis/norge/arbeidsmarked/koordinat_arbeidsmarked.dta")

colnames(coords) <- c("labormarket","id","x","y")

regionnames <- readxl::read_xlsx("~/mlpaper/rawdata/labormarketregions_names.xlsx")

geographydata <- sampledata

geographydata$xgbpred <- predict(m3_xgb, newdata=geographydata)

#Comparison with rank-rank####
geographydata$numericregion <- as.numeric(as.character(geographydata$region_child))

regionlist <- unique(geographydata$numericregion)

estfun <- function(regionnumber) {
    tempfit <- lm(earncdf_child ~ earncdf_father,
                  data=filter(geographydata,numericregion==regionnumber))
    coef <- coef(tempfit)["earncdf_father"]
    results <- data.frame(region=regionnumber,
                          coefficient=coef)
    return(results)
}

library(foreach)

results <- foreach (regionnumber=regionlist,.combine = rbind) %do% {
    estfun(regionnumber)
}

results <- data.frame(results)

results <- inner_join(x=results,y=regionnames,by=c("region"="code"))

geographydata <- inner_join(x=geographydata,y=results,
                            by=c("numericregion"="region"))

geographydata <- geographydata %>%
    group_by(name,location,numericregion) %>%
    mutate(rsquared=cor(earncdf_child,xgbpred)^2) %>%
    summarize(rsquared=mean(rsquared),
              coefficient=mean(coefficient),
              observations=n()) %>%
    ungroup()

geographydata %>%
    ggplot(aes(x=coefficient,y=rsquared)) +
    geom_point(aes(size=observations)) +
    scale_size_continuous(name="Observations",
                          guide=guide_legend(title.position="top")) +
    labs(x="Rank-Rank Slope",
         y="R-Squared") +
    theme_grey(base_size = 16) +
    theme(legend.position="bottom",legend.title.align = 0.5)

ggsave("graphs/completeness_rankcomp_scatter.pdf",
       height=7,width=7)

geographydata %>%
    ggplot(aes(x=coefficient,y=rsquared)) +
    geom_point(aes(size=observations,color=location),) +
    ggrepel::geom_text_repel(aes(label=name,color=location),show.legend=FALSE,alpha=1) +
    scale_size_continuous(name="Observations",guide=guide_legend(title.position="top")) +
    scale_color_brewer(name="Location",palette = "Set1",guide=guide_legend(title.position="top")) +
    labs(x="Rank-Rank Slope",
         y="R-Squared") +
    theme_grey(base_size = 16) +
    theme(legend.position="bottom",legend.title.align = 0.5) 

ggsave("graphs/completeness_rankcomp_scatter_labelled.pdf",
       height=9,width=9)
#Range
geographydata %>%
    select(name,rsquared,coefficient) %>%
    mutate(rsquaredrank=row_number(rsquared),
           "R-Squared"=row_number(rsquared),
           coefrank=row_number(coefficient),
           "Rank-Rank"=row_number(coefficient),
           rankdiff=abs(rsquaredrank-coefrank)) %>%
    select(-rsquared,-coefficient) %>%
    gather(key="Measure",value="Rank",-name,-rsquaredrank,-coefrank,-rankdiff) %>%
    ggplot(aes(x=Rank,y=reorder(name,coefrank),color=Measure,shape=Measure)) +
    geom_segment(aes(x=coefrank,xend=rsquaredrank,yend=reorder(name,coefrank)),color="black") +
    geom_point() +
    scale_color_brewer(name="Ranking Method",palette = "Set1") +
    scale_shape_discrete(name="Ranking Method") +
    labs(x="Rank",y="Labor Market Region") +
    #theme_grey(base_size = 16) +
    theme(legend.position="bottom")

ggsave("graphs/completeness_rankcomp_segments.pdf",
       height=7,width=7)

#Maps
mapdata <- geographydata %>%
    select(numericregion,rsquared,coefficient,observations) %>%
    mutate(fverank=row_number(rsquared),
           coefrank=row_number(coefficient))

mapdata <- inner_join(x=mapdata,y=coords,by=c("numericregion"="labormarket"))

mapdata %>%
    ggplot(aes(x=x,y=y,fill=rsquared,group=numericregion)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="R-Squared",
                         palette = "YlOrRd",direction = 1) +
    theme_void(base_size = 16) +
    theme(legend.position = c(0.2,0.7))

ggsave("graphs/completeness_rankcomp_map_rsquared.pdf",
       height=7,width=7)

mapdata %>%
    ggplot(aes(x=x,y=y,fill=coefficient,group=numericregion)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank Slope",
                         palette = "YlOrRd",direction = 1) +
    theme_void(base_size = 16) +
    theme(legend.position = c(0.2,0.7))

ggsave("graphs/completeness_rankcomp_map_slope.pdf",
       height=7,width=7)

#Gatsby####

geographydata$variance <- sapply(geographydata$numericregion,function(x)
    var(subset(sampledata,region_child==x)$earncdf_father))

geographydata %>%
    select(numericregion,rsquared,variance,coefficient,observations) %>%
    group_by(numericregion) %>%
    summarize_all(mean) %>%
    ggplot(aes(x=variance,y=rsquared)) +
    geom_smooth(method="lm") +
    geom_point(aes(size=observations)) +
    theme(legend.position = "bottom") +
    scale_size_continuous(name="Observations") +
    labs(x="Variance in Father's Earnings Percentile",
         y="FVE")

ggsave("graphs/completeness_rankcomp_gatsby.pdf",
       height=7,width=7)

#LM for observations

options(scipen=999)

geographydata %>%
    ggplot(aes(x=observations,y=rsquared)) +
    geom_smooth(method="lm") +
    geom_point(aes(size=observations)) +
    #scale_x_log10() +
    theme(legend.position = "bottom") +
    scale_size_continuous(name="Observations") +
    labs(x="Observations",
         y="R-Squared")

ggsave("graphs/completeness_rsquared_observations.pdf",
       height=7,width=7)

#Variable Importance

importance <- varImp(m3_xgb)

importance <- data.frame(importance$importance)

importance$variable <- rownames(importance)

ggplot(data=importance,aes(x=reorder(variable,Overall),y=Overall)) +
    geom_col() +
    coord_flip() +
    labs(x="Variable",y="Importance") +
    theme_grey(base_size = 16)

ggsave(file="graphs/completeness_varimp.pdf",
       height=7,width=7)



