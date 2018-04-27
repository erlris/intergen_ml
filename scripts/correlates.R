rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(caret)
library(doMC)
library(recipes)
library(haven)
library(foreach)

load("models/modelcomp_census.RData")

#Make predictions and calculate FVE
baked$xgbpred <- predict(xgb_fit,baked)

baked$county <- training$county_child

baked <- baked %>%
    filter(county!="Other")

baked <- baked %>%
    mutate(sqerror=(earncdf_child-xgbpred)^2) %>%
    group_by(county) %>%
    mutate(var=var(earncdf_child),
           fve=1-mean(sqerror)/var(earncdf_child))

baked <- baked %>%
    select(-matches("yob_child|mob_child|county_child|sqerror|var|xgbpred|earncdf_child"))

baked <- baked %>%
    group_by(county) %>%
    summarize_all(mean,na.rm=TRUE)

baked <- baked %>%
    select(-county)

baked <- baked %>%
    select(-matches("occ|isic|yob|nus"))

corrmatrix <- cor(baked,y=baked$fve,use="everything")

corrmatrix = data.frame(variable=rownames(corrmatrix),
                        correlation=corrmatrix)

corrmatrix <- corrmatrix %>%
    filter(row_number(correlation) <= 5 | row_number(correlation) >= nrow(corrmatrix)-5) %>%
    arrange(correlation)

selectvars <- paste0(corrmatrix$variable,collapse="|")

baked %>%
    select(matches(selectvars)) %>%
    rename("Individuals in Father's Household 1980"=nindh80_father,
           "Individuals in Mother's Household 1980"=nindh80_mother,
           "Gini Coefficient for Mother's Earnings When Child Was 0-4"=meanpearn_mother_age_0to4_gini,
           "Share of Fathers Living in Urban Area 1970" = urban70_father_X2,
           "Share of Mothers Living in Urban Area 1970" = urban70_mother_X2,
           "Mother's Average Earnings Rank Pre Birth"=earncdf_mother_age_prebirth,
           "Mother's Hours Worked 1970"=hours70_mother,
           "Share of Mothers Having Own Employment as Main Income Source"=wact70_mother_X1,
           "Share of Fathers Divorced 1970"=mstat70_father_X4,
           "Share of Mothers Separated 1980"=mstat80_mother_X5,
           "Fraction of Variance Explained"=fve) %>%
    cor(use="everything") %>%
    ggcorrplot::ggcorrplot(type="upper",
                           hc.order=TRUE,
                           legend.title="Correlation",
                           lab=TRUE)

ggsave("graphs/corrplot.pdf",
       height=14,width=14)

#library(partykit)

# tree <- ctree(fve ~ earncdf_mother_age_25to29 + mstat70_mother_X2,
#               data=baked,
#               minsplit=1,
#               minbucket=1,
#               alpha=0.1)
# 
# plot(tree)
