### Graphs for paper
### Intergenerational mobility as a prediction problem
# Jack Blundell and Erling Risa

### 0. Setup ------

rm(list = ls()) 	
options("scipen"=100, "digits"=4)

set.seed(123)

setwd("/Users/jack/git_repos/intergen_ml") # Jack
#setwd("/home/erling/git/intergen_ml") #Erling on server
#setwd("C:/Users/s12864/Documents/Git/intergen_ml") #Erling on laptop

# Load packages

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(stargazer)
library(haven)

# set default ggplot size
theme_set(theme_gray(base_size = 18))

### 1. Load results ------

res <- as_tibble(readRDS("data/modelresults.rds"))
res.reg <- as_tibble(readRDS("data/regionresults.rds"))

### 2. Completeness ------

### 2.1 Test

test <- res %>% filter(data == "Testing",
                       variables %in% c("Rank-Rank",
                                        "Income with multiple functional forms",
                                        "Income & wealth",
                                        "Income & education length",
                                        "Income, wealth & education length")) %>%
  select(OLS, variables, ElasticNet, XGBoost, modelnumber)

#test$variables[which(test$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
test$variables[which(test$variables == "Income with multiple functional forms")] <- "Income (flexible)"
test$ElasticNet[which(test$variables == "Income (flexible)")] <- NA


test %>%
  gather(key="Model",value="rsquared", - variables, - modelnumber) %>%
  ggplot(aes(x=reorder(str_wrap(variables,15), modelnumber),
             y=rsquared,
             fill=factor(Model,levels=c("OLS", "ElasticNet", "XGBoost"),
                         labels = c("OLS", "Elastic Net", "Gradient Boosted Trees")))) +
  stat_summary(fun.y = "mean", geom="col", position=position_dodge(),
               color="black") +
  scale_fill_brewer(palette = "Blues",name="Estimator") +
  labs(x="Included Variables",y="R-Squared") +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(title.position = "top",title.hjust = 0.5))

ggsave(file="graphs/test_R2.pdf",
       height=08,width=11)

### 2.2 Train

train <- res %>% filter(data == "Training Resamples",
                       variables %in% c("Rank-Rank",
                                        "Income with multiple functional forms",
                                        "Income & wealth",
                                        "Income & education length",
                                        "Income, wealth & education length")) %>%
  select(OLS, variables, ElasticNet, XGBoost, modelnumber)

#train$variables[which(train$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
train$variables[which(train$variables == "Income with multiple functional forms")] <- "Income (flexible)"
train$ElasticNet[which(train$variables == "Income (flexible)")] <- NA

train %>%
  gather(key="Model", value="rsquared",- variables, - modelnumber) %>%
  ggplot(aes(x = reorder( str_wrap(variables, 15), modelnumber),
             y = rsquared,
             fill = factor(Model, levels = c("OLS", "ElasticNet", "XGBoost"),
                         labels = c("OLS", "Elastic Net", "Gradient Boosted Trees")))) +
  stat_summary(fun.y = "mean", geom = "col",position=position_dodge(),
               color = "black") +
  stat_summary(fun.data = "mean_se",
               fun.args = list(mult=2),
               position = position_dodge(width=0.9),
               geom = "errorbar",
               aes(width=0.3)) +
  scale_fill_brewer(palette = "Blues",name="Estimator") +
  labs(x = "Included Variables", y = "R-Squared") +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(title.position = "top", title.hjust = 0.5))

ggsave(file = "graphs/train_R2.pdf",
       height = 08, width = 11)

### 2.3 Tables

train$variables <- gsub(x = train$variables, pattern = "&", replacement = "and")
train.tbl <- train %>%
  gather(key = "Model", value = "rsquared",- variables, - modelnumber) %>%
  group_by(variables, Model) %>%
  summarize("Training R2 (mean)" = round(mean(rsquared), digits = 3),
            "Training R2 (sd)" = round(sd(rsquared), digits = 4)) %>%
  filter(is.na(`Training R2 (mean)`) == F) %>%
  mutate(completeness = round(`Training R2 (mean)`/0.0490, digits = 2))
stargazer(train.tbl, summary = F)

test$variables <- gsub(x = test$variables, pattern = "&", replacement = "and")
test.tbl <- test %>%
  gather(key = "Model", value = "rsquared",- variables, - modelnumber) %>%
  group_by(variables, Model) %>%
  summarize("Test R2" = round(mean(rsquared), digits = 3)) %>%
  filter(is.na(`Test R2`) == F) %>%
  mutate(completeness = round(`Test R2`/0.0480, digits = 2))
stargazer(test.tbl, summary = F)

### 3. Regional results

# generate completeness index

res.reg.clean <- res.reg %>% 
  mutate(completeness=rsquared_rank/rsquared_full) %>%
  mutate(completeness=ifelse(observations>=200,completeness,NA))

# Table of results by region

res.reg.table <- res.reg %>% mutate(completeness = rsquared_rank/ rsquared_full) %>%
  mutate(earn_sd = round(earn_sd, digits = 2),
         edu_sd = round(edu_sd, digits = 2),
         wealth_sd = round(wealth_sd, digits = 2),
         earn_edu_cor = round(earn_edu_cor, digits = 2),
         earn_wealth_cor = round(earn_wealth_cor, digits = 2),
         rsquared_rank = round(rsquared_rank, digits = 3),
         rsquared_full = round(rsquared_full, digits = 3),
         completeness = round(completeness, digits = 3),
         rank_coef = round(rank_coef, digits = 2)) %>% 
  select(name, 
         obs=observations,
         completeness,
         "rr coef" = rank_coef,
         #"earn (sd)"=earn_sd,
         #"edu (sd)"=edu_sd, 
         #"wealth (sd)"=wealth_sd, 
         "earn/edu corr"=earn_edu_cor,
         "earn/wealth corr"=earn_wealth_cor,
         "R2 (rank)"=rsquared_rank, 
         "R2 (full)"=rsquared_full) %>% 
  arrange(completeness)

stargazer(res.reg.table,
          summary = F,
          column.sep.width = "2pt",
          out="tables/region_results.tex",
          float=F,
          rownames = F)

### 4. Maps ------

coordinates <- read_dta("data/coordinates_labormarkets.dta")

colnames(coordinates) <- c("region","id","x","y")

coordinates$region <- factor(coordinates$region)

### 3.1 Completeness by region

res.reg %>% 
    mutate(completeness=rsquared_rank/rsquared_full) %>%
    #mutate(completeness=ifelse(observations>=200,completeness,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=completeness)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Completeness",
                         palette = "Blues",
                         direction = 1,
                         limits = c(0.2, 1)) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/completeness_map.pdf",
       height=7,width=7)


### 3.2 Rank-rank R2 by region

res.reg %>% 
    mutate(rsquared_rank=ifelse(observations>=200,rsquared_rank,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=rsquared_rank)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank\nR-squared",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rsquared_rank_map.pdf",
       height=7,width=7)

### 3.3 Full R2 by region

res.reg %>% 
    mutate(rsquared_full=ifelse(observations>=200,rsquared_full,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=rsquared_full)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Full Model\nR-squared",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rsquared_full_map.pdf",
       height=7,width=7)

### 3.4 Rank-Rank Slope by Region

res.reg %>% 
    mutate(rank_coef=ifelse(observations>=200,rank_coef,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=rank_coef)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank\nSlope",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rankslope_map.pdf",
       height=7,width=7)

### 4. Scatter plots by region ------

### Prepare data

res.reg.scatter <- res.reg %>% mutate(completeness = rsquared_rank/ rsquared_full) %>%
  select(region, observations, earn_sd, edu_sd, wealth_sd, earn_edu_cor, earn_wealth_cor,
         rsquared_rank, rsquared_full, completeness, name)
  
### 4.1 Rank-rank R2 and complete R2 ------

plotdata <- res.reg.scatter %>%
  ggplot(aes(x = rsquared_rank, y = rsquared_full, size = observations)) +
  geom_point() + xlab("rank-rank R2") +
  ylab("Full model R2")
plotdata
ggsave(file = "graphs/regions_R2.pdf",
       height = 06, width = 10)

### 4.2 Completeness against distribution of other predictors ------

res.reg.scatter <- res.reg.scatter %>% mutate(name.lab = "")
res.reg.scatter$name.lab[which(res.reg.scatter$name %in% c("Alta", "Hallingdal"))] <- res.reg.scatter$name[which(res.reg.scatter$name %in% c("Alta", "Hallingdal"))]

res.reg.scatter %>%
  filter(observations > 200) %>%
  select(completeness, #earn_sd, edu_sd, wealth_sd,
         earn_edu_cor, earn_wealth_cor, name.lab) %>%
  rename(
         #"Wealth sd" = wealth_sd,
         "Income and Education corr" = earn_edu_cor,
         "Income and Wealth corr" = earn_wealth_cor) %>%
         #"Income sd" = earn_sd,
         #"Education sd" = edu_sd) %>%
  gather(key = variable, value = value, - completeness, - name.lab) %>%
  ggplot(aes(x = value, y = completeness, label = name.lab)) +
  geom_text(aes(label = name.lab), hjust = -.2) +
  geom_point() + 
  geom_smooth(method = "lm") +
  labs(x="Value", y = "Completeness") +
  facet_wrap("variable", scales = "free_x")

ggsave(file = "graphs/completeness_dist.pdf",
       height = 09, width = 11)

### associated calculations

summary(lm(completeness ~ earn_edu_cor, data = res.reg.scatter))
summary(lm(completeness ~ earn_wealth_cor, data = res.reg.scatter))
summary(lm(completeness ~ earn_sd, data = res.reg.scatter))
summary(lm(completeness ~ edu_sd, data = res.reg.scatter))
summary(lm(completeness ~ wealth_sd, data = res.reg.scatter))

### additional notes on graphs

# -	Densities of earnings, edu and wealth for appendix
# -	Move conditional expectations over wealth and education to appendix
# -	Prediction comparisons to appendix
