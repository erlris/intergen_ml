### Graphs for paper
### Intergenerational mobility as a prediction problem
# Jack Blundell and Erling Risa

### 0. Setup ------

rm(list = ls()) 	
options("scipen"=100, "digits"=4)

set.seed(123)

setwd("/Users/jack/git_repos/intergen_ml") # Jack

# Load packages

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(stargazer)

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
                                        "Income, wealth & education length",
                                        "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")) %>%
  select(OLS, variables, ElasticNet, XGBoost, modelnumber)

test$variables[which(test$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
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
                                        "Income, wealth & education length",
                                        "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")) %>%
  select(OLS, variables, ElasticNet, XGBoost, modelnumber)

train$variables[which(train$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
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


### 3. Maps ------

### 3.1 Completeness by region

### 3.2 Rank-rank R2 by region

### 3.3 Full R2 by region

### 4. Scatter plots by region ------

### Prepare data

res.reg.scatter <- res.reg %>% mutate(completeness = rsquared_rank/ rsquared_full) %>%
  select(region, observations, earn_sd, edu_sd, wealth_sd, earn_edu_cor, earn_wealth_cor,
         rsquared_rank, rsquared_full, completeness)
  
### 4.1 Rank-rank R2 and complete R2 ------

plotdata <- res.reg.scatter %>%
  ggplot(aes(x = rsquared_rank, y = rsquared_full, size = observations)) +
  geom_point() + xlab("rank-rank R2") +
  ylab("Full model R2")
plotdata
ggsave(file = "graphs/regions_R2.pdf",
       height = 08, width = 11)

### 4.2 Completeness against distribution of other predictors ------

res.reg.scatter %>%
  filter(observations > 200) %>%
  select(completeness, earn_sd,edu_sd, wealth_sd,
         earn_edu_cor, earn_wealth_cor) %>%
  rename("Earnings sd" = earn_sd,
         "Education sd" = edu_sd,
         "Wealth sd" = wealth_sd,
         "Earnings and Education corr" = earn_edu_cor,
         "Earnings and Wealth corr" = earn_wealth_cor) %>%
  gather(key = variable, value = value, - completeness) %>%
  ggplot(aes(x = value, y = completeness)) +
  geom_point() + 
  geom_smooth(method = "lm") +
  labs(x="Value", y = "Completeness") +
  facet_wrap("variable", scales = "free_x")

ggsave(file = "graphs/completeness_dist.pdf",
       height = 09, width = 11)

### Table of results by region

res.reg.table <- res.reg %>% mutate(completeness = rsquared_rank/ rsquared_full) %>%
  select(region, observations, earn_sd, edu_sd, wealth_sd, earn_edu_cor, earn_wealth_cor,
         rsquared_rank, rsquared_full, completeness) %>%
  mutate(earn_sd = round(earn_sd, digits = 2),
         edu_sd = round(earn_sd, digits = 2),
         wealth_sd = round(earn_sd, digits = 2),
         earn_edu_cor = round(earn_sd, digits = 2),
         earn_wealth_cor = round(earn_sd, digits = 2),
         rsquared_rank = round(earn_sd, digits = 3),
         rsquared_full = round(earn_sd, digits = 3),
         completeness = round(completeness, digits = 3))

stargazer(res.reg.table, summary = F, column.sep.width = "2pt")

### additional notes on graphs

# -	Densities of earnings, edu and wealth for appendix
# -	Move conditional expectations over wealth and education to appendix
# -	Prediction comparisons to appendix
