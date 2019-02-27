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

res <- as_tibble(readRDS("data/modelmetrics_traintest.rds"))
res.reg <- as_tibble(readRDS("data/modelmetrics_regions.rds"))

### 2. Completeness ------

### 2.1 Test

test <- res %>% filter(data == "Testing",
                       variables %in% c("Rank-Rank",
                                        "Income with multiple functional forms",
                                        "Income & wealth",
                                        "Income & education length",
                                        "Income, wealth & education length",
                                        "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region"))
  
#test$variables[which(test$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
test$variables[which(test$variables == "Income with multiple functional forms")] <- "Income (flexible)"
#test$ElasticNet[which(test$variables == "Income (flexible)")] <- NA
test$variables[which(test$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region")] <- "Extended"

test %>%  ggplot(aes(x=reorder(str_wrap(variables,15), modelnumber),
             y=rsquared,
             fill=factor(model,levels=c("OLS", "ElasticNet","Ranger","XGBoost","NeuralNet"),
                         labels = c("OLS", "Elastic Net","Random Forest", "Gradient Boosted Trees",
                                    "Neural Net")))) +
  stat_summary(fun.y = "mean", geom="col", position=position_dodge(),
               color="black") +
  scale_fill_brewer(palette = "Blues",name="Estimator") +
  labs(x="Included Variables",y="R-Squared") +
  theme(legend.position = "bottom") +
  scale_y_continuous(breaks=seq(0,0.06,0.01)) +
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
                                        "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region"))

#test$variables[which(test$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source & number of indivuduals in household")] <- "Extended"
train$variables[which(train$variables == "Income with multiple functional forms")] <- "Income (flexible)"
#test$ElasticNet[which(test$variables == "Income (flexible)")] <- NA
train$variables[which(train$variables == "Income, wealth, education length and type, occupation, marital status, urban/rural, student activity, main income source, number of indivuduals in household & birth region")] <- "Extended"

train %>%
  ggplot(aes(x = reorder( str_wrap(variables, 15), modelnumber),
             y = rsquared,
             fill=factor(model,levels=c("OLS", "ElasticNet","Ranger","XGBoost","NeuralNet"),
                         labels = c("OLS", "Elastic Net","Random Forest", "Gradient Boosted Trees",
                                    "Neural Net")))) +
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
  scale_y_continuous(breaks=seq(0,0.06,0.01)) +
  guides(fill = guide_legend(title.position = "top", title.hjust = 0.5))

ggsave(file = "graphs/train_R2.pdf",
       height = 08, width = 11)

### 2.3 Tables

train$variables <- gsub(x = train$variables, pattern = "&", replacement = "and")
train.tbl <- train %>%
  group_by(variables, model) %>%
  summarize("Training R2 (mean)" = round(mean(rsquared), digits = 3),
            "Training R2 (sd)" = round(sd(rsquared), digits = 4)) %>%
  filter(is.na(`Training R2 (mean)`) == F)
stargazer(train.tbl, summary = F,
          out="tables/train_results.tex",
          float = F)

test$variables <- gsub(x = test$variables, pattern = "&", replacement = "and")
test.tbl <- test %>%
  group_by(variables, model) %>%
  summarize("Test R2" = round(mean(rsquared), digits = 3)) %>%
  filter(is.na(`Test R2`) == F) %>%
  mutate(completeness = round(`Test R2`/0.05474, digits = 2))
stargazer(test.tbl, summary = F,
          out="tables/test_results.tex",
          float=F)

### 3. Regional results

# generate completeness index

res.reg.clean <- res.reg %>% 
  mutate(completeness=rankr2/nnr2)


# Table of results by region

res.reg.table <- res.reg.clean %>%  
  filter(n > 1000) %>%
  mutate(rankr2 = round(rankr2, digits = 3),
         nnr2 = round(nnr2, digits = 3),
         completeness = round(completeness, digits = 3)) %>% 
  select(obs = n,
         completeness,
         "R2 (rank)"= rankr2, 
         "R2 (neural net)"= nnr2) %>% 
  arrange(completeness)

stargazer(res.reg.table,
          summary = F,
          column.sep.width = "2pt",
          out="tables/region_results.tex",
          float=F)

### 4. Maps ------

coordinates <- read_dta("data/coordinates_labormarkets.dta")

colnames(coordinates) <- c("region","id","x","y")

coordinates$region <- factor(coordinates$region)

### 3.1 Completeness by region

res.reg.clean %>% rename("region" = "region_child") %>%
    mutate(completeness = ifelse(n >= 1000, completeness, NA)) %>% 
    inner_join(y=coordinates, by=c("region")) %>%
    ggplot(aes(x=x,y=y, group=region, fill=completeness)) +
    geom_polygon(color="black", size=0.1) +
    scale_fill_distiller(name="Completeness",
                         palette = "Blues",
                         direction = 1,
                         limits = c(0,1)) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/completeness_map.pdf",
       height=7,width=7)


### 3.2 Rank-rank R2 by region

res.reg.clean %>% rename("region" = "region_child") %>%
  mutate(rankr2=ifelse(n>=1000,rankr2,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=rankr2)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank\nR-squared",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rsquared_rank_map.pdf",
       height=7,width=7)

### 3.3 Full R2 by region

res.reg %>% rename("region" = "region_child") %>%
    mutate(nnr2=ifelse(n>=1000,nnr2,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=nnr2)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Full Model\nR-squared",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rsquared_full_map.pdf",
       height=7,width=7)

### 3.4 Rank-Rank Slope by Region

res.reg %>% rename("region" = "region_child") %>%
  mutate(rankcoef = ifelse(n >= 1000, rankcoef,NA)) %>% 
    inner_join(y=coordinates,by=c("region")) %>%
    ggplot(aes(x=x,y=y,group=region,fill=rankcoef)) +
    geom_polygon(color="black",size=0.1) +
    scale_fill_distiller(name="Rank-Rank\nSlope",
                         palette = "Blues",
                         direction=1) +
    theme_void(base_size = 18) +
    theme(legend.position = c(0.7,0.5))

ggsave(file="graphs/rankslope_map.pdf",
       height=7,width=7)


### OLD: not in latest version

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
