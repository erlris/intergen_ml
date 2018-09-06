rm(list=ls())

setwd("~/mlpaper/")

library(tidyverse)
library(haven)
library(recipes)
library(forcats)

#Find individuals to analyze and their parents

sampledata  <- read_dta("/data/prosjekt/generation_mobility/population/fasteoppl.dta")

#Drop individuals not born in Norway
sampledata <- filter(sampledata,foede_kommnr_land >= 100 & foede_kommnr_land < 3000 & 
                         !is.na(foede_kommnr_land))

sampledata <- select(sampledata,npid,nmpid,nfpid,foedselsaar,
                     kjoenn,foede_kommnr_land)

sampledata  <- rename(sampledata,
                      yob_child = foedselsaar,
                      gender_child = kjoenn,
                      mob_child=foede_kommnr_land)

sampledata <- filter(sampledata,yob_child >= 1970 & yob_child <= 1975)

#Get parents' birth years
birthyears <- read_dta("/data/prosjekt/generation_mobility/population/fasteoppl.dta")

birthyears <- select(birthyears,npid,foedselsaar)

#Fathers
sampledata <- left_join(x=sampledata, y=birthyears,by=c("nfpid"="npid"))

sampledata <- rename(sampledata,yob_father=foedselsaar)

#Mothers
sampledata <- left_join(x=sampledata, y=birthyears,by=c("nmpid"="npid"))

sampledata <- rename(sampledata,yob_mother=foedselsaar)

####Earnings Data####
earningsdata <- read_dta("~/incomeproject/data/dispinnt_1967-2014.dta")

earningsdata <- select(earningsdata,npid,pearn,year)

earningsdata <- inner_join(x=earningsdata,y=birthyears,by=c("npid"="npid"))

#Inflation adjustment
kpidata <- read_dta("~/data/kpi_1920-2015.dta")

earningsdata <- inner_join(x=earningsdata,y=kpidata,by=c("year"="year"))

earningsdata <- mutate(earningsdata,pearn=(pearn*100)/kpi)

earningsdata <- select(earningsdata,-kpi)

rm(kpidata)

#Children
children <- left_join(x=sampledata,y=earningsdata,by=c("npid"="npid"))

children <- filter(children, year - foedselsaar >= 30 & year - foedselsaar <= 35)

children <- children %>% 
    group_by(npid,yob_child) %>%
    summarize(meanpearn_child = mean(pearn,na.rm=TRUE)) %>%
    ungroup()

children <- children %>%
    group_by(yob_child) %>%
    mutate(earncdf_child = cume_dist(meanpearn_child)) %>%
    ungroup() %>%
    select(-yob_child)

sampledata <- left_join(x=sampledata,y=children,by=c("npid"="npid"))

rm(children)

#Fathers
fathers <- sampledata %>%
    select(nfpid,yob_child) 

fathers <- inner_join(x=fathers,y=earningsdata,by=c("nfpid"="npid"))

fathers <- distinct(fathers)

fathers <- filter(fathers,year - foedselsaar >= 40 & year - foedselsaar <= 50)

fathers <- fathers %>% 
    group_by(nfpid,yob_child) %>%
    summarize(meanpearn_father = mean(pearn,na.rm=TRUE),
              sumpearn_father = sum(pearn,na.rm=TRUE)) %>%
    ungroup()

fathers <- fathers %>%
    group_by(yob_child) %>%
    mutate(earncdf_father = cume_dist(meanpearn_father)) %>%
    ungroup()

sampledata <- left_join(x=sampledata,y=fathers,by=c("nfpid"="nfpid","yob_child"="yob_child"))

rm(fathers)

#Mothers
mothers <- sampledata %>%
    select(nmpid,yob_child) 

mothers <- inner_join(x=mothers,y=earningsdata,by=c("nmpid"="npid"))

mothers <- distinct(mothers)

mothers <- filter(mothers,year - foedselsaar >= 40 & year - foedselsaar <= 50)

mothers <- mothers %>% 
    group_by(nmpid,yob_child) %>%
    summarize(meanpearn_mother = mean(pearn,na.rm=TRUE),
              sumpearn_mother = sum(pearn,na.rm=TRUE)) %>%
    ungroup()

mothers <- mothers %>%
    group_by(yob_child) %>%
    mutate(earncdf_mother = cume_dist(meanpearn_mother)) %>%
    ungroup()

sampledata <- left_join(x=sampledata,y=mothers,by=c("nmpid"="nmpid","yob_child"="yob_child"))

rm(mothers)

rm(earningsdata)

#Calculating joint earnings ranks
sampledata <- sampledata %>%
    mutate(sumpearn_joint=sumpearn_father+sumpearn_mother)

sampledata <- sampledata %>%
    group_by(yob_child) %>%
    mutate(earncdf_joint = cume_dist(sumpearn_joint)) %>%
    ungroup()

####Wealth Data####
wealthdata <- read_dta("/data/prosjekt/generation_mobility/income/skatt_1967-2014.dta")

wealthdata <- wealthdata %>%
    select(npid,aargang,ntoform)

wealthdata <- inner_join(x=wealthdata,y=birthyears,by=c("npid"="npid"))

#Inflation Adjustment
kpidata <- read_dta("~/data/kpi_1920-2015.dta")

wealthdata <- inner_join(x=wealthdata,y=kpidata,by=c("aargang"="year"))

wealthdata <- mutate(wealthdata,ntoform=(ntoform*100)/kpi)

wealthdata <- select(wealthdata,-kpi)

rm(kpidata)

#Fathers
fathers <- sampledata %>%
    select(nfpid,yob_child) 

fathers <- inner_join(x=fathers,y=wealthdata,by=c("nfpid"="npid"))

fathers <- distinct(fathers)

fathers <- filter(fathers,aargang - foedselsaar >= 40 & aargang - foedselsaar <= 50)

fathers <- fathers %>% 
    group_by(nfpid,yob_child) %>%
    summarize(meanwealth_father = mean(ntoform,na.rm=TRUE)) %>%
    ungroup()

fathers <- fathers %>%
    group_by(yob_child) %>%
    mutate(wealthcdf_father = cume_dist(meanwealth_father)) %>%
    ungroup()

sampledata <- left_join(x=sampledata,y=fathers,by=c("nfpid"="nfpid","yob_child"="yob_child"))

rm(fathers)

#Mothers
mothers <- sampledata %>%
    select(nmpid,yob_child) 

mothers <- inner_join(x=mothers,y=wealthdata,by=c("nmpid"="npid"))

mothers <- distinct(mothers)

mothers <- filter(mothers,aargang - foedselsaar >= 40 & aargang - foedselsaar <= 50)

mothers <- mothers %>% 
    group_by(nmpid,yob_child) %>%
    summarize(meanwealth_mother = mean(ntoform,na.rm=TRUE)) %>%
    ungroup()

mothers <- mothers %>%
    group_by(yob_child) %>%
    mutate(wealthcdf_mother = cume_dist(meanwealth_mother)) %>%
    ungroup()

sampledata <- left_join(x=sampledata,y=mothers,by=c("nmpid"="nmpid","yob_child"="yob_child"))

rm(mothers)

rm(wealthdata)

####Education Data####
educationdata <- read_dta("/data/prosjekt/generation_mobility/education/f_demo.dta")

educationdata <- select(educationdata,
                        npid,
                        nus=bu,
                        eduy=bu_kltrinn)

#Drop those with unspecified education
educationdata <- educationdata %>% 
    filter(nus != 999999 & !is.na(nus))

#Find highest level of education
educationdata <- educationdata %>% 
    group_by(npid) %>%
    mutate(maxeduy=max(eduy)) %>%
    filter(eduy==max(maxeduy)) %>%
    filter(row_number()==1) %>%
    select(npid,nus,eduy)

educationdata <- educationdata %>% 
    mutate(nus=str_sub(nus,1,3))

educationdata$nus <- factor(educationdata$nus)

#Merge for chilren
sampledata  <- left_join(x=sampledata,y=educationdata,by=c("npid"="npid"))

sampledata  <- rename(sampledata,
                      nus_child=nus,
                      eduy_child=eduy)

#Merge for fathers
sampledata  <- left_join(x=sampledata,y=educationdata,by=c("nfpid"="npid"))

sampledata  <- rename(sampledata,
                      nus_father=nus,
                      eduy_father=eduy)

#Merge for mothers
sampledata  <- left_join(x=sampledata,y=educationdata,by=c("nmpid"="npid"))

sampledata  <- rename(sampledata,
                      nus_mother=nus,
                      eduy_mother=eduy)

rm(educationdata)

####Military Data####
militarydata <- read_dta("/data/prosjekt/generation_mobility/personal/sesjon_1969-2007.dta")

militarydata <- militarydata %>%
    select(npid,
           ability)

#Merge children
sampledata <- left_join(x=sampledata,y=militarydata,
                        by=c("npid"="npid"))

sampledata <- sampledata %>%
    rename(ability_child=ability)

#Merge fathers
sampledata <- left_join(x=sampledata,y=militarydata,
                        by=c("nfpid"="npid"))

sampledata <- sampledata %>%
    rename(ability_father=ability)

rm(militarydata)

####Labor market regions####
labordata <- read_dta("/data/gis/norge/arbeidsmarked/kommune-arbeidsmarked.dta")

labordata <- labordata %>%
    select(-gamle) %>%
    rename(region_child=arbeidsmarked) %>%
    mutate(kommune = as.integer(kommune))

labordata <- labordata[!duplicated(labordata),]

sampledata$mob_child  <- as.integer(sampledata$mob_child)

sampledata <- left_join(x=sampledata,y=labordata,by=c("mob_child"="kommune"))

rm(labordata)

#Redefine to factors####
sampledata <- sampledata %>%
    mutate(region_child=factor(region_child),
           mob_child=factor(mob_child))


####Census data####
fob70 <- read_dta("/data/prosjekt/generation_mobility/census/census_1970.dta")

fob70 <- select(fob70,npid,mstat70,mlength70,urban70,studa70,
                srcinc70,wact70,hours70,isic70,occ70,emplstat70,hhtype70,
                nindh70)

fob70 <- fob70 %>% 
    distinct(npid, .keep_all = TRUE)

fob70 <- fob70 %>%
    mutate(mlength70=if_else(mstat70!=2,0,mlength70))

fob70 <- fob70 %>%
    mutate(occ70 = ifelse (occ70 == "",
                           NA,
                           occ70))

fob70 <- fob70 %>%
    mutate(mstat70=factor(mstat70),
           urban70=factor(urban70),
           studa70=factor(studa70),
           srcinc70=factor(srcinc70),
           wact70=factor(wact70),
           hours70=factor(hours70),
           isic70=factor(isic70),
           occ70=factor(occ70),
           emplstat70=factor(emplstat70),
           hhtype70=factor(hhtype70))

#Merging fathers
fob70 <- fob70 %>%
    rename_all(~sub("70","70_father",.x))

sampledata <- left_join(x=sampledata,y=fob70,by=c("nfpid"="npid"))

#Merging mothers
fob70 <- fob70 %>%
    rename_all(~sub("father","mother",.x))

sampledata <- left_join(x=sampledata,y=fob70,by=c("nmpid"="npid"))

rm(fob70)

####Saving####
colnames(sampledata)
print(object.size(sampledata), units="auto")
save(sampledata,file="data/sampledata_completeness.RData")