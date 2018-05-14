#preparation
library(dplyr)
library(ranger)
library(xgboost)
library(caret)
library(SDMTools)
library(e1071)


#retrieve data
total <- read.csv('D:/development/stroke/data/StrokeData_MayUpdate.csv', na.strings = c("", "NA"), stringsAsFactors = F)

#Exploratory data analysis
str(total)
colnames(total)[colSums(is.na(total)) > 0]
summary(total$EDHour) # NA's :5
summary(total$MinstoRad) # NA's : 171
summary(total$NIHSS) #NA's : 373


#remove some columns
total <- total %>% select(-IP_Encounter,-Dependence_Level,-InptDeath,-Rehab,-RehabLOS,-TotalLOS, -AssessedByStrokeTeam)
total <- total[,order(colnames(total))]

#feature engineering
total$admitdate <- as.Date(total$admitdate, "%d/%m/%Y")

total$month <- month(total$admitdate)
total$daydiff <- as.numeric(Sys.Date()-total$admitdate)
total$admitdate <- NULL
total$Ethnicity <- as.factor(total$Ethnicity)
total$TriageCode <- as.factor(total$TriageCode)
total$FiscalYear <- NULL
total$weight <- 1/total$daydiff

#remove some outstanding outliers (univariate approach)
total <- total %>% filter(MinstoRad < 1e+05)
total <- total %>% filter(IPEventsLast6Mths < 20)

#NA handling
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-AcuteLOS, -weight)
colnames(without_na)[colSums(is.na(without_na)) > 0]


#impute NA for NIHSS

prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
model <- ranger(formula = NIHSS~. ,
                data=without_na,
                num.trees = 25,
                mtry=prod.params$mtry,
                min.node.size = prod.params$nodesize)



total[which(is.na(total$NIHSS)), "NIHSS"] <- predictions(predict(model, total))[which(is.na(total$NIHSS))]



#split the data
set.seed(54321)
outcome <- total$AcuteLOS

partition <- createDataPartition(y=outcome,
                                 p=.75,
                                 list=F)
train <- total[partition,]
test <- total[-partition,]




#prediction
load(file = "D:/development/stroke/model/acute_los_predictor.rda")






#* @get /user/<id>
function(id){
  paste0("Predicted Acute Length of Stay :", 
        round(predictions(predict(acute_los_predictor, total%>%filter(NHI==id)%>%distinct()))))  
  
}




