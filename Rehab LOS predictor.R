#preparation
library(dplyr)
library(ranger)
library(xgboost)
library(caret)
library(SDMTools)
library(e1071)
library(cvTools)
library(randomForest)
library(lubridate)
library(pROC)
library(rpart)

#retrieve data
total <- read.csv('D:/development/stroke/data/StrokeData_MayUpdate.csv', na.strings = c("", "NA"), stringsAsFactors = F)

#Exploratory data analysis
str(total)
colnames(total)[colSums(is.na(total)) > 0]
summary(total$EDHour) # NA's :5
summary(total$MinstoRad) # NA's : 171
summary(total$NIHSS) #NA's : 373


#remove some columns
total <- total %>% select(-NHI,-IP_Encounter,-Dependence_Level,-AcuteLOS,-InptDeath,-TotalLOS, -AssessedByStrokeTeam)
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
without_na <- without_na %>% select(-RehabLOS, -weight)
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

##impute NA for MinstoRad
#set.seed(1234)
#model <-randomForest(MinstoRad~. , without_na)
#total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]
#
##impute NA for EDHour
#set.seed(1234)
#model <-randomForest(EDHour~. , without_na)
#total[which(is.na(total$EDHour)), "EDHour"] <- predict(model, total)[which(is.na(total$EDHour))]
#
##impute NA for NIHSS
#without_na <- total[complete.cases(total), ]
#without_na <- without_na %>% select(-RehabLOS, -weight, -MinstoRad, -EDHour)
#colnames(without_na)[colSums(is.na(without_na)) > 0]
#set.seed(1234)
#model <-randomForest(NIHSS~. , without_na)
#total[which(is.na(total$NIHSS)), "NIHSS"] <- predict(model, total)[which(is.na(total$NIHSS))]
#
##impute NA for MinstoRad
#without_na <- total[complete.cases(total), ]
#without_na <- without_na %>% select(-RehabLOS, -weight, -NIHSS, -EDHour)
#colnames(without_na)[colSums(is.na(without_na)) > 0]
#set.seed(1234)
#model <-randomForest(MinstoRad~. , without_na)
#total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]
#
##impute NA for EDHour
#without_na <- total[complete.cases(total), ]
#without_na <- without_na %>% select(-RehabLOS, -weight, -NIHSS, -MinstoRad)
#colnames(without_na)[colSums(is.na(without_na)) > 0]
#set.seed(1234)
#model <-randomForest(EDHour~. , without_na)
#total[which(is.na(total$EDHour)), "EDHour"] <- predict(model, total)[which(is.na(total$EDHour))]
#
#total$NIHSS <- round(total$NIHSS)
#total$MinstoRad <- round(total$MinstoRad)
#total$EDHour <- round(total$EDHour)

#split the data
set.seed(54321)
outcome <- total$RehabLOS

partition <- createDataPartition(y=outcome,
                                 p=.75,
                                 list=F)
train <- total[partition,]
test <- total[-partition,]

#4 fold cross validation without outlier handling
set.seed(1234)
cross <- cvFolds(nrow(train), K=4)
cnt = 1
acc <- numeric()

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]
  
  prod.params <- list(mtry = 5, nodesize = 13) 
  set.seed(1234)
  model <- ranger(formula = RehabLOS~.-weight,
                  data=training,
                  num.trees = 25,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  
  a <- cbind(Rehab = testing$Rehab, RehabLOS = testing$RehabLOS ,prediction = round(prediction))
  a <- as.data.frame(a)
  a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
  print(mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
  
  
  acc[cnt] <- mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE)
  cnt <- cnt + 1}

mean(acc) #4.072764

#evaluation with test set without outlier handling
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = RehabLOS~.-weight,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
mean(abs(test$RehabLOS - round(prediction)), na.rm=TRUE) #5.239609

a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE) #4.370416

#multivariate outlier checking

prediction <- predictions(predict(rf_model, train))

#adding diff variable (Actual - prediction)^2
train$diff <- (prediction <- predictions(predict(rf_model, train)) - train$RehabLOS)^2

#Finding optimized quantile
outlier.checking <- data.frame(percentage = as.numeric(), MAE = as.numeric())

set.seed(1234)
cross <- cvFolds(nrow(train), K=4)
comparison <- data.frame(percentage =  80:100)
c=2

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]
  
  #Finding optimized quantile
  outlier.checking <- data.frame(percentage = as.numeric(), MAE = as.numeric())
  for(n in 80:100) {
    
    training1 <- training %>% filter(diff <= quantile(training$diff,0.01*n))
    training1$diff <- NULL
    set.seed(1234)
    rf_model <- ranger(formula = RehabLOS~.-weight,
                       data=training1,
                       num.trees = 25,
                       mtry=prod.params$mtry,
                       min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    a <- cbind(Rehab = testing$Rehab, RehabLOS = testing$RehabLOS ,prediction = round(prediction))
    a <- as.data.frame(a)
    a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
    mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE)
    print(mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    
    result <- data.frame(percentage = n, MAE = mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    outlier.checking <- rbind(outlier.checking,result)
  }
  
  comparison[c] <- outlier.checking$MAE
  c <- c + 1
  
} 

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #4.045528

#removing outliers
train <- train %>% filter(diff <= quantile(train$diff,comparison[1,1]/100))
train$diff <- NULL

#evaluation with test set
prod.params <- prod.params
set.seed(1234)
rf_model <- ranger(formula = RehabLOS~.-weight,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))

mean(abs(test$RehabLOS - round(prediction)), na.rm=TRUE) #5.308068

a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE) #4.432763

##outlier handling
#library("cluster")
#set.seed(123)
#km.res <- kmeans(train %>% select(-RehabLOS), 2, nstart = 25)
#
#train$cluster <- as.vector(km.res$cluster)
#train$cluster <- as.factor(train$cluster)
#
#
#train1 <- train %>% filter(cluster == 1)
#train1 <- train %>% select(-cluster)
#
##evaluation
##4 fold cross validation
#set.seed(1234)
#cross <- cvFolds(nrow(train1), K=4)
#cnt = 1
#acc <- numeric()
#
#for(i in 1:4){
#  datas_idx <- cross$subsets[cross$which==i,1]
#  
#  testing <- train1[datas_idx,]
#  training <- train1[-datas_idx,]
#  
#  prod.params <- list(mtry = 5, nodesize = 13) 
#  set.seed(1234)
#  model <- ranger(formula = RehabLOS~.-weight,
#                  data=training,
#                  num.trees = 25,
#                  mtry=prod.params$mtry,
#                  min.node.size = prod.params$nodesize)
#  
#  prediction <- predictions(predict(model, testing))
#  
#  
#  acc[cnt] <- mean(abs(testing$RehabLOS - round(prediction)), na.rm=TRUE)
#  cnt <- cnt + 1}
#
#mean(acc) #4.467969
#
#
#
#
##evaluation with test set
#
#prod.params <- list(mtry = 5, nodesize = 13) 
#set.seed(1234)
#rf_model <- ranger(formula = RehabLOS~.-weight,
#                   data=train,
#                   num.trees = 25,
#                   mtry=prod.params$mtry,importance = 'impurity',
#                   min.node.size = prod.params$nodesize)
#
#
#prediction <- predictions(predict(rf_model, test))
#mean(abs(test$RehabLOS - round(prediction)), na.rm=TRUE) #3.98861




#finding optimized tuned weight

set.seed(1234)
cross <- cvFolds(nrow(train), K=4)

comparison <- data.frame(int =1:30)
c=2

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]

  #finding optimized tuned weight
  weight.comparison <- data.frame(int = numeric(), MAE = numeric())
  for(n in 1:30){
    
    #evaluation with using weight
    
    tuned.weight <- (training$weight + (1.5^n)/100)/max(training$weight + (1.5^n)/100)
    
    set.seed(1234)
    prod.params <- list(mtry = 20, nodesize = 13) 
    
    rf_model1 <- ranger(formula = RehabLOS~.-weight ,
                        data=training,
                        num.trees = 25,
                        mtry=prod.params$mtry, 
                        case.weights = tuned.weight,
                        min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model1, testing))

    a <- cbind(Rehab = testing$Rehab, RehabLOS = testing$RehabLOS ,prediction = round(prediction))
    a <- as.data.frame(a)
    a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
    print(mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    
    result <- data.frame(percentage = n, MAE = mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    
    weight.comparison <- rbind(weight.comparison,result)}
  
  comparison[c] <- weight.comparison$MAE
  c <- c + 1
  
}

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #3.264988

#choosing the n
tuned.weight <- (train$weight + (1.5^comparison[1,1])/100)/max(train$weight + (1.5^comparison[1,1])/100)
train$weight <- tuned.weight

#MAE for test
set.seed(1234)
prod.params <- list(mtry = 20, nodesize = 13) 

rf_model1 <- ranger(formula = RehabLOS~. -weight,
                    data=train,
                    num.trees = 25,
                    mtry=prod.params$mtry, 
                    case.weights = train$weight,
                    min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model1, test))

mean(abs(test$RehabLOS - round(prediction)), na.rm=TRUE) #4.260391

a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE) #4.255501






#feature selection from ranger : importance = 'impurity'
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = RehabLOS~.-weight,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

v<-as.vector(rf_model$variable.importance)
train1 <- train %>% select(-RehabLOS,-weight)
w <- colnames(train1)
DF<-cbind(w,v)
DF<-as.data.frame(DF)
DF$v <- as.character(DF$v)
DF$v <- as.numeric(DF$v)

feature.importance <- DF %>% arrange(v)

#plot feature importance
library(ggplot2)
imp_matrix <- DF %>% arrange(desc(v))
imp_matrix %>%
  ggplot(aes(reorder(w, v, FUN = max), v, fill = w)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(x = "Features", y = "Importance")

#finding optimized number of features

set.seed(1234)
cross <- cvFolds(nrow(train), K=4)
cnt = 1
acc <- numeric()
comparison <- data.frame(int =1:20)
c=2

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]
  
  col.remove <- data.frame(int = numeric(), MAE = numeric())
  for(n in 1:20){
    
    fi <- feature.importance[-c(1:n),]
    
    fi <- as.vector(fi[,1])
    
    #feature selection
    training <- training[,c('RehabLOS',fi)]
    testing <- testing[,c('RehabLOS',fi)]
    
    set.seed(1234)
    prod.params <- list(mtry = 10, nodesize = 13) 
    
    rf_model1 <- ranger(formula = RehabLOS~.-weight ,
                        data=training,
                        num.trees = 25,
                        mtry=prod.params$mtry, 
                        min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model1, testing))
    
    a <- cbind(Rehab = testing$Rehab, RehabLOS = testing$RehabLOS ,prediction = round(prediction))
    a <- as.data.frame(a)
    a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
    print(mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    
    result <- data.frame(int = n, MAE = mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    col.remove <- rbind(col.remove,result)
  }
  
  comparison[c] <- col.remove$MAE
  c <- c + 1
  
}

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #3.283023

important_features <- feature.importance[-c(1:comparison[1,1]),]
important_features <- as.vector(important_features[,1])

#evaluate with test set
train1 <- train[,c('RehabLOS','weight',important_features)]
train1 <- train1[,order(colnames(train1))]

prod.params <- list(mtry = 5, nodesize = 13) 

set.seed(1234)
rf_model <- ranger(formula = RehabLOS~.-weight,
                   data=train1,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE) #4.403423


#Reducing train_data and test_data by selecting important features
train <- train[,c('RehabLOS','weight',important_features)]
test <- test[,c('RehabLOS','weight',important_features)]
train <- train[,order(colnames(train))]
test <- test[,order(colnames(test))]

#Ranger parameter tuning

tune.grid <- expand.grid(mtry = 1:(ncol(train)-2),
                         nodesize = 1:(ncol(train)-2))
set.seed(1234)
cross <- cvFolds(nrow(train), K=4)
comparison <- data.frame(int =  1:nrow(tune.grid))
c=2

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]
  
  b <- data.frame(int = numeric(), MAE = numeric())
  for(n in 1:nrow(tune.grid)) {
    prod.params <- list(mtry = as.numeric(tune.grid[n,'mtry']), 
                        nodesize = as.numeric(tune.grid[n,'nodesize']))
    
    set.seed(1234)
    rf_model <- ranger(formula = RehabLOS~.-weight,
                       data=training,
                       num.trees = 50,
                       case.weights = training$weight,
                       mtry=prod.params$mtry,
                       min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    
    a <- cbind(Rehab = testing$Rehab, RehabLOS = testing$RehabLOS ,prediction = round(prediction))
    a <- as.data.frame(a)
    a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
    print(mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    
    result <- data.frame(grid.number = n, MAE = mean(abs(testing$RehabLOS - round(a$prediction)), na.rm=TRUE))
    b <- rbind(b,result)
  }
  
  comparison[c] <- b$MAE
  c <- c + 1
  
} 

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #3.234983

prod.params <- list(mtry = as.numeric(tune.grid[comparison[1,1],'mtry']), 
                    nodesize = as.numeric(tune.grid[comparison[1,1],'nodesize']))

train$tuned.weight <- NULL
#15/7

#using test to evaluate

set.seed(1234)
RehabLOS_predictor <- ranger(formula = RehabLOS~.-weight,
                             data=train,
                             num.trees = 50,
                             mtry=prod.params$mtry,importance = 'impurity',
                             case.weights = train$weight,
                             min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(RehabLOS_predictor, test))

a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
print(mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE)) #4.209046

roc_obj <- multiclass.roc(test$RehabLOS, a$prediction)
auc(roc_obj) #0.8075

#save RehabLOS_predictor
save(RehabLOS_predictor, file = "D:/development/Stroke/Model/RehabLOS_predictor.rda")

#using XGBoost
train1 <- train %>% select(-weight)
test1 <- test%>% select(-weight)
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)

#xgb matrix
withoutRV <- train1 %>% select(-RehabLOS)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = train1$RehabLOS)
withoutRV1 <- test1 %>% select(-RehabLOS)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))

#xgboost parameters
xgb_params <- list(colsample_bytree = 0.7, #variables per tree 
                   subsample = 0.8, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 10, #tree levels
                   eta = 0.12, #shrinkage
                   eval_metric = "mae", 
                   objective = "reg:linear",
                   gamma=0)    

#cross-validation and checking iterations
set.seed(4321)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000) 

gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F, 
                   nrounds = xgb_cv$best_iteration)


#evaluation
prediction <- predict(gb_dt,dtest)
a <- cbind(Rehab = test$Rehab, RehabLOS = test$RehabLOS ,prediction = round(prediction))
a <- as.data.frame(a)
a$prediction <- ifelse(a$Rehab == 0 , 0 , round(prediction))
print(mean(abs(test$RehabLOS - round(a$prediction)), na.rm=TRUE)) #4.201711

roc_obj <- multiclass.roc(test1$RehabLOS, a$prediction)
auc(roc_obj) #0.7901


