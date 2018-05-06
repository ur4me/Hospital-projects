
#preparation
library(dplyr)
library(ranger)
library(xgboost)
library(caret)
library(SDMTools)
library(e1071)
library(cvTools)
library(randomForest)


setwd('d:/stroke/data')

#retrieve train and test
total <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)


#Exploratory data analysis
str(total)
colnames(total)[colSums(is.na(total)) > 0]
summary(total$EDHour) # NA's :5
summary(total$MinstoRad) # NA's : 171
summary(total$NIHSS) #NA's : 373


#remove some columns
total <- total %>% select(-IP_Encounter,-Dependence_Level,-InptDeath,-AcuteLOS,-RehabLOS,-TotalLOS, -AssessedByStrokeTeam)
total <- total[,order(colnames(total))]

total$Rehab <- as.factor(total$Rehab)


#NA handling
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Rehab, -FiscalYear)
colnames(without_na)[colSums(is.na(without_na)) > 0]


#impute NA for NIHSS
set.seed(1234)
model <-randomForest(NIHSS~. , without_na)
total[which(is.na(total$NIHSS)), "NIHSS"] <- predict(model, total)[which(is.na(total$NIHSS))]


#impute NA for MinstoRad
set.seed(1234)
model <-randomForest(MinstoRad~. , without_na)
total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]


#impute NA for EDHour
set.seed(1234)
model <-randomForest(EDHour~. , without_na)
total[which(is.na(total$EDHour)), "EDHour"] <- predict(model, total)[which(is.na(total$EDHour))]



#again
#impute NA for NIHSS
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Rehab, -FiscalYear, -MinstoRad, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(NIHSS~. , without_na)
total[which(is.na(total$NIHSS)), "NIHSS"] <- predict(model, total)[which(is.na(total$NIHSS))]


#impute NA for MinstoRad
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Rehab, -FiscalYear, -NIHSS, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(MinstoRad~. , without_na)
total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]


#impute NA for EDHour
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Rehab, -FiscalYear, -NIHSS, -MinstoRad)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(EDHour~. , without_na)
total[which(is.na(total$EDHour)), "EDHour"] <- predict(model, total)[which(is.na(total$EDHour))]


total$NIHSS <- round(total$NIHSS)
total$MinstoRad <- round(total$MinstoRad)
total$EDHour <- round(total$EDHour)


#separate to train and test
train <- total %>% filter(FiscalYear != 2017)
test <- total %>% filter(FiscalYear == 2017)


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
  model <- ranger(formula = Rehab~.-FiscalYear,
                  data=training,
                  num.trees = 25,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  
  
  acc[cnt] <- confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]]
  cnt <- cnt + 1}

mean(acc) #0.7309125


#evaluation with test set without outlier handling
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = Rehab~.-FiscalYear,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
confusionMatrix(prediction, test$Rehab)[["overall"]][["Accuracy"]] #I got 0.6947608







#outlier handling
#library("cluster")
#set.seed(123)
#km.res <- kmeans(train %>% select(-Rehab), 2, nstart = 25)
#
#train$cluster <- as.vector(km.res$cluster)
#train$cluster <- as.factor(train$cluster)
#
#
#train <- train %>% filter(cluster == 1)
#train <- train %>% select(-cluster)
#
##evaluation
##4 fold cross validation
#set.seed(1234)
#cross <- cvFolds(nrow(train), K=4)
#cnt = 1
#acc <- numeric()
#
#for(i in 1:4){
#  datas_idx <- cross$subsets[cross$which==i,1]
#  
#  testing <- train[datas_idx,]
#  training <- train[-datas_idx,]
#  
#  prod.params <- list(mtry = 5, nodesize = 13) 
#  set.seed(1234)
#  model <- ranger(formula = Rehab~.-FiscalYear,
#                  data=training,
#                  num.trees = 25,
#                  mtry=prod.params$mtry,
#                  min.node.size = prod.params$nodesize)
#  
#  prediction <- predictions(predict(model, testing))
#  
#  
#  acc[cnt] <- confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]]
#  cnt <- cnt + 1}
#
#mean(acc) #0.7417833
#
#
#
#
##evaluation with test set
#
#prod.params <- list(mtry = 5, nodesize = 13) 
#set.seed(1234)
#rf_model <- ranger(formula = Rehab~.-FiscalYear,
#                   data=train,
#                   num.trees = 25,
#                   mtry=prod.params$mtry,importance = 'impurity',
#                   min.node.size = prod.params$nodesize)
#
#
#prediction <- predictions(predict(rf_model, test))
#confusionMatrix(prediction, test$Rehab)[["overall"]][["Accuracy"]] #I got 0.6856492




#finding optimized tuned weight

#et.seed(1234)
#ross <- cvFolds(nrow(train), K=4)

#omparison <- data.frame(int =1:30)
#=2

#or(i in 1:4){
# datas_idx <- cross$subsets[cross$which==i,1]
# 
# testing <- train[datas_idx,]
# training <- train[-datas_idx,]
# 
# 
# 
# 
#  #finding optimized tuned weight
#  weight.comparison <- data.frame(int = numeric(), accuracy = numeric())
#  for(n in 1:30){
#    
#    #evaluation with using weight
#    
#    tuned.weight <- (training$FiscalYear+1.5^(n-2)-min(training$FiscalYear)+2)/max(training$FiscalYear+1.5^(n-2)-min(training$FiscalYear)+2)
#    
#    set.seed(1234)
#    prod.params <- list(mtry = 20, nodesize = 13) 
#    
#    rf_model1 <- ranger(formula = Rehab~.-FiscalYear ,
#                        data=training,
#                        num.trees = 25,
#                        mtry=prod.params$mtry, 
#                        case.weights = tuned.weight,
#                        min.node.size = prod.params$nodesize)
#    
#    #evaluation
#    prediction <- predictions(predict(rf_model1, testing))
#    print(confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]])
#    result <- data.frame(int = n, accuracy = confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]])
#    weight.comparison <- rbind(weight.comparison,result)}
#  
#  comparison[c] <- weight.comparison$accuracy
#  c <- c + 1
#  
#}
#
#comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
#comparison <- comparison %>% arrange(desc(average))
#max(comparison$average) #0.7454184
#
##choosing the n
#tuned.weight <- (train$FiscalYear+1.5^(comparison[1,1]-2)-min(train$FiscalYear)+2)/max(train$FiscalYear+1.5^(comparison[1,1]-2)-min(train$FiscalYear)+2)

#remove FiscalYear
train$FiscalYear <- NULL

##accuracy for test
#set.seed(1234)
#prod.params <- list(mtry = 20, nodesize = 13) 
#
#rf_model1 <- ranger(formula = Rehab~. ,
#                    data=train,
#                    num.trees = 25,
#                    mtry=prod.params$mtry, 
#                    case.weights = tuned.weight,
#                    min.node.size = prod.params$nodesize)
#
##evaluation
#prediction <- predictions(predict(rf_model1, test))
#print(confusionMatrix(prediction, test$Rehab)[["overall"]][["Accuracy"]]) #0.6970387




#feature selection from ranger : importance = 'impurity'
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = Rehab~.,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)


v<-as.vector(rf_model$variable.importance)
train1 <- train %>% select(-Rehab)
w <- colnames(train1)
DF<-cbind(w,v)
DF<-as.data.frame(DF)
DF$v <- as.character(DF$v)
DF$v <- as.numeric(DF$v)

feature.importance <- DF %>% arrange(v)

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
  
  
  

  col.remove <- data.frame(int = numeric(), accuracy = numeric())
  for(n in 1:20){
    
    
    fi <- feature.importance[-c(1:n),]
    
    fi <- as.vector(fi[,1])
    
    #feature selection
    training <- training[,c('Rehab',fi)]
    testing <- testing[,c('Rehab',fi)]
    
    
    set.seed(1234)
    prod.params <- list(mtry = 10, nodesize = 13) 
    
    rf_model1 <- ranger(formula = Rehab~. ,
                        data=training,
                        num.trees = 25,
                        mtry=prod.params$mtry, 
                        min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model1, testing))
    print(confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]])
    result <- data.frame(int = n, accuracy = confusionMatrix(prediction, testing$Rehab)[["overall"]][["Accuracy"]])
    col.remove <- rbind(col.remove,result)
    }
  
  comparison[c] <- col.remove$accuracy
  c <- c + 1
  
}



comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(desc(average))
a <- max(comparison$average) #0.7432882


important_features <- feature.importance[-c(1:comparison[1,1]),]
important_features <- as.vector(important_features[,1])




#evaluate with test set
train1 <- train[,c('Rehab',important_features)]
train1 <- train1[,order(colnames(train1))]

prod.params <- list(mtry = 5, nodesize = 13) 


set.seed(1234)
rf_model <- ranger(formula = Rehab~.,
                   data=train1,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
confusionMatrix(prediction, test$Rehab)[["overall"]][["Accuracy"]] #0.6947608





#Reducing train_data and test_data by selecting important features
train <- train[,c('Rehab',important_features)]
test <- test[,c('Rehab',important_features)]
train <- train[,order(colnames(train))]
test <- test[,order(colnames(test))]










#Ranger parameter tuning
#train$tuned.weight <- tuned.weight
train$Rehab <- as.numeric(as.character(train$Rehab))
test$Rehab <- as.numeric(as.character(test$Rehab))

#finding optimized number of features
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
  
  a <- data.frame(int = numeric(), accuracy = numeric())
  for(n in 1:nrow(tune.grid)) {
    prod.params <- list(mtry = as.numeric(tune.grid[n,'mtry']), 
                        nodesize = as.numeric(tune.grid[n,'nodesize']))
    
    set.seed(1234)
    rf_model <- ranger(formula = Rehab~.-tuned.weight,
                       data=training,
                       num.trees = 50,
                       mtry=prod.params$mtry,
                       #case.weights = training$tuned.weight,
                       min.node.size = prod.params$nodesize)
    
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    prediction <- ifelse(prediction < 0.5 , 0 , 1)
    print(confusionMatrix(as.factor(prediction), as.factor(testing$Rehab))[["overall"]][["Accuracy"]])
    
    
    result <- data.frame(grid.number = n, accuracy = confusionMatrix(as.factor(prediction), as.factor(testing$Rehab))[["overall"]][["Accuracy"]])
    a <- rbind(a,result)
  }
  
  comparison[c] <- a$accuracy
  c <- c + 1
  
} 


comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(desc(average))
max(comparison$average) #0.7543745





prod.params <- list(mtry = as.numeric(tune.grid[comparison[1,1],'mtry']), 
                    nodesize = as.numeric(tune.grid[comparison[1,1],'nodesize']))

train$tuned.weight <- NULL






#4 fold cross validation without outlier handling
set.seed(1234)
cross <- cvFolds(nrow(train), K=4)
cnt = 1
acc <- numeric()

for(i in 1:4){
  datas_idx <- cross$subsets[cross$which==i,1]
  
  testing <- train[datas_idx,]
  training <- train[-datas_idx,]
  
  prod.params <- prod.params
  set.seed(1234)
  model <- ranger(formula = Rehab~.-FiscalYear,
                  data=training,
                  num.trees = 50,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  prediction <- ifelse(prediction < 0.5 , 0 ,1)
  
  acc[cnt] <- confusionMatrix(as.factor(prediction), as.factor(testing$Rehab))[["overall"]][["Accuracy"]]
  cnt <- cnt + 1}

mean(acc) #0.7151435


#evaluation with test set without outlier handling
prod.params <- prod.params
set.seed(1234)
rf_model <- ranger(formula = Rehab~.-FiscalYear,
                   data=train,
                   num.trees = 50,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
prediction <- ifelse(prediction < 0.5 , 0 ,1)
confusionMatrix(as.factor(prediction), as.factor(test$Rehab))[["overall"]][["Accuracy"]] #I got 0.6719818



#multivariate outlier checking

prediction <- predictions(predict(rf_model, train))


#adding diff variable (Actual - prediction)^2
train$diff <- (prediction <- predictions(predict(rf_model, train)) - train$Rehab)^2

#Finding optimized quantile
outlier.checking <- data.frame(percentage = as.numeric(), MAE = as.numeric())

for(n in 80:100){
  
  #removing outliers
  train1 <- train %>% filter(diff < quantile(train$diff,0.01*n))
  train1$diff <- NULL
  
  
  #outlier checking with Ranger
  set.seed(1234)
  prod.params <- prod.params 
  
  rf_model1 <- ranger(formula = Rehab~.,
                      data=train1,
                      num.trees = 50,
                      mtry=prod.params$mtry,
                      min.node.size = prod.params$nodesize)
  
  
  #evaluation
  prediction <- predictions(predict(rf_model1, test))
  print(mean(abs(test$Rehab - round(prediction)), na.rm=TRUE))
  result <- data.frame(percentage = 0.01*n, MAE = mean(abs(test$Rehab - round(prediction)), na.rm=TRUE))
  outlier.checking <- rbind(outlier.checking,result)}

#removing outliers
outlier.checking <- outlier.checking %>% arrange(MAE)
train <- train %>% filter(diff < quantile(train$diff,as.numeric(outlier.checking[1,1])))
train$diff <- NULL



#evaluation with test set
prod.params <- prod.params
set.seed(1234)
rf_model <- ranger(formula = Rehab~.-FiscalYear,
                   data=train,
                   num.trees = 50,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
prediction <- ifelse(prediction < 0.5 , 0 ,1)
confusionMatrix(as.factor(prediction), as.factor(test$Rehab))[["overall"]][["Accuracy"]] #I got 0.667426

















#using XGBoost
train1 <- train
test1 <- test
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)


#xgb matrix
withoutRV <- train1 %>% select(-Rehab)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = train1$Rehab-1)
withoutRV1 <- test1 %>% select(-Rehab)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))


#xgboost parameters
xgb_params <- list(colsample_bytree = 0.7, #variables per tree 
                   subsample = 0.8, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 10, #tree levels
                   eta = 0.12, #shrinkage
                   objective = "binary:logistic",
                   gamma=0)    

#cross-validation and checking iterations
set.seed(4321)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000) 

gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F, 
                   nrounds = xgb_cv$best_iteration)

prediction_xgb <- predict(gb_dt,dtest)
prediction_xgb <- ifelse(prediction_xgb < 0.5 , 0 ,1)


#confustion matrix
confusionMatrix(as.factor(test1$Rehab-1), as.factor(prediction_xgb))[["overall"]][["Accuracy"]] #0.7015945

#using lm

train1 <- train
test1 <- test

train1$Rehab <- as.numeric(as.character(train1$Rehab))
test1$Rehab <- as.numeric(as.character(test1$Rehab))

lm_model <- lm(Rehab ~., train1)

prediction_lm <- predict (lm_model , test1)

prediction_lm <- ifelse(prediction_lm < 0.5 , 0 ,1)

confusionMatrix(as.factor(prediction_lm), as.factor(test1$Rehab))[["overall"]][["Accuracy"]] #0.6264237




