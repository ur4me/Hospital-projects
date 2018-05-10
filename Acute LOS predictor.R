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

#retrieve data
total <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)

#Exploratory data analysis
str(total)
colnames(total)[colSums(is.na(total)) > 0]
summary(total$EDHour) # NA's :5
summary(total$MinstoRad) # NA's : 171
summary(total$NIHSS) #NA's : 373

#remove some columns
total <- total %>% select(-IP_Encounter,-Dependence_Level,-InptDeath,-Rehab, -RehabLOS,-TotalLOS, -AssessedByStrokeTeam)
total <- total[,order(colnames(total))]

colnames(total)[colSums(is.na(total)) > 0]

#remove some outstanding outliers (univariate approach)
total <- total %>% filter(MinstoRad < 1e+05)
total <- total %>% filter(IPEventsLast6Mths < 20)

#NA handling
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-AcuteLOS, -FiscalYear)
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

#Imputing leftovers
#impute NA for NIHSS
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-AcuteLOS, -FiscalYear, -MinstoRad, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(NIHSS~. , without_na)
total[which(is.na(total$NIHSS)), "NIHSS"] <- predict(model, total)[which(is.na(total$NIHSS))]

#impute NA for MinstoRad
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-AcuteLOS, -FiscalYear, -NIHSS, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(MinstoRad~. , without_na)
total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]

#impute NA for EDHour
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-AcuteLOS, -FiscalYear, -NIHSS, -MinstoRad)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(EDHour~. , without_na)
total[which(is.na(total$EDHour)), "EDHour"] <- predict(model, total)[which(is.na(total$EDHour))]

#Change to Integer
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
  model <- ranger(formula = AcuteLOS~.-FiscalYear,
                  data=training,
                  num.trees = 25,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  
    acc[cnt] <- mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE)
  cnt <- cnt + 1}

a <- mean(acc) #4.467969

train$FiscalYear <- NULL

#evaluation with test set without outlier handling
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = AcuteLOS~.-FiscalYear,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
mean(abs(test$AcuteLOS - round(prediction)), na.rm=TRUE) #3.98861

#multivariate outlier checking

prediction <- predictions(predict(rf_model, train))

#adding diff variable (Actual - prediction)^2
train$diff <- (prediction <- predictions(predict(rf_model, train)) - train$AcuteLOS)^2

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
    
    training1 <- training %>% filter(diff < quantile(training$diff,0.01*n))
    training1$diff <- NULL
    set.seed(1234)
    rf_model <- ranger(formula = AcuteLOS~.-tuned.weight,
                       data=training1,
                       num.trees = 50,
                       mtry=prod.params$mtry,
                       #case.weights = training1$tuned.weight,
                       min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    print(mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    
    
    result <- data.frame(percentage = n, MAE = mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    outlier.checking <- rbind(outlier.checking,result)
  }
  
  comparison[c] <- outlier.checking$MAE
  c <- c + 1
  
} 

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #4.144309

#removing outliers
train <- train %>% filter(diff < quantile(train$diff,comparison[1,1]/100))
train$diff <- NULL

#evaluation with test set
prod.params <- prod.params
set.seed(1234)
rf_model <- ranger(formula = AcuteLOS~.-FiscalYear,
                   data=train,
                   num.trees = 50,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))

mean(abs(test$AcuteLOS - round(prediction)), na.rm=TRUE) #3.589977

##outlier handling
#library("cluster")
#set.seed(123)
#km.res <- kmeans(train %>% select(-AcuteLOS), 2, nstart = 25)
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
#  model <- ranger(formula = AcuteLOS~.-FiscalYear,
#                  data=training,
#                  num.trees = 25,
#                  mtry=prod.params$mtry,
#                  min.node.size = prod.params$nodesize)
#  
#  prediction <- predictions(predict(model, testing))
#  
#  
#  acc[cnt] <- mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE)
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
#rf_model <- ranger(formula = AcuteLOS~.-FiscalYear,
#                   data=train,
#                   num.trees = 25,
#                   mtry=prod.params$mtry,importance = 'impurity',
#                   min.node.size = prod.params$nodesize)
#
#
#prediction <- predictions(predict(rf_model, test))
#mean(abs(test$AcuteLOS - round(prediction)), na.rm=TRUE) #3.98861

##finding optimized tuned weight
#
#set.seed(1234)
#cross <- cvFolds(nrow(train), K=4)
#
#comparison <- data.frame(int =1:30)
#c=2
#
#for(i in 1:4){
#  datas_idx <- cross$subsets[cross$which==i,1]
#  
#  testing <- train[datas_idx,]
#  training <- train[-datas_idx,]
#  
#  
#  
#  
#  #finding optimized tuned weight
#  weight.comparison <- data.frame(int = numeric(), MAE = numeric())
#  for(n in 1:30){
#    
#    #evaluation with using weight
#    
#    tuned.weight <- (training$FiscalYear+1.5^(n-2)-min(training$FiscalYear)+2)/max(training$FiscalYear+1.5^(n-2)-min(training$FiscalYear)+2)
#    
#    set.seed(1234)
#    prod.params <- list(mtry = 20, nodesize = 13) 
#    
#    rf_model1 <- ranger(formula = AcuteLOS~.-FiscalYear ,
#                        data=training,
#                        num.trees = 25,
#                        mtry=prod.params$mtry, 
#                        case.weights = tuned.weight,
#                        min.node.size = prod.params$nodesize)
#    
#    #evaluation
#    prediction <- predictions(predict(rf_model1, testing))
#    print(confusionMatrix(prediction, testing$AcuteLOS)[["overall"]][["MAE"]])
#    result <- data.frame(int = n, MAE = confusionMatrix(prediction, testing$AcuteLOS)[["overall"]][["MAE"]])
#    weight.comparison <- rbind(weight.comparison,result)}
#  
#  comparison[c] <- weight.comparison$MAE
#  c <- c + 1
#  
#}
#
#comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
#comparison <- comparison %>% arrange(desc(average))
#max(comparison$average) #0.9334805
#
##choosing the n
#tuned.weight <- (train$FiscalYear+1.5^(comparison[1,1]-2)-min(train$FiscalYear)+2)/max(train$FiscalYear+1.5^(comparison[1,1]-2)-min(train$FiscalYear)+2)
#
##remove FiscalYear
train$FiscalYear <- NULL
#
##MAE for test
#set.seed(1234)
#prod.params <- list(mtry = 20, nodesize = 13) 
#
#rf_model1 <- ranger(formula = AcuteLOS~. ,
#                    data=train,
#                    num.trees = 25,
#                    mtry=prod.params$mtry, 
#                    case.weights = tuned.weight,
#                    min.node.size = prod.params$nodesize)
#
##evaluation
#prediction <- predictions(predict(rf_model1, test))
#print(confusionMatrix(prediction, test$AcuteLOS)[["overall"]][["MAE"]]) #0.9339408

#feature selection from ranger : importance = 'impurity'
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = AcuteLOS~.,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

v<-as.vector(rf_model$variable.importance)
train1 <- train %>% select(-AcuteLOS)
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
    training <- training[,c('AcuteLOS',fi)]
    testing <- testing[,c('AcuteLOS',fi)]
    
    set.seed(1234)
    prod.params <- list(mtry = 10, nodesize = 13) 
    
    rf_model1 <- ranger(formula = AcuteLOS~. ,
                        data=training,
                        num.trees = 25,
                        mtry=prod.params$mtry, 
                        min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model1, testing))
    print(mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    result <- data.frame(int = n, MAE = mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    col.remove <- rbind(col.remove,result)
  }
  
  comparison[c] <- col.remove$MAE
  c <- c + 1
  
}

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #2.44936


important_features <- feature.importance[-c(1:comparison[1,1]),]
important_features <- as.vector(important_features[,1])

#evaluate with test set
train1 <- train[,c('AcuteLOS',important_features)]
train1 <- train1[,order(colnames(train1))]

prod.params <- list(mtry = 5, nodesize = 13) 

set.seed(1234)
rf_model <- ranger(formula = AcuteLOS~.,
                   data=train1,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
mean(abs(test$AcuteLOS - round(prediction)), na.rm=TRUE) #3.562642

#R squared
res <- caret::postResample(test$AcuteLOS, round(prediction))
rsq <- res
rsq #0.2179243

#Reducing train_data and test_data by selecting important features
train <- train[,c('AcuteLOS',important_features)]
test <- test[,c('AcuteLOS',important_features)]
train <- train[,order(colnames(train))]
test <- test[,order(colnames(test))]

#Ranger parameter tuning
#train$tuned.weight <- tuned.weight

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
  
  a <- data.frame(int = numeric(), MAE = numeric())
  for(n in 1:nrow(tune.grid)) {
    prod.params <- list(mtry = as.numeric(tune.grid[n,'mtry']), 
                        nodesize = as.numeric(tune.grid[n,'nodesize']))
    
    set.seed(1234)
    rf_model <- ranger(formula = AcuteLOS~.-tuned.weight,
                       data=training,
                       num.trees = 50,
                       mtry=prod.params$mtry,
                       #case.weights = training$tuned.weight,
                       min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    print(mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    
    result <- data.frame(grid.number = n, MAE = mean(abs(testing$AcuteLOS - round(prediction)), na.rm=TRUE))
    a <- rbind(a,result)
  }
  
  comparison[c] <- a$MAE
  c <- c + 1
  
} 

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(average)
min(comparison$average) #2.605695

prod.params <- list(mtry = as.numeric(tune.grid[comparison[1,1],'mtry']), 
                    nodesize = as.numeric(tune.grid[comparison[1,1],'nodesize']))

train$tuned.weight <- NULL

#using test to evaluate

set.seed(1234)
acute_los_predictor <- ranger(formula = AcuteLOS~.,
                   data=train,
                   num.trees = 50,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

#evaluation
prediction_rf <- predictions(predict(acute_los_predictor, test))
mean(abs(test$AcuteLOS - round(prediction_rf)), na.rm=TRUE) #3.551253

#save acute_los_predictor
save(acute_los_predictor, file = "D:/Stroke/Model/acute_los_predictor.rda")


#using XGBoost
train1 <- train
test1 <- test
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)

#xgb matrix
withoutRV <- train1 %>% select(-AcuteLOS)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = train1$AcuteLOS)
withoutRV1 <- test1 %>% select(-AcuteLOS)
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

prediction_xgb <- predict(gb_dt,dtest)

#mae evaluation
mean(abs(test$AcuteLOS - round(prediction_xgb)), na.rm=TRUE) #3.517084

#R squared
res <- caret::postResample(test$AcuteLOS, round(prediction_xgb))
rsq <- res[2]
rsq #0.2267877
