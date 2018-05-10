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

total <- total %>% filter(InptDeath == 0)
#remove some columns
total <- total %>% select(-IP_Encounter,-InptDeath, -AssessedByStrokeTeam)
total <- total[,order(colnames(total))]

total$Dependence_Level <- as.factor(total$Dependence_Level)

#remove some outstanding outliers (univariate approach)
total <- total %>% filter(MinstoRad < 1e+05)
total <- total %>% filter(IPEventsLast6Mths < 20)

#NA handling
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Dependence_Level, -FiscalYear)
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

#imputing leftovers
#impute NA for NIHSS
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Dependence_Level, -FiscalYear, -MinstoRad, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(NIHSS~. , without_na)
total[which(is.na(total$NIHSS)), "NIHSS"] <- predict(model, total)[which(is.na(total$NIHSS))]

#impute NA for MinstoRad
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Dependence_Level, -FiscalYear, -NIHSS, -EDHour)
colnames(without_na)[colSums(is.na(without_na)) > 0]
set.seed(1234)
model <-randomForest(MinstoRad~. , without_na)
total[which(is.na(total$MinstoRad)), "MinstoRad"] <- predict(model, total)[which(is.na(total$MinstoRad))]

#impute NA for EDHour
without_na <- total[complete.cases(total), ]
without_na <- without_na %>% select(-Dependence_Level, -FiscalYear, -NIHSS, -MinstoRad)
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
  model <- ranger(formula = Dependence_Level~.-FiscalYear,
                  data=training,
                  num.trees = 25,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  
  acc[cnt] <- confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]]
  cnt <- cnt + 1}

mean(acc) #0.7633584

#evaluation with test set without outlier handling
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = Dependence_Level~.-FiscalYear,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
confusionMatrix(prediction, test$Dependence_Level)[["overall"]][["Accuracy"]] #I got 0.7790974


#outlier handling
#library("cluster")
#set.seed(123)
#km.res <- kmeans(train %>% select(-Dependence_Level), 2, nstart = 25)
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
#  model <- ranger(formula = Dependence_Level~.-FiscalYear,
#                  data=training,
#                  num.trees = 25,
#                  mtry=prod.params$mtry,
#                  min.node.size = prod.params$nodesize)
#  
#  prediction <- predictions(predict(model, testing))
#  
#  
#  acc[cnt] <- confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]]
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
#rf_model <- ranger(formula = Dependence_Level~.-FiscalYear,
#                   data=train,
#                   num.trees = 25,
#                   mtry=prod.params$mtry,importance = 'impurity',
#                   min.node.size = prod.params$nodesize)
#
#
#prediction <- predictions(predict(rf_model, test))
#confusionMatrix(prediction, test$Dependence_Level)[["overall"]][["Accuracy"]] #I got 0.6856492




#finding optimized tuned weight

#set.seed(1234)
#cross <- cvFolds(nrow(train), K=4)

#comparison <- data.frame(int =1:30)
#c=2

#for(i in 1:4){
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
#    rf_model1 <- ranger(formula = Dependence_Level~.-FiscalYear ,
#                        data=training,
#                        num.trees = 25,
#                        mtry=prod.params$mtry, 
#                        case.weights = tuned.weight,
#                        min.node.size = prod.params$nodesize)
#    
#    #evaluation
#    prediction <- predictions(predict(rf_model1, testing))
#    print(confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]])
#    result <- data.frame(int = n, accuracy = confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]])
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
#rf_model1 <- ranger(formula = Dependence_Level~. ,
#                    data=train,
#                    num.trees = 25,
#                    mtry=prod.params$mtry, 
#                    case.weights = tuned.weight,
#                    min.node.size = prod.params$nodesize)
#
##evaluation
#prediction <- predictions(predict(rf_model1, test))
#print(confusionMatrix(prediction, test$Dependence_Level)[["overall"]][["Accuracy"]]) #0.6970387


#feature selection from ranger : importance = 'impurity'
prod.params <- list(mtry = 5, nodesize = 13) 
set.seed(1234)
rf_model <- ranger(formula = Dependence_Level~.,
                   data=train,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

v<-as.vector(rf_model$variable.importance)
train1 <- train %>% select(-Dependence_Level)
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
  
  col.remove <- data.frame(int = numeric(), accuracy = numeric())
  for(n in 1:20){
    
    fi <- feature.importance[-c(1:n),]
    
    fi <- as.vector(fi[,1])
    
    #feature selection
    training <- training[,c('Dependence_Level',fi)]
    testing <- testing[,c('Dependence_Level',fi)]
    
    set.seed(1234)
    prod.params <- list(mtry = 10, nodesize = 13) 
    
    rf_model1 <- ranger(formula = Dependence_Level~. ,
                        data=training,
                        num.trees = 25,
                        mtry=prod.params$mtry, 
                        min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model1, testing))
    print(confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]])
    result <- data.frame(int = n, accuracy = confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]])
    col.remove <- rbind(col.remove,result)
  }
  
  comparison[c] <- col.remove$accuracy
  c <- c + 1
  
}

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(desc(average))
max(comparison$average) #0.7729537

important_features <- feature.importance[-c(1:comparison[1,1]),]
important_features <- as.vector(important_features[,1])

#evaluate with test set
train1 <- train[,c('Dependence_Level',important_features)]
train1 <- train1[,order(colnames(train1))]

prod.params <- list(mtry = 5, nodesize = 13) 

set.seed(1234)
rf_model <- ranger(formula = Dependence_Level~.,
                   data=train1,
                   num.trees = 25,
                   mtry=prod.params$mtry,importance = 'impurity',
                   #case.weights = tuned.weight,
                   min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(rf_model, test))
confusionMatrix(prediction, test$Dependence_Level)[["overall"]][["Accuracy"]] #0.783848

#Reducing train_data and test_data by selecting important features
train <- train[,c('Dependence_Level',important_features)]
test <- test[,c('Dependence_Level',important_features)]
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
  
  a <- data.frame(int = numeric(), accuracy = numeric())
  for(n in 1:nrow(tune.grid)) {
    prod.params <- list(mtry = as.numeric(tune.grid[n,'mtry']), 
                        nodesize = as.numeric(tune.grid[n,'nodesize']))
    
    set.seed(1234)
    rf_model <- ranger(formula = Dependence_Level~.-tuned.weight,
                       data=training,
                       num.trees = 50,
                       mtry=prod.params$mtry,
                       #case.weights = training$tuned.weight,
                       min.node.size = prod.params$nodesize)
    
    #evaluation
    prediction <- predictions(predict(rf_model, testing))
    print(confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]])
    
    result <- data.frame(grid.number = n, accuracy = confusionMatrix(as.factor(prediction), as.factor(testing$Dependence_Level))[["overall"]][["Accuracy"]])
    a <- rbind(a,result)
  }
  
  comparison[c] <- a$accuracy
  c <- c + 1
  
} 

comparison$average <- (comparison$V2 + comparison$V3 + comparison$V4 + comparison$V5)/4
comparison <- comparison %>% arrange(desc(average))
max(comparison$average) #0.7756926

prod.params <- list(mtry = as.numeric(tune.grid[comparison[1,1],'mtry']), 
                    nodesize = as.numeric(tune.grid[comparison[1,1],'nodesize']))

#train$tuned.weight <- NULL

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
  model <- ranger(formula = Dependence_Level~.-FiscalYear,
                  data=training,
                  num.trees = 50,
                  mtry=prod.params$mtry,
                  min.node.size = prod.params$nodesize)
  
  prediction <- predictions(predict(model, testing))
  
  acc[cnt] <- confusionMatrix(prediction, testing$Dependence_Level)[["overall"]][["Accuracy"]]
  cnt <- cnt + 1}

mean(acc) #0.7862705

#evaluation with test set without outlier handling
prod.params <- prod.params
set.seed(1234)
Dependence_Level_predictor <- ranger(formula = Dependence_Level~.-FiscalYear,
                                     data=train,
                                     num.trees = 50,
                                     mtry=prod.params$mtry,importance = 'impurity',
                                     min.node.size = prod.params$nodesize)

#evaluation
prediction <- predictions(predict(Dependence_Level_predictor, test))
confusionMatrix(prediction, test$Dependence_Level)[["overall"]][["Accuracy"]] #I got 0.7721519

library(pROC)
roc_obj <- multiclass.roc(as.numeric(prediction), as.numeric(test$Dependence_Level))
auc(roc_obj) #0.830995


#using XGBoost
train1 <- train
test1 <- test
train1[] <- lapply(train1, as.numeric)
test1[]<-lapply(test1, as.numeric)
train1$Dependence_Level <- train1$Dependence_Level -1
test1$Dependence_Level <- test1$Dependence_Level -1

#xgb matrix
withoutRV <- train1 %>% select(-Dependence_Level)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = train1$Dependence_Level)
withoutRV1 <- test1 %>% select(-Dependence_Level)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))

#xgboost parameters
xgb_params <- list(colsample_bytree = 0.7, #variables per tree 
                   subsample = 0.8, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 10, #tree levels
                   eta = 0.12, #shrinkage
                   eval_metric = "mlogloss", 
                   objective = "multi:softmax",
                   num_class=5,
                   gamma=0)   

#cross-validation and checking iterations
set.seed(4321)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000) 

gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F, 
                   nrounds = xgb_cv$best_iteration)

prediction_xgb <- predict(gb_dt,dtest)

#evaluation
confusionMatrix(as.factor(test1$Dependence_Level), as.factor(prediction_xgb))[["overall"]][["Accuracy"]] #0.783848

library(pROC)
roc_obj <- multiclass.roc(prediction_xgb, as.numeric(test$Dependence_Level))
auc(roc_obj) #0.8550721

