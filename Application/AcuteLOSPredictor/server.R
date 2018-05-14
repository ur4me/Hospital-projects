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
library(shiny)




load("D:/etc/Shiny/AcuteLOSPredictor/data/acute_los_predictor.rda")

shinyServer(function(input, output){
  
  
  values <- reactiveValues()
  
  newEntry <- observe({
    values$df$Age <- as.integer(input$Age)
    values$df$EDDisch <- as.integer(input$EDDisch)
    values$df$EDHour <- as.integer(input$EDHour)
    values$df$MinstoRad <- as.integer(input$MinstoRad)
    values$df$month <- as.numeric(input$month)
    values$df$NIHSS <- as.numeric(input$NIHSS)
    values$df$Stay36HrsPlus <- as.integer(input$Stay36HrsPlus)
    values$df$StrokeUnit <- as.integer(input$StrokeUnit)
    values$df$TACS <- as.integer(input$TACS)
  })
  output$results <- renderPrint({
    ds1 <- values$df
   a <- predictions(predict(acute_los_predictor, data.frame(ds1)))
    a <- round(a)
 
    cat(a)
  })
})