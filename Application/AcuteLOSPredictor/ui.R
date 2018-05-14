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

shinyUI(pageWithSidebar(
  headerPanel("Acute LOS Predictor"),
  sidebarPanel(
    p("Select data field to predict Acute LOS"),
    sliderInput(
      inputId = "Age",
      label = h3("Age:"),
      min = 19,
      max = 104,
      step = 1,
      value = 73),

    radioButtons("EDDisch", label = h3("EDDisch:"),
                 choices = list("No" = "0","Yes" = "1"), 
                 selected = "0"),
    sliderInput(
      inputId = "EDHour",
      label = h3("EDHour:"),
      min = 0,
      max = 23,
      step = 1,
      value = 13),
    sliderInput(
      inputId = "MinstoRad",
      label = h3("MinstoRad:"),
      min = 0,
      max = 200,
      step = 1,
      value = 88),
    sliderInput(
      inputId = "month",
      label = h3("month:"),
      min = 1,
      max = 12,
      step = 1,
      value = 7),
    sliderInput(
      inputId = "NIHSS",
      label = h3("NIHSS:"),
      min = 0,
      max = 29,
      step = 1,
      value = 3),
    radioButtons("Stay36HrsPlus", label = h3("Stayed more than 36 hours?:"),
                 choices = list("No" = "0","Yes" = "1"), 
                 selected = "1"),
    radioButtons("StrokeUnit", label = h3("StrokeUnit:"),
                 choices = list("No" = "0","Yes" = "1"), 
                 selected = "1"),
    radioButtons("TACS", label = h3("TACS:"),
                 choices = list("No" = "0","Yes" = "1"), 
                 selected = "1")
    
  
    
  ),
  mainPanel(
    h3("Estimated Acute Length of stay (days):"),
    h2(verbatimTextOutput("results")),
    p("Please note that this is just an estimation."),
    p("That means this value should be different to actual."),
    style = "position:fixed;right:1px;")))