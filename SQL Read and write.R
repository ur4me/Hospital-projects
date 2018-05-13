rep_date = NULL
beg_date = "2015-01-01"
base_dir = "G:/groups/ivise/"
prod_trees = 2500
optimize.model = TRUE

  
  library(RODBC)
  library(dplyr)
  library(ranger)
  library(xgboost)
  
  if(!exists('rep_date')){ rep_date <- Sys.Date()}
  if(is.null(rep_date)){ rep_date <- Sys.Date()}
  if(!exists('base_dir')){ base_dir <- "G:/groups/ivise/"}
  
  monnb <- function(d) { 
    lt <- as.POSIXlt(as.Date(d, origin="1900-01-01")); lt$year*12 + lt$mon 
  } 
  
  ###WRITE TO LOG
  log_con <- file(paste0(base_dir, "production/los/logs/create.log"), open="a")
  cat(paste0("Starting model creation script at: ",Sys.time(),"..."), file=log_con,  sep="\n")
  cat(paste0("Input date: ",rep_date,"..."), file=log_con,  sep="\n")
  
  ###GRAB DATA HERE
  dbred_con <- odbcDriverConnect('driver={SQL Server};server=VMMH1SQL002;database=db_red;trusted_connection=true')
  
  qry <- gsub("%END_DATE%", as.character(Sys.Date()), 
              gsub("%BEG_DATE%", beg_date,
                   paste0(readLines(paste0(base_dir, "production/los/sql/records.txt"),warn = F), collapse="\n")))
  
  qry_data <- sqlQuery(dbred_con,qry, stringsAsFactors = F)
  
  RODBC::odbcCloseAll()
  ##########
  
  cat(paste0("Successfully pulled in new records at: ",Sys.time(),"..."), file=log_con,  sep="\n")
  
  if(length(qry_data) > 1 & !is.null(qry_data)){
    cat(paste0("Max date of new records is: ",max(qry_data$admission_date),"..."), file=log_con,  sep="\n")
  } else {
    cat(paste0("Unable to pull new records, exiting script"), file=log_con,  sep="\n")
    q()
  }
  
  ###DEFINE RESPONSE VARIABLE AND PREDICTORS
  predictors <- c("sub_specialty", "Laparoscopic", "surgical_theatre_group",
                  "age_years", "ASA",  "CPAC_score",paste0("p",c(1:8),"_code"))
  response <- "LOS_Total_days"
  
  ###EVALUATION COLUMNS
  eval_cols <- c("waitlist_encounter_id", "admission_date", "discharge_date")
  
  elective_data <- qry_data[,c(response, predictors, eval_cols)]
  
  
  
  
  
  ###WRITE PREDICTIONS TO DATABASE
  dbdump_con <- odbcDriverConnect('driver={SQL Server};server=VMMH1SQL002;database=datadumps;trusted_connection=true')
  #sqlQuery(dbdump_con, "delete from dbo.elective_los_pred_prod")
  sqlSave(dbdump_con, waitlist, tablename = "dbo.elective_los_pred_prod", append = TRUE,
          varTypes = c(surgical_theatre_booking_date="date",discharge_date="date",
                       model_date="date", pred_date="date", pred_made="datetime"))
  RODBC::odbcCloseAll()