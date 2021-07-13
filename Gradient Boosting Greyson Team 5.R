# IMPORT ESSENTIAL LIBRARIES
library(caret)
library(skimr)
library(tidyverse)
library(RANN)
library(glmnet)
library(ROCR)

Greyson_data <- read.csv("Greyson.csv", header = T)  
SummaryStats <- skim(Greyson_data)  
SummaryStats

Greyson_data$Renewal <- as.factor(Greyson_data$Renewal) # Factor  - categorical

Greyson_data$CustomerID <- NULL 
set.seed(502) 
index <- createDataPartition(Greyson_data$Renewal,
                             p = .8,list = FALSE)  
Greyson_train <- Greyson_data[index,]  
Greyson_test <- Greyson_data[-index,]

Greyson_train$HomeValue[is.na(Greyson_train$HomeValue)]<-10  
Greyson_test$HomeValue[is.na(Greyson_test$HomeValue)]<-10  

#impute on train and test set

preProcess_missingdata_model <- preProcess(Greyson_train, 
                                           method='medianImpute')  
preProcess_missingdata_model 
Greyson_train<-predict(preProcess_missingdata_model, 
                       newdata=Greyson_train)

preProcess_missingdata_model <- preProcess(Greyson_test, 
                                           method='medianImpute')  
preProcess_missingdata_model 
Greyson_test<-predict(preProcess_missingdata_model, 
                       newdata=Greyson_test)  
#separate response and predictor variables
x <- select(Greyson_data, -Renewal)
y <- select(Greyson_data, Renewal)

#make dummy variables
dummies_model <- dummyVars(~., data=x)

dummy_train <- data.frame(predict(dummies_model, newdata = Greyson_train))
Greyson_train <- cbind(Renewal=Greyson_train$Renewal, dummy_train)

dummy_test<- data.frame(predict(dummies_model, newdata = Greyson_test))
Greyson_test <- cbind(Renewal=Greyson_test$Renewal,dummy_test)

#set up parallel processing to run model
library(doParallel)
num_cores<-detectCores(logical=FALSE)
num_cores
cl <- makePSOCKcluster(num_cores-1)
registerDoParallel(cl)

set.seed(502)
grad_boost_model <- train(Renewal ~ .,
                          data = Greyson_train, 
                          method = 'xgbTree', 
                          trControl = trainControl(method = "cv",
                                                   number = 5), 
                          verbose = FALSE 
)
grad_boost_model 
#plot the model 
plot(grad_boost_model) 
#important predictor variables 
important_variables <- varImp(grad_boost_model) 
plot(important_variables) 

#work on tuning the model
#tuning parameter selection ideas for model to try 
#training set cross validation 
xgb_grid <- expand.grid(nrounds = c(25,50,75,100,200),
                        eta = c(0.025, 0.05, 0.075, 0.1),
                        max_depth = c(2, 3, 4, 5, 6),
                        gamma = 0, 
                        colsample_bytree = 1, 
                        min_child_weight = 1, 
                        subsample = 1) 


tune_gradboost_model <- train(Renewal ~., 
                              data = Greyson_train, 
                              method = 'xgbTree', 
                              trControl = trainControl(method = "cv", number = 5),
                              tuneGrid = xgb_grid,
                              verbose = FALSE) 
plot(tune_gradboost_model) 
tune_gradboost_model
#evaluate model on unseen data, the test set using ROC and AUC
Greyson_prob <- predict(grad_boost_model, newdata = Greyson_test, type = 'prob') 
Greyson_prob  

library(ROCR) 
pred_gbm <- prediction(Greyson_prob[,2], Greyson_test$Renewal) 
perf_gbm<- performance(pred_gbm, 'tpr', 'fpr') 
#Plot ROC curve 
plot(perf_gbm, colorize = TRUE)  
#AUC 
AUC_untuned <- unlist(slot(performance(pred_gbm, 'auc'), 'y.values'))  
AUC_untuned 

#tuned reporting
Greyson_prob_tuned <- predict(tune_gradboost_model, newdata = Greyson_test, type = 'prob') 
Greyson_prob_tuned

pred_gbm_tuned <- prediction(Greyson_prob_tuned[,2], Greyson_test$Renewal) 
perf_gbm_tuned <- performance(pred_gbm_tuned, 'tpr', 'fpr') 
#Plot ROC curve 
plot(perf_gbm_tuned, colorize = TRUE)  
#AUC 
AUC_tuned <- unlist(slot(performance(pred_gbm_tuned, 'auc'), 'y.values'))  
AUC_tuned 

#stop running parallel processing
stopCluster(cl)
registerDoSEQ()

AUC_untuned 
AUC_tuned 
