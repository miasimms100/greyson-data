# IMPORT ESSENTIAL LIBRARIES
library(caret)
library(skimr)
library(tidyverse)
library(RANN)
library(glmnet)
library(ROCR)
#MODEL
library(randomForest)

# IMPORT DATA
greyson_data <- read.csv("Greyson.csv", header = T)

summaryStats <- skim(greyson_data)
summaryStats


greyson_data$Renewal <- as.factor(greyson_data$Renewal) # Factor  - categorical

greyson_data$CustomerID <- NULL #remove customer ID 

# PARTITION INTO TRAINING AND TEST DATA-------------------------------------------------------
set.seed(502) # Set seed for reproducibility
index <- createDataPartition(greyson_data$Renewal, p=.8, list = FALSE)
train <- greyson_data[index,] # Create the training data 
test  <- greyson_data[-index,] # Create the test data

dim(train)
dim(test)

# IMPUTE -----------------------------------------------------------------
preProc <- preProcess(train, method='knnImpute') 

# Use impute method to impute in the training and testing data sets separately.
train <- predict(preProc, newdata=train)
test  <- predict(preProc, newdata=test)

X <- select(greyson_data, -Renewal)
#y <- select(greyson_data$Renewal)


# MAKE DUMMY VARIABLES-----------------------------------------------
dummies_model <- dummyVars(~., data=X)

dummy_train <- data.frame(predict(dummies_model, newdata = train))
train <- cbind(Renewal=train$Renewal, dummy_train)

dummy_test<- data.frame(predict(dummies_model, newdata = test))
test <- cbind(Renewal=test$Renewal,dummy_test)



# LOGISTIC REGRESSION------------------------------------------

# Fit Logistic Regression Model with Lasso 
model <- train(
  Renewal~.,
  data = train,
  method = "glmnet",
  trControl = trainControl(
    method = "cv", 
    number = 5
  )
) 

# Plot variable importance
plot(varImp(model))

# List the Important Coefficients Selected
coef(model$finalModel, model$bestTune$lambda)


# GET PREDICTIONS USING TESTING SET DATA------------------------
prob <- predict(model, test, 
                type="prob")

# EVALUATE MODEL PERFORMANCE 
# Get AUC and ROC curve for LASSO Model.

pred <- prediction(prob[,2], test$Renewal)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

AUC <- unlist(slot(performance(pred, "auc"), "y.values"))
AUC

# END LOGISTIC REGRESSION MODEL ---------------------------------------------


