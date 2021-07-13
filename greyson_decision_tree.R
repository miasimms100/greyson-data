# Libraries
library(caret)
library(skimr)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(ROCR)
library(RANN)
library(imputeMissings)

# Import data
greyson_data <- read.csv(file = "MSBA/MSBA 635/Greyson.csv", header=T)

# Exploratory Data Analysis
summaryStats <- skim(greyson_data)
summaryStats

# Create dummy variables
renewal_predictors <- select(greyson_data, -Renewal)
dummies_model <- dummyVars(~ ., data = renewal_predictors)
renewal_predictors_dummy <- data.frame(predict(dummies_model, newdata = greyson_data))
greyson_data <- cbind(Renewal = greyson_data$Renewal, renewal_predictors_dummy)

# Convert to Factor
greyson_data$Renewal <- as.factor(greyson_data$Renewal)
greyson_data$Renewal <- fct_recode(greyson_data$Renewal, notrenewed = '0', renew = '1')

# Set seed
set.seed(502)

# Partition Data
index <- createDataPartition(greyson_data$Renewal, p = 0.8, list = FALSE)
greyson_train <- greyson_data[index,]
greyson_test <- greyson_data[-index,]

# Impute missing variables
trainMedians <- compute(greyson_train, method = 'median/mode')
greyson_train <- impute(greyson_train, object = trainMedians)
sum(is.na(greyson_train)) # Should be 0 if all missing are gone
greyson_test <- impute(greyson_test, object = trainMedians)
sum(is.na(greyson_test)) # Should be 0 if all missing are gone

# Create model
greyson_model <- train(Renewal ~ .,
                       data = greyson_train,
                       method = "rpart",
                       trControl = trainControl(method = 'cv', number = 5,
                                                classProbs = TRUE,
                                                summaryFunction = twoClassSummary),
                       metric = 'ROC')

greyson_model

plot(greyson_model)
plot(varImp(greyson_model))

# Plot model
rpart.plot(greyson_model$finalModel, type=5)

prob_renew <- predict(greyson_model, greyson_test, type='prob')

# Evaluate and plot ROC
pred_tree <- prediction(prob_renew[,2], greyson_test$Renewal, label.ordering = c("notrenewed","renew"))
perf_tree <- performance(pred_tree, "tpr", "fpr")
plot(perf_tree, colorize=TRUE)

#Get the AUC
auc_tree <- unlist(slot(performance(pred_tree, "auc"), "y.values"))
auc_tree
