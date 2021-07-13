#Using a Quality assurance model to perform benchmarks among age groups, Nirosha ran the Random Forest algorithm...NT
#in doing so, Nirosha hopes to achieve an accuracy rate higher than 75% to see if she has a good fit...NT
#model for determining which age group among the customers made recurring purchases...NT
necessary_packages <- c("dplyr", "ggplot2", "gapminder", "readr", "tidyverse")

packageTable <- data.frame(necessary_packages)

Required_Packages <- function() {
  install.packages(necessary_packages, repos = "http://cran.rstudio.com")
  #Exploratory DA
  #Install packages
  library(tidyverse)
  library(skimr)
  library(tidyr)
  library(stringr)
  library(caret)
  library(dplyr)
}

Required_Packages()

Greyson_data <- read.csv(file = "Greyson.csv", header=T)

Age_Groups_Among_Customers_Recurrence <- Greyson_data %>%
  select(Renewal, Age, MagazineStatus, DollarsPerIssue, TotalPaidOrders)

summaryStats <- skim(Age_Groups_Among_Customers_Recurrence) #skim(Greyson_data)
View(summaryStats)
summaryStats #is like summary but more detailed...NT

#Checking for Null, NAs, and Empty...NT
any(is.na(Age_Groups_Among_Customers_Recurrence$Renewal)) #No missing data...NT
any(is.na(Age_Groups_Among_Customers_Recurrence$Age)) #No missing data...NT
any(is.na(Age_Groups_Among_Customers_Recurrence$MagazineStatus)) #No missing data...NT
any(is.na(Age_Groups_Among_Customers_Recurrence$DollarsPerIssue)) #NA exists...NT
any(is.na(Age_Groups_Among_Customers_Recurrence$TotalPaidOrders)) #NA exists...NT

#How many ways can we arrange the given data for Renewal, Age, and Magazine Status...NT
#You might wonder why we would possibly care about the factorial function. 
#It's very useful for when we're trying to count how many different orders 
#there are for things or how many different ways we can combine things. For example, 
#how many different ways can we arrange n things? We have n choices for the first thing...NT
#Using the Factor function let's determine the number of levels 
checkmakevector <- as.vector(Age_Groups_Among_Customers_Recurrence)
checkmakevector

giveMe_CombinationsPlease <- as.factor(checkmakevector) #We are going to exclude NA from data to determine levels...NT

giveMe_CombinationsPlease
#Exclude missing data meaning NAs...NT
#giveMe_CombinationsPlease$DollarsPerIssue [is.na(giveMe_CombinationsPlease$DollarsPerIssue) == TRUE] <- ""
#Greyson_data$Renewal<-fct_recode(Greyson_data$Renewal, notrenewed = "0", renew = "1") 

#Here is the start of data cleaning using preprocessing as it is a very decisive way of determining sentiment...NT
preProcess_missingdata_model <- preProcess(checkmakevector, method='medianImpute') 

preProcess_missingdata_model

Greyson_makeData_Significant <- predict(preProcess_missingdata_model, newdata=checkmakevector) 

skim(Greyson_makeData_Significant)
Greyson_makeData_Significant

#make dummy variables...NT 
greyson_matrix <- model.matrix(~ ., data = Greyson_makeData_Significant) #Figured it out here the code is actually removing the Renewal column...TCDW

greyson_matrix

greyson_matrix <- data.frame(greyson_matrix[,-1]) #get rid of intercept 

greyson_matrix
# You notice that Age, MagazineStatusB, C, E, N, O, S should be treated as a number for 1 for subscribe 0 for doesn't and age should just be an integer...NT
#Now let's transform the data...NT
Greyson_trainBetter <- transform(
  greyson_matrix,
  Renewal=as.integer(Renewal),
  Age=as.integer(Age),
  MagazineStatusB=as.integer(MagazineStatusB),
  MagazineStatusC=as.integer(MagazineStatusC),
  MagazineStatusE=as.integer(MagazineStatusE),
  MagazineStatusN=as.integer(MagazineStatusN),
  MagazineStatusO=as.integer(MagazineStatusO),
  MagazineStatusS=as.integer(MagazineStatusS),
  DollarsPerIssue=as.factor(DollarsPerIssue),
  TotalPaidOrders=as.factor(TotalPaidOrders)
)

sapply(Greyson_trainBetter, class)

summary(Greyson_trainBetter)

#Splitting the data
set.seed(502) 

trainindex <- createDataPartition(Greyson_trainBetter$Renewal, p = .8, list = FALSE) 
head(trainindex)
trainindex


Greyson_train <- Greyson_trainBetter[trainindex,] #Greyson_data[index,] 

Greyson_test <- Greyson_trainBetter[-trainindex,]

#Here is the start of data cleaning using preprocessing as it is a very decisive way of determining sentiment...NT
preProcess_missingdata_model <- preProcess(Greyson_train, method='medianImpute') 

preProcess_missingdata_model

# Use impute method to impute in the training and testing data sets separately.
ttrain_predict <- predict(preProcess_missingdata_model, newdata=Greyson_train)
ttest_predict  <- predict(preProcess_missingdata_model, newdata=Greyson_test)

ttrain_predict
ttest_predict


library(randomForest)
require(caTools)

sample <- sample.split(Greyson_trainBetter$Renewal, SplitRatio = .75)

ToRun_Forest_trainData <- subset(Greyson_trainBetter, sample == TRUE)
ToRun_Forest_testData <- subset(Greyson_trainBetter, sample == FALSE)

dim(ToRun_Forest_trainData)
dim(ToRun_Forest_testData)
ToRun_Forest_trainData
dim(Greyson_trainBetter)

QA_Benchmark_model_with_rf <- randomForest(Renewal ~ Age + MagazineStatusB + MagazineStatusE + MagazineStatusC + MagazineStatusN + MagazineStatusO + MagazineStatusS,
                                           data=ToRun_Forest_trainData)
QA_Benchmark_model_with_rf


QA_Benchmark_model_with_rf2 <- randomForest(Renewal ~ Age,
                                           data=ToRun_Forest_trainData)
QA_Benchmark_model_with_rf2

#It's just the proportion of votes or prob would be meaningful for other types of alogrithms but since this is regression then its trees in the ensemble only matter...TCDW
p1 = predict(QA_Benchmark_model_with_rf, Greyson_trainBetter)
p2 = predict(QA_Benchmark_model_with_rf, Greyson_trainBetter, norm.votes = TRUE)

identical(p1,p2)
#[1] TRUE

