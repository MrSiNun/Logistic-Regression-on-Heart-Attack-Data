# DATA 630 Assignment 2
# Written by Daanish Ahmed
# Semester Summer 2017
# June 17, 2017
# Professor Edward Herranz

# This R script creates a logistic regression model using a dataset containing 
# information on heart attack victims.  The purpose of this assignment is to 
# generate a model to estimate the probability of a variable having a certain 
# outcome.  I am focusing on predicting the variable fstat, which represents 
# a patient's vital status at the time of his or her last follow-up session 
# (indicating whether they are alive or dead).  This script consists of several 
# components, including opening and initializing the data, exploration and 
# preprocessing, building and analyzing the logistic regression model, and 
# creating a minimal adequate model that excludes insignificant variables.



# This section of code covers opening the dataset and initializing the packages 
# that are used in this script.

# Sets the working directory for this assignment.  Please change this directory 
# to whichever directory you are using, and make sure that all files are placed 
# in that location.
setwd("~/Class Documents/2016-17 Summer/DATA 630/R/Assignment 2")

# We do not need to install or load any packages in this section of the script.  
# The logistic regression function is located in the stats package, which by 
# default is loaded automatically into R.

# Opens the CSV file "whas500.csv".
heart_atk <- read.csv(file="whas500.csv", head=TRUE, sep=",")

# End of opening the dataset.



# This section of code covers data preprocessing.  It includes exploration of 
# the original dataset, removing variables, and dealing with missing values.

# Previews the heart attack dataset.
View(heart_atk)

# Shows the descriptive statistics for all variables in the dataset.
summary(heart_atk)

# Displays the structure of the data.  This is necessary to see if there are 
# any unique identifiers (IDs) that can be removed.  Such variables are not 
# useful for the analysis and should be removed.
str(heart_atk)

# The first variable is an ID, and we remove it.
heart_atk <- heart_atk[, -1]

# Verifies that the ID variable has been removed.
str(heart_atk)

# This function checks to see how many missing values are in each variable.
apply(heart_atk, 2, function(heart_atk) sum(is.na(heart_atk)))

# Since there are no missing values in any of the variables, we do not need to 
# replace any values.

# Since we are using logistic regression, it is important to convert the 
# dependent variable fstat into a factor so that the glm function will perform
# logistic regression.
heart_atk$fstat <- as.factor(heart_atk$fstat)

# We remove several variables.  Variables 15, 16, and 17 are dates, while 
# variable 14 refers to the year, all of which are unneeded.  # Variable 19, 
# dstat, is directly correlated with the dependent variable fstat, thus we 
# remove it.  Variables 18 and 20 are independent variables that are correlated
# with each other, thus we remove them as well.
heart_atk <- heart_atk[, -c(14, 15, 16, 17, 18, 19, 20)]

# Verifies that the unwanted variables have been removed.
View(heart_atk)

# Several variables have numeric values 0 and 1 that represent categorical 
# values.  We convert these into factors.  This step is not required to 
# generate the model, but it is useful for analysis.
heart_atk$gender <- as.factor(heart_atk$gender)
heart_atk$cvd <- as.factor(heart_atk$cvd)
heart_atk$afb <- as.factor(heart_atk$afb)
heart_atk$sho <- as.factor(heart_atk$sho)
heart_atk$chf <- as.factor(heart_atk$chf)
heart_atk$av3 <- as.factor(heart_atk$av3)
heart_atk$miord <- as.factor(heart_atk$miord)
heart_atk$mitype <- as.factor(heart_atk$mitype)

# Verifies that the variables have been converted to factors.
summary(heart_atk)

# End of data preprocessing.



# This section of code covers the creation of the logistic regression model.
# It includes dividing the data into training and test datasets, creating 
# and analyzing the model, and plotting the model.

# Generates a random seed to allow us to reproduce the results.
set.seed(1234)

# The following code splits the heart attack dataset into a training set 
# consisting of 70% of the data and a test set containing 30% of the data.
ind <- sample(2, nrow(heart_atk), replace = TRUE, prob = c(0.7, 0.3))
train.data <- heart_atk [ind == 1, ]
test.data <- heart_atk [ind == 2, ]

# Generates the logistic regression model on the training data.
model <- glm(fstat ~ ., family = binomial, data = train.data)

# Displays important information including the summary statistics, the 
# values of the intercept and the coefficients, and the p-values.
summary(model)

# Builds the confusion matrix for the training data.
table(round(predict(model, train.data, type="response")), train.data$fstat)

# Builds the confusion matrix for the test data.
mypredictions <- round(predict(model, test.data, type="response"))
table(mypredictions, test.data$fstat)

# Creates the residuals plot for the data.
plot(predict(model), residuals(model), col=c("blue"))
lines(lowess(predict(model), residuals(model)), col=c("black"), lwd=2)
abline(h=0, col="grey")

# End of creating the logistic regression model.



# This section of code covers the creation of the minimal adequate model. 
# This model will remove all variables that are not significant, leaving 
# only significant variables.

# This command shows a step model that removes insignificant variables 
# one at a time until only significant variables remain.  This step is 
# needed to obtain the significant variables to generate the actual 
# minimal adequate model.
summary(step(model))

# Creates the minimal adequate model using the variables found at the 
# final iteration of the step command.
mamodel <- glm(fstat ~ age + hr + diasbp + sho + chf + av3, 
               family=binomial, train.data)

# Shows the summary statistics of the minimal adequate model.
summary(mamodel)

# Builds the confusion matrix for the training data using the minimal 
# adequate model.
table(round(predict(mamodel, train.data, type="response")), train.data$fstat)

# Builds the confusion matrix for the test data.
mypredictions <- round(predict(mamodel, test.data, type="response"))
table(mypredictions, test.data$fstat)

# Creates the residuals plot for the minimal adequate model.
plot(predict(mamodel), residuals(mamodel), col=c("blue"))
lines(lowess(predict(mamodel), residuals(mamodel)), col=c("black"), lwd=2)
abline(h=0, col="grey")

# End of creating minimal adequate model.



# The next two sections of code cover the implementation of the Naive Bayes 
# classification model.  This model is generated for the purpose of comparing
# it to my original model and seeing which one is more accurate.

# The following code covers the preprocessing of the dataset to prepare it for 
# implementation with the Naive Bayes method.

# The following packages are required for implementing the Naive Bayes model 
# and discretization.  If you have not installed these packages yet, please 
# remove the two # symbols below.
# install.packages("arules")
# install.packages("e1071")

# Loads the arules and e1071 packages into the system.
library("arules")
library("e1071")

# We need to discretize all of the variables that are not already factors.  This 
# is required before running the Naive Bayes method.  Since most of the variables 
# have already been converted, we only need to convert a few.
heart_atk$age <- discretize(heart_atk$age, "frequency", categories=6)
heart_atk$hr <- discretize(heart_atk$hr, "frequency", categories=6)
heart_atk$sysbp <- discretize(heart_atk$sysbp, "frequency", categories=6)
heart_atk$diasbp <- discretize(heart_atk$diasbp, "frequency", categories=6)
heart_atk$bmi <- discretize(heart_atk$bmi, "frequency", categories=6)

# Verifies that all variables are factors.
str(heart_atk)

# End of Naive Bayes preprocessing section.



# The following code covers the creation of the Naive Bayes model.  It involves 
# splitting the data into training and test sets, creating and analyzing the 
# model, and plotting the results.

# Generates a random seed to allow us to reproduce the results.
set.seed(1234)

# We once again split the data into a training set containing 70% of the data 
# and a test set containing 30% of the data.
ind <- sample(2, nrow(heart_atk), replace = TRUE, prob = c(0.7, 0.3))
train.data <- heart_atk[ind == 1, ]
test.data <- heart_atk[ind == 2, ]

# Generates the Naive Bayes classification model on the training data.
model <- naiveBayes(fstat ~ ., train.data)

# Displays the model output, including probabilities for each variable.
print(model)

# Builds the confusion matrix for the training data.
table(predict(model, train.data), train.data$fstat)

# Builds the confusion matrix for the test data.
table(predict(model, test.data), test.data$fstat)

# Generates a mosaic plot showing the predicted values of fstat compared to 
# the actual values.  The blue color shows when the prediction is equal to 
# the actual value.  Red indicates all misclassified instances.
mosaicplot(table(predict(model, test.data), test.data$fstat), shade=TRUE, 
           main="Predicted vs. Actual FSTAT")

# End of creating Naive Bayes classification model.

# End of script.

