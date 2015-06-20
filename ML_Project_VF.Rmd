---
title: "ML_Project"
author: "Ilham"
date: "Wednesday, June 17, 2015"
output: html_document
---
Practical Machine Learning - Course Project

Introduction

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3dQPkfdGB

The training data consist of accelerometer data collected from the belt, forearm, arm, and dumbell, and a label ("classe"") identifying the quality of the activity. The testing data consists of accelerometer data without the "classe"" label. The goal of this project is to predict the manner in which they did the exercise. 

# Load libraries and data

```{r, message=FALSE, warning=FALSE}
library(caret)
```

```{r}
train <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
test <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```

# Exploratory Analysis

Check dimension, names and take a look at the first rows

```{r, results='hide'}
dim(train)
names(train)
head(train)

```

The data set has `r nrow(train)` observations and `r ncol(train)` possible predictors. 

# Preprocess data

Create cross validation set: Split the training data in a train and test set to validate the model.

```{r}
set.seed(150)
index <- createDataPartition(y=train$classe, p=0.6, list=F)
train_t <- train[index, ]
train_v <- train[-index, ]
```

Remove variables with a lot of missing values (> 50%). I will not use any imputation method, because I will build my model with Random Forest algorithm, which has an effective method for estimating missing data and maintaining accuracy when a large proportion of the data are missing.

```{r}
indexNA <- sapply(train_t, function(x) mean(is.na(x))) > 0.5
train_t <- train_t[, indexNA==F]
train_v <- train_v[, indexNA==F]
```

Remove variables with nearly zero variance, and variables that are not useful for our prediction. 

Check and remove predictors with zero variance

```{r}
nzv <- nearZeroVar(train_t)
train_t <- train_t[, -nzv]
train_v <- train_v[, -nzv]
```

After removing the zero variance predictors, the set has `r ncol(train_t)` possible predictors.

Now, remove the frist 5 variables that are not useful for our prediction: timestamp, user-name, new-window, num_window and X

```{r}
train_t <- train_t[, -(1:5)]
train_v <- train_v[, -(1:5)]
```

There are still `r ncol(train_t)` possible predictors. According to the documentation, Random Forest can handle thousands of input variables without variable deletion.So, we think we may have a good performance with `r ncol(train_t)` predictors.

# Model Building

As mentioned in the documentation, random forests algorithm gives estimates of what variables are important in the classification and generates an internal unbiased estimate of the generalization error as the forest building progresses.

We fit the model on train_t, and instruct the "train" function to use 5-fold cross-validation to select optimal tuning parameters for the model.

The 'classe' variable is the outcome, the attribute we want to predict.

```{r,message=FALSE,warning=FALSE}
fitControl <- trainControl(method="cv", number=5, verboseIter=F)
# fit model on train_t
fit <- train(classe ~ ., data=train_t, method="rf", trControl=fitControl)
```

```{r}
# print final model 
fit$finalModel
```

# Testing the model

Now, we use the fitted model to predict the label ("classe") in train_v, and show the confusion matrix to compare the predicted versus the actual labels.

```{r}
# use model to predict classe in validation set (train_v)
preds <- predict(fit, newdata=train_v)
```

# Out of Sample Error

According to the documentation, random forest algorithm does not overfit, and performs cross validation internally. 

```{r}
# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(train_v$classe, preds)
```
 
The confusion matrix shows high accuracy on the test set. The accuracy is 99.75%.
Therefore, the predicted out-of-sample error is 0.25 %.

The out-of-sample error calculation: 

```{r}
outOfSampleError <- sum(preds != train_v$classe) * 100 / nrow(train_v)
outOfSampleError 
```

# Re-training the Selected Model

Before predicting on the test set, it is important to train the model on the full training set (train), rather than using a model trained on a reduced training set (train_t). So, we will repeat everything we did above on train and test:

```{r}
nzv <- nearZeroVar(train)
train <- train[, -nzv]
test <- test[, -nzv]
```

```{r}
indexNA <- sapply(train, function(x) mean(is.na(x))) > 0.5
train <- train[, indexNA==F]
test <- test[, indexNA==F]
```

```{r}
train <- train[, -(1:5)]
test <- test[, -(1:5)]
```

```{r}
# re-fit model using full training set (train)
fitControl <- trainControl(method="cv", number=5, verboseIter=F)
fit <- train(classe ~ ., data=train, method="rf", trControl=fitControl)
```

# Making Test Set Predictions
Now, we use the model to predict the label for the observations in test, and write those predictions to individual files:

```{r}
# predict on test set
preds <- predict(fit, newdata=test)
preds
```

# Write up

```{r}
# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
pml_write_files(preds)
```

