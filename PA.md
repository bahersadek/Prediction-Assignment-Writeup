---
title: "Prediction Assignment Writeup by Baher Anwar"
author: "Baher Anwar"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---
**Synopsis:** This project involves analyzing of the Weight Lifting Exercises (WLE) Dataset.  
Young health participants performed one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:  
- Class A: Exactly according to the specification.  
- Class B: Throwing the elbows to the front.  
- Class C: Lifting the dumbbell only halfway.  
- Class D: Lowering the dumbbell only halfway.  
- Class E: Throwing the hips to the front.     

## Exploring the Weight Lifting Exercises Data 

### Loading the data.


```r
## load libraries
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(data.table)
## load data 
pmlDT <- read.csv("CSV\\pml-training.csv")
pmlTST <- read.csv("CSV\\pml-testing.csv")
```

First i Checking which column names are common among testing and training, so we can exclude the ones who are not common.  
Checking the classe balance in the training set to see whether there is anything in particular i should be concerned with.   



```r
clen <- length(intersect(colnames(pmlDT),colnames(pmlTST)))
barplot(table(pmlDT$classe))
```

![](PA_files/figure-html/LengthCalc-1.png)<!-- -->

**159**  variables in common.

Inspecting if there are some features that only include NA in the testing set 
as it should not be used in model training. 


```r
test_na <-sapply(pmlTST, FUN=function(x){
  length(which(is.na(x)))
})
nalen <- length(names(test_na[test_na==20]))
```
There are **100** features that are only NA, removing these from the training set (and test set).


### Partitioning the Dataset

 we will split our data into a training data set (60%) and a testing data set (40%). This will allow us to estimate the out of sample error of our predictor.
 

```r
## splitting data for model validation
set.seed(12345)

inTrain <- createDataPartition(pmlDT$classe, p=0.6, list=FALSE)
training <- pmlDT[inTrain,!names(pmlDT) %in% names(test_na[test_na>0])]
testing <- pmlDT[-inTrain,!names(pmlDT) %in% names(test_na[test_na>0])]

dim(training); dim(testing);
```

```
## [1] 11776    60
```

```
## [1] 7846   60
```
Checking if the split generates any all-NA features in either split

```r
table(sapply(training, function(x){
  all(is.na(x))
}))
```

```
## 
## FALSE 
##    60
```

```r
table(sapply(testing, function(x){
  all(is.na(x))
}))
```

```
## 
## FALSE 
##    60
```
## Building Models
Several models were considered and run for this analysis Such :  
Linear Discriminant Analysis (lda)    
Recursive Partitioning and Regression Trees (rpart).  
Boosting.    
Generalized linear models.
Random forests.   
Each used to predict the class on the test data produced in data preprocessing. 


```r
Modctl <- trainControl(method="cv", number=10, repeats=1)
```

```
## Warning: `repeats` has no meaning for this resampling method.
```
### Decision Tree Model

```r
FitDT <- train(classe ~., method="rpart", data = training, trControl=Modctl)
```
### Linear Discriminant Analysis Model

```r
FitLDA <- train(classe~., method="lda", data = training, trControl=Modctl)
```
### Random Forest Model

```r
set.seed(1234)
cv <- trainControl(method="cv", number=3)
modRf <- train(classe~.,data=training, method="rf", trControl=cv, verbose=F)
```

## Apply Prediction on Training Data.
### Predicting Based on Decision Tree.

```r
predictionDT <- predict(FitDT, newdata =testing)
confusionMatrix(predictionDT, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    0    0    0
##          C    0    0    0    0    0
##          D    0    0    0    0    0
##          E    0    0 1368 1286 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6617          
##                  95% CI : (0.6511, 0.6722)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5695          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.0000   0.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   0.5856
## Pos Pred Value         1.0000   1.0000      NaN      NaN   0.3521
## Neg Pred Value         1.0000   1.0000   0.8256   0.8361   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.0000   0.0000   0.1838
## Detection Prevalence   0.2845   0.1935   0.0000   0.0000   0.5220
## Balanced Accuracy      1.0000   1.0000   0.5000   0.5000   0.7928
```
### Predicting Based on Linear Discriminant.

```r
predictionLDA <- predict(FitLDA, newdata =testing)
confusionMatrix(predictionLDA, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1517    0    0    0
##          C    0    1 1368    0    0
##          D    0    0    0 1286    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9999     
##                  95% CI : (0.9993, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9998     
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9993   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   0.9998   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   0.9993   1.0000   1.0000
## Neg Pred Value         1.0000   0.9998   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1933   0.1744   0.1639   0.1838
## Detection Prevalence   0.2845   0.1933   0.1745   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9997   0.9999   1.0000   1.0000
```
### Predicting Based on Random Forest.

```r
predictionRF <- predict(modRf, newdata =testing)
confusionMatrix(predictionRF, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1517    0    0    0
##          C    0    0 1368    0    0
##          D    0    0    0 1286    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9999     
##                  95% CI : (0.9993, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9998     
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9993   1.0000   1.0000   1.0000
## Specificity            0.9998   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         0.9996   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   0.9998   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1933   0.1744   0.1639   0.1838
## Detection Prevalence   0.2846   0.1933   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9999   0.9997   1.0000   1.0000   1.0000
```

## Apply Prediction on Testing Data 
### Predicting Based on Decision Tree.

```r
pDT <- predict(FitDT, pmlTST)
pDT
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```
### Predicting Based on Linear Discriminant.

```r
pRF <- predict(FitLDA, pmlTST)
pRF
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```
### Predicting Based on Random Forest.

```r
pBO <- predict(modRf, pmlTST)
pBO
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```


## Conclusion
In this project, we were able to fit a model that predicts with over 99 percent accuracy the class of dumbell exercises given a series of movements. A random forest model proved to be the most accurate model, however it was also computationally intensive and may take a long time if the number of variables increases greatly.
