Predicting the "classe" Variable using Multiple Variables 
========================================================
## Executive Summary

This analysis is performed to predict the "classe" variable by using classification agorithms multiple variables from a study conducted by Velloso et al. (2013). The analysis focuses on predicting  the five methods in which the particpants performed dumbbell Biceps lifts in 5 different ways.

## Data
The data used in this analysis is obtained from from this source: http://groupware.les.inf.puc-rio.br/har. The data was collected on six individuals who were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).


## Loading and Preprocessing Data
The required libraries and data for this analysis are first loaded. The data from the following sources: "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" and "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"was downloaded and placed in a local directory. Next the 
all the columns with empty or null values and the first six columns that are not relevant to the analysis are are removed from the training and the testing datasets.


```r
setwd("F:/practicalMachine/Project")
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
training<-read.csv("pml-training.csv", header=TRUE)
testing<- read.csv("pml-testing.csv", header=TRUE)

NAs<-apply(training,2,function(x) {sum(is.na(x)| x=="")})

training<- training[,which(NAs == 0)]
training<-training[,-c(1:6)]

## processing the testing dataset
NAst<-apply(testing,2,function(x) {sum(is.na(x)| x=="")})
testing<- testing[,which(NAst == 0)]
names<-names(training)
names<- names[-length(names)]
testing<-testing[,names]
```

### Data Analysis
### Model
Once the data preprecessing is completed, the model was built using the Random Forest algorithm which uses cross validation with 4 k-folds is used. These will give us 75% of data for training and 25% of data for cross validation as shown below. 

```r
## Creating folds for training
trainFolds<-createFolds(y=training$classe,k=4,list=TRUE,returnTrain=T)
tFolds<-sapply(trainFolds,length)
tFolds
```

```
## Fold1 Fold2 Fold3 Fold4 
## 14717 14715 14716 14718
```

```r
## Creating Folds for Cross validation
crossValFolds<-createFolds(y=training$classe,k=4,list=TRUE,returnTrain=F)
cvFolds<-sapply(crossValFolds,length)
cvFolds
```

```
## Fold1 Fold2 Fold3 Fold4 
##  4905  4906  4906  4905
```

```r
## Percentages of traing and cv data
tFolds/(tFolds+cvFolds)
```

```
## Fold1 Fold2 Fold3 Fold4 
##  0.75  0.75  0.75  0.75
```
Then the model for the training data is developed as follows.


```r
modFit <- train(classe ~.,data = training,method="rf", trControl = trainControl(method = "cv", number = 4))
```

The parameter "trainControl(method = "cv", number = 4))" uses 4 folds data slicing and will divide the trainingdata set 75% for training data and 25% for cross validation data. The resulting model is shown below.We can see that there are 19622 samples and 53 predictors, and 5 classes: 'A', 'B', 'C', 'D', and 'E' in the final model.  Accuracy was used to select the optimal model and in our case the Accuracy is 0.999 showing a very high Accuracy rate. 


### Results


```r
modFit
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 14717, 14717, 14716, 14716 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.002   
##   30    1         1      1e-04        1e-04   
##   50    1         1      9e-04        0.001   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

### Prediction

Then the optimal RF model developed using the training dataset is used to do preditcion using the testing data and answer the 20 questions in the assignment. The results of the final Model are shown below.


```r
options(warn=-1)
prediction<-predict(modFit,testing)
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.14%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5578    1    0    0    1   0.0003584
## B    4 3791    2    0    0   0.0015802
## C    0    5 3417    0    0   0.0014611
## D    0    0   10 3205    1   0.0034204
## E    0    0    0    3 3604   0.0008317
```


The results show an estimated errorr rate of 0.13%. 

## REFERENCES
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


