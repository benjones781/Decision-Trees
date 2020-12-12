#I've churned butter once or twice...
#identifying credit card churn with decision trees
#data set: https://www.kaggle.com/sakshigoyal7/credit-card-customers

library(caTools)
library(caret)
library(tree)
library(randomForest)
library(adabag)

#randomize row order of a data frame
shuffle <- function(df) {
  rows <- sample(nrow(df))
  return(df[rows, ])
}

customers <- read.csv('BankChurners.csv')
customers <- customers[-c(1, 22, 23)] #remove unnecessary columns

set.seed(74)
random_sample <- sample.split(customers$Attrition_Flag, SplitRatio = .75) #preserves ratio of target classification (Attrition_Flag)
train <- shuffle(subset(customers, random_sample == TRUE))
test  <- shuffle(subset(customers, random_sample == FALSE))


#Build Decision Trees
#with 'Attrited Customer' defined as +, prioritize sensitivity
#beause it's preferable to have more false +'s than false -'s

#single tree
single_tree <- tree::tree(Attrition_Flag ~ ., data=train)
single_tree_pred <- table(predict(single_tree, test, type = "class"), test$Attrition_Flag)
confusionMatrix(single_tree_pred)

#random forest
random_forest <- randomForest::randomForest(Attrition_Flag ~ ., data=train, ntree=300)
random_forest_pred <- table(predict(random_forest, test, type = "class"), test$Attrition_Flag)
confusionMatrix(random_forest_pred)
#importance(random_forest)

#Adaptive Boosting (AdaBoost)
adaboost <- adabag::boosting(Attrition_Flag ~ ., data=train, mfinal=300)
adaboost_pred_temp <- predict(adaboost, test, type = "class")
adaboost_pred <- table(adaboost_pred_temp$class, test$Attrition_Flag)
confusionMatrix(adaboost_pred)

#Gradient boosting
gradient_boosting <- caret::train(Attrition_Flag ~ ., data = train, method = "gbm", metric="Kappa", verbose=0)
gradient_boosting_pred <- table(predict(gradient_boosting, test), test$Attrition_Flag)
confusionMatrix(gradient_boosting_pred)

#Conclusion
#a single decision tree resulted in an accuracy of .92 and sensitivity of .75
#3 different methods were then tested to try and get an improvement
#AdaBoost had the best result, with an accuracy of .97 and sensitivity of .89
adaboost_final <- adabag::boosting(Attrition_Flag ~ ., data=customers, mfinal=1000) #final model using entire data set
