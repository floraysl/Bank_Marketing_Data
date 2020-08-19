library(formattable)
library(MASS)
library(ElemStatLearn)
library(e1071)
library(kernlab)
library(randomForest)
library(gbm)
library(plyr)
library(glmnet)
library(ggplot2)
library(party)
library(rpart)
library(tree)
setwd("~/Desktop/5330- Data Mining  2/miniproject")
traindf <- read.csv("BankMarketing_training.csv")
testdf <- read.csv("BankMarketing_testing.csv")

set.seed(1)
#==========================Exploratory Data Analysis =======================
#=========(a) Response Variable 
#yes and no percentage in training dataset 
percent(table(traindf$y)['yes']/nrow(traindf))
percent(table(traindf$y)['no']/nrow(traindf))
#yes and no percentage in testing dataset 
percent(table(testdf$y)['yes']/nrow(testdf))
percent(table(testdf$y)['no']/nrow(testdf))
#=========(b) Predictor

#===age 
ggplot(data=traindf,aes(x=age)) + 
  geom_bar(alpha=0.75,fill="tomato",color="black") +
  ggtitle("Age Distribution") + 
  theme_bw()
#summary
summary(traindf$age)
#mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
getmode(traindf$age)


#===Job
#summary
summary(traindf$job)
#mode
getmode(traindf$job)


#Data Cleaning --> change resposne yes--> 1/ no--> 0
traindf<-traindf[,-1]
testdf<-testdf[,-1]
traindf$y<- revalue(traindf$y, c("yes"=1))
traindf$y<- revalue(traindf$y, c("no"=0))
#traindf$y<-as.integer(traindf$y)
testdf$y<- revalue(testdf$y, c("yes"=1))
testdf$y<- revalue(testdf$y, c("no"=0))
#testdf$y<-as.integer(testdf$y)
#======================Modeling and Data Analysis============
#=======(a) LDA & QDA
#====LDA 
LDA <- lda(traindf[ ,2:21], traindf[ ,22])
LDA.pred <- predict(LDA, testdf[ ,2:21])$class
LDA.result = table(LDA.pred, testdf$y)
#====QDA 
QDA<- qda(traindf[ ,2:21], traindf[ ,22])
QDA.pred <- predict(QDA, testdf[ ,2:21])$class
QDA.result = table(QDA.pred, testdf$y)


#=======(b) Logistic regression and Linear SVM 
#====Logistic regression 
lg <- glm(y ~ ., data = traindf, family = binomial)
round(summary(lg)$coef, dig=3)
lg.pred <- predict(lg, testdf[ ,1:20]) > 0.5
lg.result <- table(lg.pred, testdf[ ,21])
#lg.pred   0   1
#    FALSE 909  87
#    TRUE   32 508
print(paste(' -Accuracy of the model is ', percent((lg.result[1,1] + lg.result[2,2])/sum(lg.result))))
print(paste(' -Sensitivity of the model is ', percent((lg.result[2,2])/sum(lg.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((lg.result[1,1])/sum(lg.result[ ,1]))))

#====Linear SVM

svm <- svm(y ~ ., data = traindf, type='C-classification', kernel='linear',scale=FALSE, cost = 10000)
svm.pred <- predict(svm, testdf[ ,1:20])
svm.result <- table(svm.pred, testdf[ ,21])
print('Linear SVM results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((svm.result[1,1] + svm.result[2,2])/sum(svm.result))))
print(paste(' -Sensitivity of the model is ', percent((svm.result[2,2])/sum(svm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((svm.result[1,1])/sum(svm.result[ ,1]))))

#========(c) Non-linear SVM 
nsvm <- ksvm(y ~., data = traindf, kernel = 'rbfdot')
nsvm.pred <- predict(nsvm, testdf[ ,1:20]) 
nsvm.result <- table(nsvm.pred, testdf[, 21])
print(paste(' -Accuracy of the model is ', percent((nsvm.result[1,1] + nsvm.result[2,2])/sum(nsvm.result))))
print(paste(' -Sensitivity of the model is ', percent((nsvm.result[2,2])/sum(nsvm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((nsvm.result[1,1])/sum(nsvm.result[ ,1]))))


#=======(e) Contiunous 
train_con<-traindf[,c(1,11:14,16:21)]
test_con<-testdf[,c(1,11:14,16:21)]
#====LDA 
LDA.con <- lda(train_con[ ,1:10], train_con[ ,11])
LDA.con.pred <- predict(LDA.con, test_con[ ,1:10])$class
LDA.con.result = table(LDA.con.pred, test_con$y)
print(paste(' -Accuracy of the model is ', percent((LDA.con.result[1,1] + LDA.con.result[2,2])/sum(LDA.con.result))))
print(paste(' -Sensitivity of the model is ', percent((LDA.con.result[2,2])/sum(LDA.con.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((LDA.con.result[1,1])/sum(LDA.con.result[ ,1]))))

#====QDA 
QDA.con<- qda(train_con[ ,1:10], train_con[ ,11])
QDA.con.pred <- predict(QDA.con, test_con[ ,1:10])$class
QDA.con.result = table(QDA.con.pred, test_con$y)
print(paste(' -Accuracy of the model is ', percent((QDA.con.result[1,1] + QDA.con.result[2,2])/sum(QDA.con.result))))
print(paste(' -Sensitivity of the model is ', percent((QDA.con.result[2,2])/sum(QDA.con.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((QDA.con.result[1,1])/sum(QDA.con.result[ ,1]))))

#====Logistic regression 
lg_con <- glm(y ~ ., data = train_con, family = binomial)
round(summary(lg_con)$coef, dig=3)
lg.con.pred <- predict(lg_con, test_con[ ,1:10]) > 0.5
lg.con.result <- table(lg.con.pred, test_con[ ,11])
print(paste(' -Accuracy of the model is ', percent((lg.con.result[1,1] + lg.con.result[2,2])/sum(lg.con.result))))
print(paste(' -Sensitivity of the model is ', percent((lg.con.result[2,2])/sum(lg.con.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((lg.con.result[1,1])/sum(lg.con.result[ ,1]))))

#====Linear SVM
svm.con <- svm(y ~ ., data = train_con, type='C-classification', kernel='linear',scale=FALSE, cost = 10000)
svm.con.pred <- predict(svm.con, test_con[ ,1:10])
svm.con.result <- table(svm.con.pred, test_con[ ,11])
print(paste(' -Accuracy of the model is ', percent((svm.con.result[1,1] + svm.con.result[2,2])/sum(svm.con.result))))
print(paste(' -Sensitivity of the model is ', percent((svm.con.result[2,2])/sum(svm.con.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((svm.con.result[1,1])/sum(svm.con.result[ ,1]))))


#===============(f) random prediction
testdf$rand = rbinom(nrow(testdf), 1, 0.5)
rand.result<- table(testdf$rand, testdf[, 21])
print(paste(' -Accuracy of the model is ', percent((rand.result[1,1] + rand.result[2,2])/sum(rand.result))))
print(paste(' -Sensitivity of the model is ', percent((rand.result[2,2])/sum(rand.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((rand.result[1,1])/sum(rand.result[ ,1]))))


#==============(g) random forest 
rf<- randomForest(as.factor(y) ~ ., data = traindf, ntree = 500, mtry = 7, nodesize = 10, importance = TRUE, cv=5)
rf.pred<- predict(rf, testdf[ ,1:20])
rf.result <- table(rf.pred, testdf$y)
print(paste(' -Accuracy of the model is ', percent((rf.result[1,1] + rf.result[2,2])/sum(rf.result))))
print(paste(' -Sensitivity of the model is ', percent((rf.result[2,2])/sum(rf.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((rf.result[1,1])/sum(rf.result[ ,1]))))

#=============(h) importance 
barplot(importance(rf)[,3])

#=============(i)Boosting 
gbm.fit = gbm(y~., data = traindf, 
              distribution="adaboost", n.trees= 1000, shrinkage=0.1, bag.fraction=0.8, cv.folds=5)
gbm.pred<- predict(gbm.fit, testdf[ ,1:20])
gbm.result <- table(gbm.pred, testdf$y)
print(paste(' -Accuracy of the model is ', percent((gbm.result[1,1] + gbm.result[2,2])/sum(gbm.result))))
print(paste(' -Sensitivity of the model is ', percent((gbm.result[2,2])/sum(gbm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((gbm.result[1,1])/sum(gbm.result[ ,1]))))
