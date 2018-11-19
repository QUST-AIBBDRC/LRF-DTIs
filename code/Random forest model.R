#5-fold cross-validation
library(caret)
folds <- createFolds(newData$yy,k=5)
#Random forest
library(randomForest)
sum<-0
sum1<-0
sum2<-0
sum3<-0
for(i in 1:5)
{fold_test <- newData[folds[[i]],]     #Take folds[[i]] as a test set  
fold_train <- newData[-folds[[i]],]    #The remaining data is used as a training set
print(i)         #i stands for group number
m<-randomForest(fold_train[,-1],fold_train$yy,ntree=500)
p<-predict(m,fold_test[,-1],type="response")
p1<-predict(m,fold_test[,-1],type="prob")
duibi<-data.frame(prob=p,obs=fold_test$yy)
library(caret)
jieguo<-confusionMatrix(duibi$prob,duibi$obs,positive = "1")
print(jieguo)
sum<-jieguo$overall[1]+sum
average<-sum/5
sum1<-jieguo$byClass[1]+sum1
average1<-sum1/5
sum2<-jieguo$byClass[2]+sum2
average2<-sum2/5
p11=data.frame(p1)
roc_results=data.frame(fold_test$yy,p11)
library(ROCR)
pred<-prediction(predictions = roc_results$X1,labels = roc_results$fold_test.yy)
perf<-performance(pred,measure = "tpr",x.measure = "fpr")
plot(perf,main="Lasso+Random forest",col="blue",lwd=3)
abline(a=0,b=1,lty=2)
perf.auc<-performance(pred,measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
sum3<-as.numeric(perf.auc@y.values)+sum3
average3<-sum3/5
}
print(average)    #The value of ACC
print(average1)   #The value of SE
print(average2)   #The value of SP
print(average3)   #The value of AUC