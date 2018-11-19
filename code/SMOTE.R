install.packages("DMwR")
library(DMwR)
#xx_lasso is the matrix after feature selection
mydata_smote<-data.frame(yy,xx_lasso)
mydata_smote$yy<-factor(mydata_smote$yy)
newData <- SMOTE(yy ~ ., mydata_smote, perc.over = 500,perc.under=120)  

