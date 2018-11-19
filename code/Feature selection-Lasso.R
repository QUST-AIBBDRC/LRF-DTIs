install.packages("HDCI")
library(HDCI)
#xx_matrix is a matrix of eigenwector,yy_matrix is a label of sample
Lasso<-Lasso(xx_matrix,yy_matrix,lambda=0.00005,fix.lambda=TRUE,cv.OLS=FALSE)
c<-colnames(xx_matrix[,which(Lasso$beta!=0)])
xx_lasso<-xx_matrix[,which(Lasso$beta!=0)]    #xx_lasso is a matrix of feature selection

