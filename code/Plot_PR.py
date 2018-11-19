import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
import matplotlib as mpl
import pandas as pd
lw=1
mpl.rcParams['font.sans-serif']='Times New Roman'
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ytest= pd.read_csv('GPCR_lasso_suijisenlin.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = precision_recall_curve(ytest_GTB[:,0], ytest_GTB[:,2])
aupr1=average_precision_score(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line 
plt.figure(figsize=(6,5)) 
plt.title('GPCR',fontsize=18)
plt.plot(tpr, fpr, color='blue',
lw=lw, label='Random Forest')
ytest= pd.read_csv('GPCR_lasso_jueceshu.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = precision_recall_curve(ytest_GTB[:,0], ytest_GTB[:,2])
aupr2=average_precision_score(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
plt.plot(tpr, fpr, color='red',
lw=lw, label='Decision Tree')
ytest= pd.read_csv('GPCR_lasso_luojihuigui.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = precision_recall_curve(ytest_GTB[:,0], ytest_GTB[:,1])
aupr4=average_precision_score(ytest_GTB[:,0], ytest_GTB[:,1])
#the size of line  
plt.plot(tpr, fpr, color='green',
lw=lw, label='Logistic Regression')
ytest= pd.read_csv('GPCR_lasso_pusubeiyesi.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = precision_recall_curve(ytest_GTB[:,0], ytest_GTB[:,2])
aupr3=average_precision_score(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
plt.plot(tpr, fpr, color='brown',
lw=lw, label='Naïve Bayes')
plt.xlim([0.2, 1.03])
plt.ylim([0.5, 1.03])
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 11}
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
legend = plt.legend(prop=font,loc="lower left")
plt.savefig('GPCR.svg',dpi=2000,format='svg')