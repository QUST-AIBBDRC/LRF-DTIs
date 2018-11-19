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
fpr, tpr, _ = roc_curve(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
plt.figure(figsize=(6,5)) 
plt.title('GPCR',fontsize=18)
plt.plot(fpr, tpr, color='blue',
lw=lw, label='Random Forest')
ytest= pd.read_csv('GPCR_lasso_jueceshu.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_GTB[:,0], ytest_GTB[:,2])
#the size of line  
plt.plot(fpr, tpr, color='red',
lw=lw, label='Decision Tree')
ytest= pd.read_csv('GPCR_lasso_luojihuigui.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_GTB[:,0], ytest_GTB[:,1])
#the size of line  
plt.plot(fpr, tpr, color='green',
lw=lw, label='Logistic Regression')
ytest= pd.read_csv('GPCR_lasso_pusubeiyesi.csv') 
ytest_GTB=np.array(ytest,dtype=np.float)
fpr, tpr, _ = roc_curve(ytest_GTB[:,0], ytest_GTB[:,2]) 
plt.plot(fpr, tpr, color='brown',
lw=lw, label='Naïve Bayes')
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 11}
plt.xlabel('False positive rate',fontsize=13)
plt.ylabel('True positive rate',fontsize=13)
legend = plt.legend(prop=font,loc="lower right")
plt.savefig('GPCR_roc.svg',dpi=2000,format='svg')


