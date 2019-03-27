# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:34:55 2019

@author: Osama ML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sbrn
import statsmodels as stat

dtst = pd.read_csv('PFE_RECHARGE_2017M01(33).csv')


X = dtst.iloc[:,[-2,-1]].values
y = dtst.iloc[:, 2].values



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.30,random_state=0)

from sklearn.preprocessing import StandardScaler

stdSc = StandardScaler()
X_train = stdSc.fit_transform(X_train) 
X_test  = stdSc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classification = DecisionTreeClassifier(
        criterion ='entropy',random_state=0)
classification.fit(X_train,y_train)

y_prediction = classification.predict(X_test)


from sklearn.metrics import confusion_matrix
mat_conf=confusion_matrix(y_test,y_prediction)

from matplotlib.colors import ListedColormap
X_set , y_set = X_train , y_train

X1 , X2 = np.meshgrid(np.arange(start= X_set[:,-1].min() - 1, stop = X_set[:,-1].max() + 1 ,step =0.01),
                      np.arange(start= X_set[:,-2].min() - 1, stop = X_set[:,-2].max() + 1 ,step =0.01))

mplt.contourf(X1 , X2, classification.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
alpha =0.25 , cmap=ListedColormap(('yellow','blue')))

mplt.xlim(X1.min(),X1.max())
mplt.xlim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    mplt.scatter(X_set[y_set== j, 1],X_set[y_set == j, 0],c = ListedColormap(('red', 'black'))(i), label =j)
                 
    
    mplt.title('Classification de l arbre de decision')
    mplt.xlabel('montant bonus')
    mplt.ylabel('montant recharge')
    mplt.legend()
    mplt.plot()