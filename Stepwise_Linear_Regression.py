# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:03:27 2018

@author: Arpita
"""

#Operating Profit (EBIT) = Sale - COGS - SGA - Dep & Amort 
#Y(ebit) = Intercept + beta1(sale) - beta2(cogs) - beta3(xsga)-beta3(DP) 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
#%matplotlib inline

df_C = pd.read_csv("C:\\Users\\Arpita\\Desktop\\Book\\Fall 2018\\Machine Learning\\Assignmnet1\\Assignment 2\\compustat_annual_2000_2017_with link information.csv")
#Removed rows with more than 70% missing
df_C = df_C[df_C.columns[df_C.isnull().sum()/df_C.shape[0]<0.7]]

# To get only numeric columns
df_C = df_C._get_numeric_data()
#df_C.select_dtypes(include=np.number).shape
df_C.shape

#finding columns that are not within the 1% and 99% quantile
q1 = df_C.quantile(0.01)
q2 = df_C.quantile(0.99)
IQR = q2-q1
df_C = df_C[~((df_C<(q1-1.5*IQR))|(df_C>(q2+1.5*IQR))).any(axis=1)]

df_C.shape
#Fill empty rows with median value of the column 
df_C = df_C.apply(lambda x:x.fillna(x.median()))

#Define the response and predictor variables 
X = df_C.drop(['ebit'],axis=1)
Y = df_C.loc[:,'ebit']

linM = LinearRegression()
#X2 = X.iloc[:,0:10]
X2 = X
linM.fit(X2,Y)

X2.head()
C_val = 0.05
d={}
popped_cols = []


def stepwiseR(X,Y): 
    c_val = 0.05
    X=sm.add_constant(X,has_constant='add')
    ols1 = sm.OLS(Y,X)
    ols_result1 = ols1.fit()
    if len(ols_result1.pvalues)==2:
        p = ols_result1.pvalues[1]

        if(p<c_val):
            return True,p
        else:     
            return False,0
    else:     
        p = ols_result1.pvalues[-1]
    #print(p)
  
    if(p<c_val):
        p_dict1={}
        for i in range(1,len(ols_result1.pvalues)):
            p_dict1[X.columns[i]]=ols_result1.pvalues[i]
                       
        return True,p,p_dict1
    else:     
        return False,0,{}
        
def First_min(Initial_Iter): 
   
    d={}
    for k,v in Initial_Iter.items():
        if v==min(Initial_Iter.values()):
            d[k]=v
            break    
    return d

def second_min(U,V,d):
    label = min(U,key=U.get)
    for m in list(d.keys()):
        if V[label][m]>0.05:
            d.pop(m)
            popped_cols.append(m)
        
    d[label]=U[label] 
    return d


for j in range(0,(len(X2.columns)-1)):
    if len(d)==0: 
        Initial_Iter={}        
        for i in X2.columns:
            a,b= stepwiseR(X2[[i]],Y)
            if (a==True):
                Initial_Iter[i]=b

        d = First_min(Initial_Iter)
        #print(d)
    else:
        NewCols_P={} 
        PrevCols_P={}

        X_new = X2[list(d.keys())]
        X_rem= X2.drop(popped_cols,axis=1) 
        X_rem=X_rem.drop(X_new.columns, axis=1)

        for i in X_rem.columns: 
            X_2I=pd.concat([X_new,X_rem[i]],axis=1,sort=False)
            xx,yy,zz = stepwiseR(X_2I,Y)
            if (xx==True):
                #print(i)
                NewCols_P[i]=yy
                PrevCols_P[i]=zz
        if NewCols_P:
            d= second_min(NewCols_P,PrevCols_P,d)
        else: 
            break
print(d)    

# Significant X columns
X = X2[list(d.keys())]

# Splitting the dataset into the Training set and Test set
X=sm.add_constant(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

linM.fit(X_train, Y_train)         #fit regressor to training set

# Predicting the Test set results
y_pred = linM.predict(X_test)
y_pred=pd.DataFrame(y_pred)
y_pred.columns = ['Predicted']

print('Coefficients: \n', linM.coef_)
print('Variance score: %.2f' % round(r2_score(Y_test,y_pred)*100,2))









