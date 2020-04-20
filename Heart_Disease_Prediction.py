#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:23:28 2020

@author: amankumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from warnings import simplefilter
simplefilter('ignore', category=Warning)


ds=pd.read_csv('processed.cleveland.csv')
ds['oldpeak']=ds['oldpeak'].astype(int)
ds=ds.replace('?','nan')
ds['num']=ds.num.map({0:0,1:1,2:1,3:1,4:1})



X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
X=imputer.fit_transform(X)



from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X=scale.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='lbfgs')
classifier.fit(X_train,y_train)

pred=classifier.predict(X_test)

logistic_acc=0
for i in range(len(X_test)):
    if pred[i]==y_test[i]:
        logistic_acc+=1

print("Accuracy of Logistic Regression = ",logistic_acc/len(X_test))

#Support Vector Machine
from sklearn.svm import SVC
svc_classifier=SVC(kernel='rbf',gamma='scale')
svc_classifier.fit(X_train,y_train)

pred_svc=svc_classifier.predict(X_test)

svm_acc=0
for i in range(len(X_test)):
    if pred_svc[i]==y_test[i]:
        svm_acc+=1

print("Accuracy of Support Vector Machine = ",svm_acc/len(X_test))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)

pred_knn=knn_classifier.predict(X_test)
knn_acc=0
for i in range(len(X_test)):
    if pred_knn[i]==y_test[i]:
        knn_acc+=1

print("Accuracy of K Nearest Neigbour = ",knn_acc/len(X_test))

#Naive Baiyes
from sklearn.naive_bayes import GaussianNB
nb_classifier=GaussianNB()
nb_classifier.fit(X_train,y_train)

pred_nb=nb_classifier.predict(X_test)
nb_acc=0
for i in range(len(X_test)):
    if pred_nb[i]==y_test[i]:
        nb_acc+=1

print("Accuracy of Naive Bayes = ",nb_acc/len(X_test))


#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc_classifier=DecisionTreeClassifier(criterion='gini')
dtc_classifier.fit(X_train,y_train)

pred_dtc=dtc_classifier.predict(X_test)
dtc_acc=0
for i in range(len(X_test)):
    if pred_dtc[i]==y_test[i]:
        dtc_acc+=1

print("Accuracy of Decision Tree Classifier = ",dtc_acc/len(X_test))


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc_classifier=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc_classifier.fit(X_train,y_train)

pred_rfc=rfc_classifier.predict(X_test)
rfc_acc=0
for i in range(len(X_test)):
    if pred_rfc[i]==y_test[i]:
        rfc_acc+=1

print("Accuracy of Random Forest Classifier = ",rfc_acc/len(X_test))


'''
Since the best accuracy among all the classifier has been given by Support Vector Machine and K nearest negbour.
So for prediction we should consider any of the above two algorithm.
'''

