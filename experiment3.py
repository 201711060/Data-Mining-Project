# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:14:36 2018

@author: shail
"""

import pandas as pd

ds = pd.read_csv('C:\Users\shail\Downloads\classification.csv')

X = ds[['E-Government Index']]  
y = ds['group']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set for E-government index: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = clf.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 



X = ds[['Telecommunication Infrastructure Index']]  
y = ds['group']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set for Telecomm Infra index: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = clf.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
X = ds[['Human Capital Index']]  
y = ds['group']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set Humal capital index: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = clf.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
X = ds[['Online Service Index']]  
y = ds['group']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set online service index: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = clf.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
X = ds[['E-Participation Index']]  
y = ds['group']  
from sklearn.cross_validation import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)  
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: E-participation index {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.metrics import classification_report, confusion_matrix  
y_pred = clf.predict(X_test)  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


